"""Train the inpainting baseline (MONAI + Lightning).

  python src/diffusion/train.py --data-root <raw> --train-ids splits/train_ids.txt \
      --val-ids splits/val_small_ids.txt --out-dir runs/unet_baseline

--overfit N : sanity check on N fixed samples (no aug); val_l1 should crater toward ~0.
"""
import argparse
import sys
from datetime import timedelta
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from data import InpaintCrops, read_ids
from litmod import LitInpaint


class EMA(pl.Callback):
    """Exponential moving average of the net weights (ported from Track A — its biggest lever).

    Keep EMA weights loaded through validation + checkpoint (so best.ckpt holds EMA weights, selected
    by EMA-val), restore live weights at the next train epoch. Inference unchanged.
    """
    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.shadow = None
        self.backup = None

    def on_fit_start(self, trainer, pl_module):
        if self.shadow is None:                       # survive warm-start: init from loaded weights
            self.shadow = {k: v.detach().clone().float() for k, v in pl_module.net.state_dict().items()}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, *args):
        d = self.decay
        for k, v in pl_module.net.state_dict().items():
            if torch.is_floating_point(v):
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.shadow[k] = v.detach().clone()

    def on_validation_start(self, trainer, pl_module):
        if self.shadow is None:
            return
        sd = pl_module.net.state_dict()
        self.backup = {k: v.detach().clone() for k, v in sd.items()}
        pl_module.net.load_state_dict({k: self.shadow[k].to(sd[k].dtype) for k in sd})

    def on_train_epoch_start(self, trainer, pl_module):
        if self.backup is not None:
            pl_module.net.load_state_dict(self.backup)
            self.backup = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="raw GLI root: used for val and overfit")
    ap.add_argument("--train-root", nargs="+", default=None,
                    help="train data root(s); defaults to --data-root. Point at GLI-inpainting-x5 "
                         "(mask aug) and/or pooled cohorts (GoAT). Per-sample files auto-expanded.")
    ap.add_argument("--train-ids", required=True)
    ap.add_argument("--val-ids", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--crop", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--init-ckpt", default=None,
                    help="warm-start: load model WEIGHTS from this checkpoint (keeps current hparams "
                         "+ fresh optimizer; not a full Lightning resume). Continue from a trained net.")
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--model", choices=["unet", "diffusion"], default="unet")
    ap.add_argument("--w-l1", type=float, default=1.0)
    ap.add_argument("--w-mse", type=float, default=1.0)
    ap.add_argument("--w-ssim", type=float, default=0.0)
    # diffusion (Track B) only:
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--infer-steps", type=int, default=4)
    ap.add_argument("--n-avg", type=int, default=1)
    ap.add_argument("--w-wave", type=float, default=1.0)
    ap.add_argument("--w-recon", type=float, default=1.0)
    ap.add_argument("--lr-schedule", choices=["cosine"], default=None)
    ap.add_argument("--prior-root", default=None, help="dir of train prior fills {id}{sfx}-t1n-inference.nii.gz -> 3rd conditioning channel (in_channels 24->32)")
    ap.add_argument("--val-prior-root", default=None, help="dir of val prior fills (3rd channel for val)")
    ap.add_argument("--ema", type=float, default=0.0, help=">0: EMA decay (e.g. 0.999) — Track A's top lever")
    ap.add_argument("--flash", action="store_true", help="memory-efficient flash attention (needed for large crops)")
    ap.add_argument("--strong-aug", action="store_true", help="+ intensity(gamma) + small-rotation augmentation")
    ap.add_argument("--weight-decay", type=float, default=0.0, help=">0: AdamW weight decay (regularization)")
    ap.add_argument("--save-every", type=int, default=0, help=">0: snapshot a ckpt every N epochs (mid-training scoring)")
    # data mixing (pool extra cohorts on top of --train-root):
    ap.add_argument("--goat-root", default=None, help="oversample pre-gen items under this root")
    ap.add_argument("--goat-oversample", type=int, default=1, help="replicate GoAT items N× (higher representation)")
    ap.add_argument("--hcp-root", default=None, help="HCP-in-BraTS root → on-the-fly healthy masks")
    ap.add_argument("--hcp-oversample", type=int, default=1, help="N fresh on-the-fly masks per HCP brain/epoch")
    ap.add_argument("--hcp-pool-cache", default="runs/_shape_pool.pkl", help="cached real-tumor shape pool")
    ap.add_argument("--gli-otf-donor", default=None,
                    help="npz of per-GLI-brain merged donor (eda/gen_merged_donor.py) -> GLI on-the-fly "
                         "fresh-mask training (mask augmentation) with the merged prior")
    ap.add_argument("--gli-ids", default="splits/train_ids.txt", help="GLI brain ids for --gli-otf-donor")
    ap.add_argument("--gli-oversample", type=int, default=5, help="N fresh on-the-fly decoys per GLI brain/epoch")
    ap.add_argument("--hcp-sigma", type=float, default=4.0,
                    help="Gaussian blur sigma for on-the-fly HCP prior; 0 = unblurred (merged prior)")
    ap.add_argument("--hcp-retr-donor", default=None,
                    help="npz of per-HCP-brain best donor (eda/gen_hcp_best_donor.py) → on-the-fly HCP "
                         "emits a LIVE blurred-retrieval prior (3ch), poolable with a prior model's GLI items")
    ap.add_argument("--decoy-large-frac", type=float, default=0.0,
                    help="on-the-fly HCP: fraction of healthy decoys upsized (target = max of 2 natural "
                         "size draws), gently fattening the Q4 big-hole tail; 0 = natural, ~0.3 = slight")
    # Stage-1 coarse-to-fine (whole downsampled brain; no crop, no prior):
    ap.add_argument("--stage1", action="store_true",
                    help="Stage-1 coarse model: train LitDiffusion(24ch) on WHOLE downsampled volumes")
    ap.add_argument("--low-shape", type=int, nargs=3, default=[96, 96, 64],
                    help="Stage-1 low resolution (each axis MUST be divisible by 16)")
    ap.add_argument("--pad-shape", type=int, nargs=3, default=[240, 240, 160], help="Stage-1 pre-downsample pad shape")
    ap.add_argument("--ds-root", default=None, help="Stage-1: dir of precomputed {name}{sfx}.npz (fast dataloading)")
    ap.add_argument("--val-ds-root", default=None, help="Stage-1: precomputed val npz dir")
    # On-the-fly BIG-hole masks (target the large-decoy score cap; fresh mask each epoch):
    ap.add_argument("--onthefly-bighole", action="store_true",
                    help="train on fresh on-the-fly masks with a big-biased healthy decoy (OnTheFlyBigHole)")
    ap.add_argument("--size-range", type=float, nargs=2, default=[25000, 120000],
                    help="healthy-decoy target-volume range (log-uniform); natural median ~17k")
    ap.add_argument("--onthefly-oversample", type=int, nargs="+", default=[5],
                    help="items/epoch per --train-root (single value = all roots); e.g. 5 40 5 for GLI/GoAT/HCP")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tr_ids, va_ids, aug = read_ids(args.train_ids), read_ids(args.val_ids), True
    train_root = args.train_root or args.data_root
    if args.overfit:
        base = [i for i in tr_ids if i.strip()][:args.overfit]
        va_ids, tr_ids, aug = base, base, False
        train_root = args.data_root               # overfit from the raw single-sample data
        args.batch, args.workers = min(args.batch, len(base)), 0
        print("OVERFIT MODE:", base)

    cache = bool(args.overfit)   # cache only for the tiny overfit set (bounded memory)
    if args.onthefly_bighole:    # fresh big-biased healthy decoys each epoch (target large-decoy cap)
        from data import OnTheFlyBigHole             # InpaintCrops is imported at module level
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data"))
        from mask_sampler import load_or_build_pool, MaskSampler
        sampler = MaskSampler(load_or_build_pool(args.hcp_pool_cache, brats_root=args.data_root))
        roots = args.train_root or [args.data_root]
        ovs = args.onthefly_oversample
        ovs = ovs * len(roots) if len(ovs) == 1 else ovs
        keep = set(i.strip() for i in tr_ids if i.strip())
        items = []
        for root, k in zip(roots, ovs):
            rp = Path(root)
            names = [p.name for p in sorted(rp.glob("*"))
                     if p.is_dir() and p.name in keep and (rp / p.name / f"{p.name}-t1n.nii.gz").exists()]
            items += [(rp / n, n) for n in names] * k
        tr = OnTheFlyBigHole(items, sampler, crop=args.crop, size_range=tuple(args.size_range), augment=aug)
        va = InpaintCrops(args.data_root, va_ids, crop=args.crop, augment=False)
        print(f"OnTheFlyBigHole: {len(items)} items/epoch, {len(roots)} roots ovs={ovs}, size_range={args.size_range}")
    elif args.stage1:            # Stage-1: whole downsampled brain (no crop, no prior)
        from data import DownsampledWhole
        tr = DownsampledWhole(train_root, tr_ids, low_shape=args.low_shape, pad_shape=args.pad_shape,
                              augment=aug, cache=cache, strong_aug=args.strong_aug, ds_root=args.ds_root)
        va = DownsampledWhole(args.data_root, va_ids, low_shape=args.low_shape, pad_shape=args.pad_shape,
                              augment=False, cache=cache, ds_root=args.val_ds_root)
    else:
        tr = InpaintCrops(train_root, tr_ids, crop=args.crop, augment=aug, cache=cache, prior_root=args.prior_root,
                          strong_aug=args.strong_aug)
        va = InpaintCrops(args.data_root, va_ids, crop=args.crop, augment=False, cache=cache,
                          prior_root=args.val_prior_root)
    if args.overfit:
        tr.items = tr.items * 250                  # many steps/epoch (cache makes repeats cheap)
    if args.goat_root and args.goat_oversample > 1:   # higher GoAT representation (pre-gen masks)
        groot = str(Path(args.goat_root))
        extra = [it for it in tr.items if str(it[0]).startswith(groot)]
        tr.items = tr.items + extra * (args.goat_oversample - 1)
        print(f"GoAT oversample ×{args.goat_oversample}: {len(extra)} items → +{len(extra)*(args.goat_oversample-1)}")
    if args.hcp_root:                                  # pool HCP healthy brains, on-the-fly masks
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data"))
        from mask_sampler import load_or_build_pool, MaskSampler
        from data import HCPOnTheFly
        from torch.utils.data import ConcatDataset
        pool = load_or_build_pool(args.hcp_pool_cache, brats_root=args.data_root)
        hcp_ids = sorted(p.name for p in Path(args.hcp_root).glob("*") if p.is_dir())
        hcp_ids = hcp_ids * args.hcp_oversample        # each replica draws a FRESH mask → anti-overfit
        if args.hcp_retr_donor:                        # prior model: HCP gets a LIVE blurred-retrieval prior (3ch)
            from data import OnTheFlyRetrHCP
            hcp = OnTheFlyRetrHCP(args.hcp_root, hcp_ids, MaskSampler(pool), args.hcp_retr_donor,
                                  crop=tuple(args.crop), augment=aug, sigma=args.hcp_sigma,
                                  decoy_large_frac=args.decoy_large_frac)
        else:
            hcp = HCPOnTheFly(args.hcp_root, hcp_ids, MaskSampler(pool), crop=tuple(args.crop), augment=aug,
                              strong_aug=args.strong_aug, decoy_large_frac=args.decoy_large_frac)
        tr = ConcatDataset([tr, hcp])
        print(f"pooled HCP on-the-fly: +{len(hcp)} (×{args.hcp_oversample} fresh masks/brain/epoch"
              f"{', +live retr-prior' if args.hcp_retr_donor else ''}, decoy_large_frac={args.decoy_large_frac})")
    if args.gli_otf_donor:                              # GLI mask-augmentation: fresh decoys + merged prior
        from data import OnTheFlyGLI
        from mask_sampler import load_or_build_pool, MaskSampler
        gpool = load_or_build_pool(args.hcp_pool_cache, brats_root=args.data_root)
        gli_ids = [l.strip() for l in open(args.gli_ids) if l.strip()]
        gli_ids = gli_ids * args.gli_oversample          # each replica draws a FRESH decoy -> mask aug
        gli = OnTheFlyGLI(args.data_root, gli_ids, MaskSampler(gpool), args.gli_otf_donor,
                          crop=tuple(args.crop), augment=aug, sigma=args.hcp_sigma)
        tr = ConcatDataset([tr, gli])
        print(f"pooled GLI on-the-fly (mask-aug): +{len(gli)} fresh healthy decoys/epoch, sigma={args.hcp_sigma}")
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    vl = DataLoader(va, batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    print(f"train {len(tr)}  val {len(va)}  steps/epoch {len(tl)}")

    in_channels = 32 if (args.prior_root or args.val_prior_root) else 24   # +8 dwt(prior) conditioner
    if args.model == "diffusion":
        from litdiff import LitDiffusion
        model = LitDiffusion(lr=args.lr, num_train_timesteps=args.timesteps,
                             infer_steps=args.infer_steps, n_avg=args.n_avg,
                             w_wave=args.w_wave, w_recon=args.w_recon, w_ssim=args.w_ssim,
                             lr_schedule=args.lr_schedule, in_channels=in_channels,
                             use_flash_attention=args.flash, weight_decay=args.weight_decay)
        monitor = "val_loss"
    else:
        model = LitInpaint(lr=args.lr, w_l1=args.w_l1, w_mse=args.w_mse, w_ssim=args.w_ssim)
        monitor = "val_l1"
    if args.init_ckpt:
        sd = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)["state_dict"]
        msd = model.state_dict()
        for k in list(sd):                         # zero-init: expand conv-in, new prior channels = 0
            if k in msd and sd[k].shape != msd[k].shape:
                w, t = sd[k], msd[k]
                if w.dim() == 5 and w.shape[0] == t.shape[0] and w.shape[2:] == t.shape[2:] and t.shape[1] > w.shape[1]:
                    nw = torch.zeros_like(t); nw[:, :w.shape[1]] = w   # copy voided+mask conditioners; prior ch = 0
                    sd[k] = nw
                    print(f"conv-in expanded {w.shape[1]}->{t.shape[1]} (new prior channels zero-init): {k}")
                else:
                    del sd[k]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"warm-start {args.init_ckpt}: loaded; missing={len(missing)} unexpected={len(unexpected)}")
    ckpt = ModelCheckpoint(dirpath=out, monitor=monitor, mode="min",
                           save_top_k=1, save_last=True, filename="best")
    callbacks = [ckpt]
    if args.save_every > 0:                        # periodic snapshots for mid-training real-metric scoring
        snap = ModelCheckpoint(dirpath=str(out / "snaps"), every_n_epochs=args.save_every,
                               save_top_k=-1, filename="ep{epoch:03d}")
        callbacks.append(snap)
        print(f"snapshot every {args.save_every} epochs -> {out}/snaps")
    if args.ema > 0:
        callbacks.append(EMA(decay=args.ema))
        print(f"EMA enabled, decay={args.ema}")
    trainer = pl.Trainer(
        accelerator="gpu", devices=args.devices,
        strategy="ddp" if args.devices > 1 else "auto",
        precision=args.precision, max_epochs=args.epochs,
        max_time=timedelta(minutes=args.max_minutes) if args.max_minutes > 0 else None,
        default_root_dir=str(out), callbacks=callbacks,
        logger=CSVLogger(save_dir=str(out), name="csv"),
        log_every_n_steps=10, num_sanity_val_steps=0, enable_progress_bar=False,
    )
    trainer.fit(model, tl, vl)
    print(f"best ckpt: {ckpt.best_model_path}  best {monitor}: {float(ckpt.best_model_score or 0):.4f}")


if __name__ == "__main__":
    main()
