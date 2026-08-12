"""Train the inpainting baseline (MONAI + Lightning).

  python src/segresnet/train.py --data-root <raw> --train-ids splits/train_ids.txt \
      --val-ids splits/val_small_ids.txt --out-dir runs/unet_baseline

--overfit N : sanity check on N fixed samples (no aug); val_l1 should crater toward ~0.
"""
import argparse
from datetime import timedelta
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from data import InpaintCrops, InpaintWinner, read_ids
from litmod import LitInpaint


class EMA(pl.Callback):
    """Exponential moving average of the net weights.

    Strategy (avoids fragile callback-ordering with ModelCheckpoint): keep EMA weights loaded
    through the whole validation phase and the gap before the next train epoch, then restore the
    live weights at the start of the next train epoch. So validation metrics AND the checkpoint
    saved at on_validation_end both use EMA weights (consistent selection), while training always
    resumes from the live weights. Inference is unchanged — best.ckpt already holds EMA weights.
    """
    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.shadow = None
        self.backup = None

    def on_fit_start(self, trainer, pl_module):
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
    ap.add_argument("--data-root", required=True, help="root for val (and train unless --train-root)")
    ap.add_argument("--train-root", default=None, help="separate root for training data (e.g. mask-augmented dataset)")
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
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--w-l1", type=float, default=1.0)
    ap.add_argument("--w-mse", type=float, default=1.0)
    ap.add_argument("--w-ssim", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.0, help=">0: EMA decay (e.g. 0.999)")
    ap.add_argument("--arch", default="basicunet",
                    choices=["basicunet", "unet_w", "attn", "segresnet", "dynunet"])
    ap.add_argument("--ssim-ds", type=int, default=2, help="avg-pool factor before SSIM loss (1=full-res, faithful)")
    ap.add_argument("--data-style", default="ours", choices=["ours", "winner"],
                    help="winner: clip/max [-1,1] norm, void-only-healthy, enumerate all masks")
    ap.add_argument("--aug-flips", action="store_true", help="winner data: on-the-fly flips + intensity jitter")
    ap.add_argument("--match-infer-void", action="store_true",
                    help="winner data: void UNION + union mask channel (match inference; fix train/infer mismatch)")
    ap.add_argument("--weight-decay", type=float, default=0.0, help=">0: AdamW weight decay (regularize)")
    ap.add_argument("--init-ckpt", default=None, help="warm-start net weights from this checkpoint (continue training)")
    ap.add_argument("--ckpt-monitor", default="val_l1", choices=["val_l1", "val_ssim"],
                    help="checkpoint selection metric (val_ssim = pick best SSIM)")
    ap.add_argument("--lr-schedule", choices=["cosine"], default=None,
                    help="cosine anneal LR->0 over --epochs (else constant LR)")
    ap.add_argument("--merged-donor-npz", default=None,
                    help="per-brain merged-retrieval donor npz -> paste into void as a 3rd input channel (train)")
    ap.add_argument("--val-prior-root", default=None,
                    help="frozen native merged-prior dir for val ({name}-t1n-inference.nii.gz)")
    ap.add_argument("--hcp-root", default=None, help="HCP-in-BraTS dir to pool (on-the-fly masking)")
    ap.add_argument("--hcp-ids", default=None, help="HCP id list file (default: all HCP-* in --hcp-root)")
    ap.add_argument("--hcp-pool-cache", default="data/hcp_shape_pool.pkl", help="tumor-shape pool cache")
    ap.add_argument("--mask-aug-p", type=float, default=0.0,
                    help="prob. of replacing a GLI decoy with a fresh synthetic healthy void (winner mask-aug)")
    ap.add_argument("--mask-aug-size", type=int, nargs=2, default=[4000, 45000],
                    help="synthetic-void volume range (log-uniform); small to match val decoys")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tr_ids, va_ids, aug = read_ids(args.train_ids), read_ids(args.val_ids), True
    if args.overfit:
        base = [i for i in tr_ids if i.strip()][:args.overfit]
        va_ids, aug = base, False
        tr_ids = base * 250                       # many steps/epoch (cache makes repeats cheap)
        args.batch, args.workers = min(args.batch, len(base)), 0
        print("OVERFIT MODE:", base)

    cache = bool(args.overfit)   # cache only for the tiny overfit set (bounded memory)
    train_root = args.train_root or args.data_root
    # shared tumor-shape sampler for HCP pooling and/or GLI mask-augmentation (built once)
    sampler = None
    if args.hcp_root or args.mask_aug_p > 0:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data"))
        from mask_sampler import load_or_build_pool, MaskSampler
        pool = load_or_build_pool(args.hcp_pool_cache, brats_root=args.data_root)  # real BraTS shapes
        sampler = MaskSampler(pool)
    if args.data_style == "winner":
        tr = InpaintWinner(train_root, tr_ids, crop=args.crop, augment=args.aug_flips,
                           match_infer=args.match_infer_void,
                           synth_sampler=(sampler if args.mask_aug_p > 0 else None),
                           synth_p=args.mask_aug_p, synth_size=tuple(args.mask_aug_size),
                           merged_donor_npz=args.merged_donor_npz)
        va = InpaintWinner(args.data_root, va_ids, crop=args.crop, augment=False,
                           match_infer=args.match_infer_void,   # val uses official decoys (no aug)
                           prior_root=args.val_prior_root)
    else:
        tr = InpaintCrops(train_root, tr_ids, crop=args.crop, augment=aug, cache=cache)
        va = InpaintCrops(args.data_root, va_ids, crop=args.crop, augment=False, cache=cache)
    if args.mask_aug_p > 0:
        print(f"GLI mask-aug: p={args.mask_aug_p} synthetic healthy voids, size {tuple(args.mask_aug_size)}")
    if args.hcp_root:                              # pool healthy HCP brains (on-the-fly masking)
        from data import HCPOnTheFly
        from torch.utils.data import ConcatDataset
        hcp_ids = (read_ids(args.hcp_ids) if args.hcp_ids
                   else [p.name for p in sorted(Path(args.hcp_root).glob("HCP-*"))])
        hcp = HCPOnTheFly(args.hcp_root, hcp_ids, sampler, crop=args.crop,
                          augment=args.aug_flips, style=args.data_style,
                          merged_donor_npz=args.merged_donor_npz)
        tr = ConcatDataset([tr, hcp])
        print(f"pooled HCP: +{len(hcp)} healthy brains (shape pool {len(pool['pool'])} tumors)")
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    vl = DataLoader(va, batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    print(f"train {len(tr)}  val {len(va)}  steps/epoch {len(tl)}")

    data_range = 2.0 if args.data_style == "winner" else 1.0   # SSIM range: [-1,1] vs [0,1]
    in_channels = 3 if (args.merged_donor_npz or args.val_prior_root) else 2
    model = LitInpaint(lr=args.lr, w_l1=args.w_l1, w_mse=args.w_mse, w_ssim=args.w_ssim,
                       ssim_ds=args.ssim_ds, arch=args.arch, roi=tuple(args.crop),
                       data_range=data_range, weight_decay=args.weight_decay,
                       lr_schedule=args.lr_schedule, in_channels=in_channels)
    if args.init_ckpt:                              # warm-start weights to continue training
        import torch as _t
        sd = _t.load(args.init_ckpt, map_location="cpu", weights_only=False)["state_dict"]
        # zero-init: if warm-starting a 2ch model into a 3ch one, copy the existing
        # conv-in weights and zero-fill the new prior channel (starts identical to baseline).
        msd = model.state_dict()
        for k, w in list(sd.items()):
            if k in msd and w.dim() == 5 and msd[k].shape[1] > w.shape[1]:
                nw = _t.zeros_like(msd[k]); nw[:, :w.shape[1]] = w
                sd[k] = nw
                print(f"conv-in expanded {w.shape[1]}->{msd[k].shape[1]} (new prior channel zero-init): {k}")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"warm-start from {args.init_ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    mon, mode = ("val_ssim", "max") if args.ckpt_monitor == "val_ssim" else ("val_l1", "min")
    ckpt = ModelCheckpoint(dirpath=out, monitor=mon, mode=mode,
                           save_top_k=1, save_last=True, filename="best")
    print(f"checkpoint selection: {mon} ({mode})")
    callbacks = [ckpt]
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
    print(f"best ckpt: {ckpt.best_model_path}  best val_l1: {float(ckpt.best_model_score or 0):.4f}")


if __name__ == "__main__":
    main()
