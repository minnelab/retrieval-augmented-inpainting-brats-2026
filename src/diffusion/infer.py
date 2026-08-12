"""Inference on a split with MONAI sliding-window over the full volume.

Saves {case}-t1n-inference.nii.gz, composited (result = voided outside the mask, prediction
inside) at original intensity/affine — exactly what the challenge evaluation expects.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from scipy import ndimage
from monai.inferers import sliding_window_inference

from data import load, per_case_scale, centered_bbox, crop_or_pad
from litmod import LitInpaint


_TTA_FLIPS = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]   # 8 axis-flip combos


def _adaptive_n_avg(size_vox, n_max, n_min=1):
    """Per-hole sampling budget from the measured n_avg-by-decoy-size saturation curve (wdm_hcp3 on
    val-150): small holes saturate at ~2, mid at ~4-8, big keep improving to ~16 (32 adds nothing).
    Thresholds are the val-150 size quartile edges. `n_max` = the big-hole tier, `n_min` = a floor for
    ALL holes. So `n_min=1,n_max=8` -> 2/4/8/8 (adapt-DOWN, speed); `n_min=8,n_max=16` -> 8/8/8/16
    (BOOST-only: never under-samples, only bumps big holes -> can slightly beat uniform). Applied per
    component (we can't tell the scored decoy from the unscored tumor at inference)."""
    na = 2 if size_vox < 4654 else 4 if size_vox < 18841 else 8 if size_vox < 51128 else n_max
    return max(n_min, min(na, n_max))


def diffuse_volume(model, voided, mask, prior=None, roi=128, steps=4, n_avg=1, margin=8, cap=192,
                   device=None, tta=False, offset_tta=1, batch_cap=32,
                   adaptive=False, n_avg_max=None, n_avg_min=1):
    """Wavelet-diffusion fill. n_avg + 8-flip TTA + offset-TTA all pull toward the conditional mean and
    are BATCHED into one GPU pass per crop (flips + n_avg). With `adaptive`, the per-hole n_avg is set
    from the hole size (see _adaptive_n_avg) -> less sampling on the many small/easy holes. Composite
    outside mask. `prior`: optional conditioning volumes."""
    device = device or next(model.parameters()).device
    scale = per_case_scale(voided)
    result = voided.copy()
    flips = _TTA_FLIPS if tta else [()]
    skw = {}
    n_avg_max = n_avg_max or n_avg
    labels, n = ndimage.label(mask)
    rng = np.random.default_rng(1234)                       # deterministic offsets
    for i in range(1, n + 1):
        comp = labels == i
        na = _adaptive_n_avg(int(comp.sum()), n_avg_max, n_avg_min) if adaptive else n_avg   # per-hole budget
        coords = np.argwhere(comp)
        lo, hi = coords.min(0), coords.max(0) + 1
        center = (lo + hi) // 2
        size = [min(cap, voided.shape[ax], max(roi, int(np.ceil(((hi - lo)[ax] + 2 * margin) / 16) * 16)))
                for ax in range(3)]
        size = [s - s % 16 for s in size]                   # keep divisible by 16 after clamping
        starts = [centered_bbox(center, size, voided.shape)]   # 1st crop = centered (== offset_tta=1)
        for _ in range(offset_tta - 1):                     # extra crops: jitter start, keep void inside
            sl = []
            for ax in range(3):
                lo_s = max(0, hi[ax] - size[ax])
                hi_s = min(lo[ax], voided.shape[ax] - size[ax])
                s0 = int(rng.integers(lo_s, hi_s + 1)) if hi_s > lo_s else max(0, min(center[ax] - size[ax] // 2, voided.shape[ax] - size[ax]))
                sl.append((s0, s0 + size[ax]))
            starts.append(sl)
        csum = np.zeros(voided.shape, np.float32)           # void voxels are inside EVERY crop -> avg
        for sl in starts:
            v = crop_or_pad(voided, sl, size) / scale
            m = crop_or_pad(comp.astype(np.float32), sl, size)
            pr = [crop_or_pad(p, sl, size) / scale for p in prior] if prior else None
            vb = np.stack([np.flip(v, fl).copy() if fl else v for fl in flips])     # (F, *size) flip-batch
            mb = np.stack([np.flip(m, fl).copy() if fl else m for fl in flips])
            vt = torch.from_numpy(vb)[:, None].to(device)                           # (F,1,D,H,W)
            mt = torch.from_numpy(mb)[:, None].to(device)
            pt = None
            if pr:
                pb = np.stack([np.stack([np.flip(p, fl).copy() if fl else p for p in pr]) for fl in flips])
                pt = torch.from_numpy(pb).to(device)                                # (F,P,D,H,W)
            # sample() flattens the F flips x n_avg trajectories and chunks them at batch_cap, halving
            # the chunk on CUDA OOM -> fast on a big GPU, degrades gracefully on a small one (no crash).
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pred = model.sample(vt, mt, prior=pt, steps=steps, n_avg=na, max_batch=batch_cap, **skw)
            predn = pred[:, 0].float().cpu().numpy()         # (F, *size)
            avg = np.mean([np.flip(predn[j], fl) if fl else predn[j] for j, fl in enumerate(flips)], axis=0)
            src = tuple(slice(s, e) for s, e in sl)
            csum[src] += avg * scale
        result[comp] = (csum[comp] / len(starts)).astype(voided.dtype)
    return result


def inpaint_volume(model, voided, mask, roi=(128, 128, 128), device=None):
    """Run sliding-window inference and composite. Returns the full-resolution result
    (voided outside the mask, prediction inside) at the original intensity scale.

    Device-aware: bf16 autocast on CUDA, plain fp32 on CPU — so the same code path
    runs locally, on a cluster, and inside the submission container.
    """
    device = device or next(model.parameters()).device
    scale = per_case_scale(voided)
    x = torch.from_numpy(np.stack([voided / scale, mask.astype(np.float32)]))[None].to(device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                         enabled=(device.type == "cuda")):
        pred = sliding_window_inference(x, roi_size=tuple(roi), sw_batch_size=4,
                                        predictor=model, overlap=0.25, mode="gaussian")
    pred = pred[0, 0].float().cpu().numpy() * scale
    result = voided.copy()
    result[mask] = pred[mask]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--model", choices=["unet", "diffusion"], default="unet")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--n-avg", type=int, default=1)
    ap.add_argument("--tta", action="store_true", help="8-flip test-time augmentation (diffusion)")
    ap.add_argument("--offset-tta", type=int, default=1, help="N jittered crop positions per void, averaged")
    ap.add_argument("--prior-root", default=None, help="dir of prior fills {id}-t1n-inference.nii.gz -> 3rd conditioning channel")
    ap.add_argument("--batch-cap", type=int, default=32, help="max total batch (flips*n_avg-chunk) per GPU pass; lower if TTA OOMs on big crops")
    ap.add_argument("--adaptive-navg", action="store_true", help="per-hole n_avg by decoy size (fewer samples on small holes)")
    ap.add_argument("--n-avg-max", type=int, default=None, help="big-hole tier under --adaptive-navg (e.g. 16 boosts big holes)")
    ap.add_argument("--n-avg-min", type=int, default=1, help="floor n_avg for all holes under --adaptive-navg (e.g. 8 = boost-only: never under-sample)")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.model == "diffusion":
        from litdiff import LitDiffusion
        # flash attention (memory-efficient SDPA) at inference: math-identical weights, far less memory
        # -> lets us batch n_avg x TTA in one pass on any GPU (avoids the attn-matrix blowup).
        _flash = os.environ.get("FLASH", "1") not in ("0", "false", "False", "")   # FLASH=0 to disable (debug/parity)
        model = LitDiffusion.load_from_checkpoint(args.ckpt, use_flash_attention=_flash).cuda().eval()
        fill = lambda v, m, p: diffuse_volume(model, v, m, prior=p, roi=args.roi[0], steps=args.steps,
                                              n_avg=args.n_avg, tta=args.tta, offset_tta=args.offset_tta,
                                              batch_cap=args.batch_cap,
                                              adaptive=args.adaptive_navg, n_avg_max=args.n_avg_max, n_avg_min=args.n_avg_min)
    else:
        model = LitInpaint.load_from_checkpoint(args.ckpt).cuda().eval()
        fill = lambda v, m, p: inpaint_volume(model, v, m, roi=args.roi)

    prior_roots = [Path(args.prior_root)] if args.prior_root else []
    ids = [i.strip() for i in Path(args.ids).read_text().splitlines() if i.strip()]
    root = Path(args.data_root)
    for k, name in enumerate(ids):
        d = root / name
        vimg = nib.load(str(d / f"{name}-t1n-voided.nii.gz"))
        voided = np.asarray(vimg.get_fdata(), dtype=np.float32)
        mask = load(d / f"{name}-mask.nii.gz") > 0.5
        prior = [load(pr / f"{name}-t1n-inference.nii.gz") for pr in prior_roots] or None

        result = fill(voided, mask, prior)
        nib.save(nib.Nifti1Image(result, vimg.affine, vimg.header),
                 out / f"{name}-t1n-inference.nii.gz")
        if (k + 1) % 20 == 0:
            print(f"{k+1}/{len(ids)}", flush=True)
    print(f"wrote {len(ids)} predictions to {out}")


if __name__ == "__main__":
    main()
