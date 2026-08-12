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
from monai.inferers import sliding_window_inference

from data import load, per_case_scale, winner_norm
from litmod import LitInpaint


# flip-TTA: the 8 axis-flip combinations over the 3 spatial dims (2,3,4) of the (1,2,X,Y,Z) tensor.
_TTA_FLIPS = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]


def inpaint_volume(model, voided, mask, roi=(128, 128, 128), device=None, tta=False, style="ours",
                   prior=None):
    """Run sliding-window inference and composite. Returns the full-resolution result
    (voided outside the mask, prediction inside) at the original intensity scale.

    Device-aware: bf16 autocast on CUDA, plain fp32 on CPU — so the same code path
    runs locally, on a cluster, and inside the submission container.

    tta: average predictions over the 8 axis-flips (flip input, infer, flip back). Matches the
    flip augmentation used in training; pulls toward the mean -> small SSIM/PSNR gain (A3).
    """
    device = device or next(model.parameters()).device
    if style == "winner":
        vn, mx = winner_norm(voided)                      # [-1,1], holes ~ -1
        v_in = vn
    else:
        scale = per_case_scale(voided)
        v_in = voided / scale
    chans = [v_in, mask.astype(np.float32)]
    if prior is not None:                                 # retrieval prior 3rd channel (winner-normed)
        chans.append(winner_norm(prior)[0] if style == "winner" else prior / scale)
    x0 = torch.from_numpy(np.stack(chans))[None].to(device)
    flips = _TTA_FLIPS if tta else [()]
    acc = None
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                         enabled=(device.type == "cuda")):
        for fl in flips:
            x = torch.flip(x0, dims=fl) if fl else x0
            p = sliding_window_inference(x, roi_size=tuple(roi),
                                         sw_batch_size=int(os.environ.get("SW_BATCH", "4")),
                                         predictor=model, overlap=0.25, mode="gaussian")
            p = torch.flip(p, dims=fl) if fl else p
            acc = p if acc is None else acc + p
    pred = (acc / len(flips))[0, 0].float().cpu().numpy()
    if style == "winner":
        pred = (pred + 1) / 2 * mx                         # denorm [-1,1] -> original intensity
    else:
        pred = pred * scale
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
    ap.add_argument("--tta", action="store_true", help="8-flip test-time augmentation")
    ap.add_argument("--data-style", default="ours", choices=["ours", "winner"])
    ap.add_argument("--prior-root", default=None,
                    help="frozen native merged-prior dir ({name}-t1n-inference.nii.gz) -> 3rd channel")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model = LitInpaint.load_from_checkpoint(args.ckpt).cuda().eval()

    ids = [i.strip() for i in Path(args.ids).read_text().splitlines() if i.strip()]
    root = Path(args.data_root)
    for k, name in enumerate(ids):
        d = root / name
        vimg = nib.load(str(d / f"{name}-t1n-voided.nii.gz"))
        voided = np.asarray(vimg.get_fdata(), dtype=np.float32)
        mask = load(d / f"{name}-mask.nii.gz") > 0.5
        prior = load(Path(args.prior_root) / f"{name}-t1n-inference.nii.gz") if args.prior_root else None

        result = inpaint_volume(model, voided, mask, roi=args.roi, tta=args.tta, style=args.data_style,
                                prior=prior)
        nib.save(nib.Nifti1Image(result, vimg.affine, vimg.header),
                 out / f"{name}-t1n-inference.nii.gz")
        if (k + 1) % 20 == 0:
            print(f"{k+1}/{len(ids)}", flush=True)
    print(f"wrote {len(ids)} predictions to {out}")


if __name__ == "__main__":
    main()
