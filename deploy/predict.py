"""BraTS 2026 Task 4 (inpainting) submission entrypoint — wavelet-diffusion model.

The challenge runner mounts an input directory of cases and an output directory, then
runs this container with no arguments. For every case it finds a voided T1 and a mask:

    <case>-t1n-voided.nii.gz   +   <case>-mask.nii.gz

and writes the inpainted volume back as:

    <case>-t1n-inference.nii.gz

(composited: voided input outside the mask, model prediction inside; original affine/header).

Model = the UNCONDITIONED wavelet-diffusion `wdm_hcp2` (best on our val-150: SSIM 0.872, beats the
2025-winner container 0.865). Inference uses few-step DDIM sampling with sample-averaging (n_avg) and
8-flip test-time augmentation (tta) — the config that scored 0.87196. No tissue-prior pipeline is
needed (unlike the prior-conditioned variant), so the container is fully self-contained.

Input/output dirs and the checkpoint come from $INPUT_DIR / $OUTPUT_DIR / $CKPT
(defaults /input, /output, /app/model.ckpt). Inference knobs: $ROI (128,128,128), $STEPS (4),
$N_AVG (4), $TTA (1=on). Overridable as positional args: python predict.py [INPUT] [OUTPUT] [CKPT].

No model code lives here — it imports the trained network and the inference core from src/.
"""
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data import load                        # noqa: E402
from infer import diffuse_volume             # noqa: E402
from litdiff import LitDiffusion             # noqa: E402

VOIDED_SUFFIX = "-t1n-voided.nii.gz"
MASK_SUFFIX = "-mask.nii.gz"
OUT_SUFFIX = "-t1n-inference.nii.gz"


def main():
    argv = sys.argv[1:]
    in_dir = Path(argv[0] if len(argv) > 0 else os.environ.get("INPUT_DIR", "/input"))
    out_dir = Path(argv[1] if len(argv) > 1 else os.environ.get("OUTPUT_DIR", "/output"))
    ckpt = argv[2] if len(argv) > 2 else os.environ.get("CKPT", "/app/model.ckpt")
    roi = int(os.environ.get("ROI", "128,128,128").split(",")[0])
    steps = int(os.environ.get("STEPS", "4"))
    n_avg = int(os.environ.get("N_AVG", "4"))
    tta = os.environ.get("TTA", "1") not in ("0", "false", "False", "")
    # boost-only adaptive n_avg (8/8/8/16): floor 8 everywhere, bump big holes to 16 -> never
    # under-samples, only improves the big-hole cases we rank worst on (helps the per-case rank-sum).
    adaptive = os.environ.get("ADAPTIVE", "1") not in ("0", "false", "False", "")
    n_avg_min = int(os.environ.get("N_AVG_MIN", "8"))
    n_avg_max = int(os.environ.get("N_AVG_MAX", "16"))
    batch_cap = int(os.environ.get("BATCH_CAP", "32"))     # OOM auto-falls-back, so safe on small GPUs

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} ckpt={ckpt} roi={roi} steps={steps} n_avg={n_avg} tta={tta} "
          f"adaptive={adaptive}({n_avg_min}-{n_avg_max}) batch_cap={batch_cap}", flush=True)
    print(f"input={in_dir} output={out_dir}", flush=True)

    # flash attention: math-identical, far less memory -> batched TTA/n_avg fit on any GPU.
    kw = {"use_flash_attention": True} if device.type == "cuda" else {}
    model = LitDiffusion.load_from_checkpoint(ckpt, map_location=device, **kw).to(device).eval()

    voided_files = sorted(in_dir.rglob(f"*{VOIDED_SUFFIX}"))
    if not voided_files:
        sys.exit(f"no *{VOIDED_SUFFIX} files found under {in_dir}")
    print(f"found {len(voided_files)} case(s)", flush=True)

    for k, vp in enumerate(voided_files):
        name = vp.name[: -len(VOIDED_SUFFIX)]
        mp = vp.with_name(f"{name}{MASK_SUFFIX}")
        if not mp.exists():
            print(f"WARN skip {name}: missing {mp.name}", flush=True)
            continue

        vimg = nib.load(str(vp))
        voided = np.asarray(vimg.get_fdata(), dtype=np.float32)
        mask = load(mp) > 0.5

        result = diffuse_volume(model, voided, mask, prior=None, roi=roi, steps=steps,
                                n_avg=n_avg, tta=tta, device=device, batch_cap=batch_cap,
                                adaptive=adaptive, n_avg_min=n_avg_min, n_avg_max=n_avg_max)
        nib.save(nib.Nifti1Image(result, vimg.affine, vimg.header),
                 out_dir / f"{name}{OUT_SUFFIX}")
        print(f"[{k + 1}/{len(voided_files)}] wrote {name}{OUT_SUFFIX}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
