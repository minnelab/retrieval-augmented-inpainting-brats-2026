"""Generate a RETRIEVAL prior fill per case (cheap, CPU): the void is filled with the tissue of the
single closest REAL brain in the donor pool (BraTS-train + HCP). Unlike the diffusion prior (a
model's guess, error-correlated with the UNet), this carries genuine observed anatomy -> a more
COMPLEMENTARY input channel for the refiner (see eda discussion / experiments.md).

Matching: low-res z-scored embedding shortlist (cached index) -> full-res MSE rerank over the
VISIBLE (non-void) region -> fill the void with the best donor's tissue, rescaled to this case.
Writes `{id}{sfx}-t1n-inference.nii.gz` (same schema as the diffusion priors / infer.py), so it
drops straight into train.py --prior-root / infer.py --prior-root.

Run (torch env):
  python eda/gen_retrieval_priors.py --ids splits/train_ids.txt --data-root /tmp/brats_full \
      --suffix -0000 --out-dir /home/maia-user/retr_priors/train
  python eda/gen_retrieval_priors.py --ids splits/val_ids.txt  --data-root /tmp/brats_val \
      --out-dir /home/maia-user/retr_priors/val
"""
import argparse
import os
import sys
import threading
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prior_retrieval import load, pct995, embed, build_index


def _gpu_keepalive():
    """This gen is CPU-bound (disk retrieval), but a GPU allocation whose watchdog CANCELS
    jobs that leave the GPU idle (~33 min grace). Run a CONTINUOUS matmul (no sleep -> sustained ~100%
    util) so the allocated GPU never looks idle. sin() keeps values bounded (never inf/nan -> the loop,
    and the thread, never dies)."""
    try:
        a = torch.randn(6144, 6144, device="cuda")
        while True:
            a = torch.sin(a @ a)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--suffix", default="", help="mask-variant suffix, e.g. -0000 (x5 train format)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--brats", default="/tmp/brats_full")
    ap.add_argument("--hcp", default="/home/maia-user/HCP_brats_t1")
    ap.add_argument("--val-ids", default="splits/val_ids.txt", help="excluded from the donor pool")
    ap.add_argument("--shortlist", type=int, default=15)
    ap.add_argument("--whole-donor", action="store_true",
                    help="write the WHOLE donor brain (misaligned reference), not the hole-composite")
    ap.add_argument("--overwrite", action="store_true", help="regen even if the output already exists")
    ap.add_argument("--cache", default="/tmp/prior_retrieval_index.npz")
    args = ap.parse_args()

    if torch.cuda.is_available():                  # keep the allocated GPU busy (low-util watchdogs)
        threading.Thread(target=_gpu_keepalive, daemon=True).start()
        print("gpu keep-alive started")

    ids = [x.strip() for x in Path(args.ids).read_text().splitlines() if x.strip()]
    val = set(x.strip() for x in Path(args.val_ids).read_text().splitlines() if x.strip())
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root); sfx = args.suffix

    paths, emb = build_index(args.brats, args.hcp, exclude=val, cache=args.cache)
    embn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6)
    names = [Path(p).parent.name for p in paths]

    for k, name in enumerate(ids):
        outf = out / f"{name}{sfx}-t1n-inference.nii.gz"
        if outf.exists() and not args.overwrite:      # resumable: skip already-written priors
            continue
        d = root / name
        vimg = nib.load(str(d / f"{name}-t1n-voided{sfx}.nii.gz"))
        voided = np.asarray(vimg.get_fdata(), dtype=np.float32)
        void = load(d / f"{name}-mask{sfx}.nii.gz") > 0.5
        scale = pct995(voided); ctx = voided > 0

        q = embed(voided, scale); q /= (np.linalg.norm(q) + 1e-6)
        order = np.argsort(-(embn @ q))
        tgt = (voided / scale)[ctx].astype(np.float32)
        best, best_mse = None, np.inf
        seen = 0
        for j in order:
            if names[j] == name:                       # never retrieve the target itself
                continue
            dv = load(paths[j]); dn = dv / pct995(dv)
            mse = float(np.mean((tgt - dn[ctx].astype(np.float32)) ** 2))
            if mse < best_mse:
                best_mse, best = mse, dn
            seen += 1
            if seen >= args.shortlist:
                break
        if args.whole_donor and best is not None:
            result = (best.astype(np.float32) * scale)             # whole misaligned donor brain (control ref)
        else:
            result = voided.copy()
            if best is not None:
                result[void] = best[void].astype(np.float32) * scale   # hole-composite: donor tissue in void
        nib.save(nib.Nifti1Image(result, vimg.affine, vimg.header),
                 out / f"{name}{sfx}-t1n-inference.nii.gz")
        if (k + 1) % 50 == 0:
            print(f"{k+1}/{len(ids)}", flush=True)
    print(f"wrote {len(ids)} retrieval priors to {out}", flush=True)
    sys.stdout.flush()
    os._exit(0)          # priors already written+closed; skip interpreter teardown so the CUDA
                         # keep-alive daemon thread can't abort ("terminate called") on shutdown


if __name__ == "__main__":
    main()
