#!/usr/bin/env python
"""Freeze N healthy inpainting masks per HCP brain to disk, in the BraTS inpainting file schema, so
retrieval priors can be generated for HCP (they are per-(brain, mask) and can't use on-the-fly masks).

For each HCP brain (only `{name}-t1n.nii.gz` exists) we draw the SAME masks HCPOnTheFly draws at train
time — MaskSampler places a real tumor-shape + a healthy decoy from the shape pool into the healthy
brain — but write them out as fixed files. Because HCP is entirely healthy the whole void is valid GT,
so mask == mask-healthy == the union (tumor-shape ∪ decoy). Emits, per variant k:
    {name}-t1n.nii.gz  {name}-t1n-voided-000k.nii.gz  {name}-mask-000k.nii.gz  {name}-mask-healthy-000k.nii.gz
This drops straight into InpaintCrops (--train-root) and gen_retrieval_priors (--data-root/--suffix).

Run (torch env, CPU is fine):
  python data/freeze_hcp_inpainting.py --hcp-root $DATA_HCP --out-root $DATA/HCP-inpainting-x5 \
      --pool-cache $DATA/derived/shape_pool.pkl --brats-root $DATA_GLI --samples 5 --seed 2026
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mask_sampler import load_or_build_pool, MaskSampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hcp-root", required=True, help="dir of HCP brains ({name}/{name}-t1n.nii.gz)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--pool-cache", required=True, help="cached real-tumor shape pool (shared w/ training)")
    ap.add_argument("--brats-root", required=True, help="BraTS GLI root — builds the shape pool if cache absent")
    ap.add_argument("--samples", type=int, default=5, help="frozen masks per brain (x5 to match GLI)")
    ap.add_argument("--brain-thr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--limit", type=int, default=0, help=">0: only first N brains (smoke test)")
    args = ap.parse_args()

    pool = load_or_build_pool(args.pool_cache, brats_root=args.brats_root)
    sampler = MaskSampler(pool)
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.name for p in Path(args.hcp_root).glob("*") if p.is_dir())
    if args.limit:
        ids = ids[:args.limit]

    for k, name in enumerate(ids):
        img = nib.load(str(Path(args.hcp_root) / name / f"{name}-t1n.nii.gz"))
        t1 = np.asarray(img.get_fdata(), dtype=np.float32)
        brain = t1 > args.brain_thr * t1.max()
        d = out_root / name; d.mkdir(parents=True, exist_ok=True)
        save = lambda a, fn: nib.save(nib.Nifti1Image(a, img.affine, img.header), str(d / fn))
        save(t1, f"{name}-t1n.nii.gz")
        rng = np.random.default_rng(args.seed + k)                 # deterministic, reproducible masks
        for s in range(args.samples):
            sfx = f"-{s:04d}"
            m_tumor, m_healthy = sampler.sample(brain, rng)
            union = (m_tumor | m_healthy)
            save((t1 * (~union)).astype(np.float32), f"{name}-t1n-voided{sfx}.nii.gz")
            save(union.astype(np.uint8), f"{name}-mask{sfx}.nii.gz")
            save(union.astype(np.uint8), f"{name}-mask-healthy{sfx}.nii.gz")   # HCP: whole void is GT
        if (k + 1) % 50 == 0:
            print(f"{k+1}/{len(ids)}", flush=True)
    print(f"froze {len(ids)} HCP brains x{args.samples} masks -> {out_root}")


if __name__ == "__main__":
    main()
