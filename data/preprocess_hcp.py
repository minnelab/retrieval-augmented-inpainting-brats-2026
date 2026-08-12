#!/usr/bin/env python
"""Bring HCP T1 (FreeSurfer `norm.nii.gz`) into BraTS space → extra *healthy* training data.

Why (see program.md §2/§7 + eda/FINDINGS.md): BraTS Task 4 is scored ONLY on synthesized
*healthy* tissue (the tumor-shaped healthy decoy blob). The biggest free data lever is generating
more healthy examples. HCP is hundreds of healthy brains. Putting each HCP T1 into the exact BraTS
grid lets the existing pipeline (data/generate_masks.py → src/diffusion/data.py → official metric) consume
them unchanged: drop synthetic decoy masks into real healthy tissue and you get unlimited
(t1n, t1n-voided, mask, mask-healthy) tuples with *genuine* healthy ground truth.

What this does, per subject: rigid-register `norm.nii.gz` to the BraTS-SRI24 atlas and resample
onto the BraTS grid (240,240,155 @ 1mm, LPS), writing `{case}-t1n.nii.gz` + the rigid matrix
(reusable later to warp the subject's diffusion-derived maps into the same space).

The HCP `norm.nii.gz` is already skull-stripped + intensity-normalized (FreeSurfer, conformed
256^3 @ 1mm, AC-PC aligned), so no skull-strip is needed — only rigid alignment + reslice. We
register to the *skull-stripped* atlas to match.

Usage (full 848-subject run, parallel, resumable):
  python data/preprocess_hcp.py \
      --hcp-root    /home/maia-user/vMRE/HCP_1200 \
      --output-root /home/maia-user/HCP_brats_t1 \
      --workers 40

Smoke test on 3 subjects:
  python data/preprocess_hcp.py --hcp-root /home/maia-user/vMRE/HCP_1200 \
      --output-root /tmp/hcp_brats_test --limit 3 --workers 3
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

# One BLAS thread per process — we parallelize across subjects, not within a registration,
# so this avoids 40 workers each spawning 48 threads and thrashing the cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import nibabel as nib

# BraTS-SRI24 skull-stripped atlas (defines the BraTS grid + is the registration target).
# Same artifact brainles-preprocessing uses; auto-downloaded once if absent.
ATLAS_URL = ("https://zenodo.org/api/records/15927391/files/"
             "brats_sri24_skullstripped.nii/content")


def get_atlas(atlas_path: Path) -> Path:
    if not atlas_path.exists():
        atlas_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[atlas] downloading BraTS-SRI24 (skull-stripped) → {atlas_path}")
        urllib.request.urlretrieve(ATLAS_URL, atlas_path)
    return atlas_path


def register_one(args):
    """Rigid-register one HCP T1 to the atlas and resample onto the BraTS grid."""
    subj_dir, out_dir, atlas_path, t1_name, overwrite = args
    case = f"HCP-{subj_dir.name}-000"          # BraTS-style case id
    case_out = out_dir / case
    t1_out = case_out / f"{case}-t1n.nii.gz"
    if t1_out.exists() and not overwrite:
        return (case, "skip")
    t1_in = subj_dir / t1_name
    if not t1_in.exists():
        return (case, f"missing {t1_name}")

    try:
        # Imported inside the worker so each process initializes its own dipy state.
        from dipy.align.imaffine import (AffineRegistration, MutualInformationMetric,
                                         transform_centers_of_mass)
        from dipy.align.transforms import RigidTransform3D

        atlas = nib.load(str(atlas_path))
        static = np.asarray(atlas.get_fdata(), dtype=np.float32).squeeze()
        static_affine = atlas.affine

        mov = nib.load(str(t1_in))
        moving = np.asarray(mov.get_fdata(), dtype=np.float32)
        moving_affine = mov.affine

        # 1) coarse: align centers of mass. 2) refine: rigid via mutual information.
        com = transform_centers_of_mass(static, static_affine, moving, moving_affine)
        # sampling_proportion<1 evaluates MI on a random voxel subset — the dominant speed lever
        # (dense MI is ~5x slower); 20% is plenty for a rigid fit between conformed brains.
        metric = MutualInformationMetric(nbins=32, sampling_proportion=0.2)
        # HCP norm is already conformed/AC-PC aligned, so only a small rigid correction is
        # needed — modest iters suffice (≈Dice 0.9+ vs atlas) and keep the 848-case batch fast.
        affreg = AffineRegistration(metric=metric, level_iters=[1000, 200, 50],
                                    sigmas=[3.0, 1.0, 0.0], factors=[4, 2, 1], verbosity=0)
        rigid = affreg.optimize(static, moving, RigidTransform3D(), None,
                                static_affine, moving_affine,
                                starting_affine=com.affine)

        # Resample moving onto the static (BraTS) grid; clip interpolation undershoot to 0.
        resampled = rigid.transform(moving)
        resampled = np.clip(resampled, 0, None).astype(np.float32)

        case_out.mkdir(parents=True, exist_ok=True)
        out_img = nib.Nifti1Image(resampled, static_affine)
        out_img.header.set_zooms((1.0, 1.0, 1.0))
        nib.save(out_img, str(t1_out))
        # Persist the rigid world→world matrix (reuse later for diffusion maps → same space).
        np.savetxt(case_out / f"{case}-to_brats.txt", rigid.affine)
        return (case, "ok")
    except Exception as e:  # keep the batch going; report the failure
        return (case, f"ERROR {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hcp-root", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--atlas", type=Path,
                    default=Path(__file__).resolve().parent / "atlas" / "brats_sri24_skullstripped.nii")
    ap.add_argument("--t1-name", default="norm.nii.gz", help="T1 filename inside each subject dir")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help=">0: only first N subjects")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    atlas_path = get_atlas(args.atlas)
    subjects = sorted(p for p in args.hcp_root.iterdir()
                      if p.is_dir() and (p / args.t1_name).exists())
    if args.limit > 0:
        subjects = subjects[:args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] {len(subjects)} subjects | atlas={atlas_path} | out={args.output_root} "
          f"| workers={args.workers}")

    tasks = [(s, args.output_root, atlas_path, args.t1_name, args.overwrite) for s in subjects]
    counts = {"ok": 0, "skip": 0, "fail": 0}
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            it = pool.imap_unordered(register_one, tasks)
            for i, (case, status) in enumerate(it, 1):
                bucket = "ok" if status == "ok" else "skip" if status == "skip" else "fail"
                counts[bucket] += 1
                if bucket == "fail":
                    print(f"  [{i}/{len(tasks)}] {case}: {status}")
                elif i % 25 == 0 or i == len(tasks):
                    print(f"  [{i}/{len(tasks)}] ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}")
    else:
        for i, t in enumerate(tasks, 1):
            case, status = register_one(t)
            bucket = "ok" if status == "ok" else "skip" if status == "skip" else "fail"
            counts[bucket] += 1
            print(f"  [{i}/{len(tasks)}] {case}: {status}")

    print(f"[done] ok={counts['ok']} skip={counts['skip']} fail={counts['fail']} → {args.output_root}")
    if counts["fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
