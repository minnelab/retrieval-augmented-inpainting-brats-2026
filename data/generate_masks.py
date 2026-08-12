#!/usr/bin/env python
"""Generate BraTS inpainting training data (t1n-voided, mask, mask-healthy, mask-unhealthy)
from a raw BraTS segmentation dataset, using the official challenge pipeline (vendored in
data/maskgen/include.py).

Works for any BraTS dataset whose cases contain `{case}-t1n.nii.gz` and `{case}-seg.nii.gz`
in (240,240,155), 1mm space — i.e. GLI (regenerate with more masks per brain) and
MET / meningioma (for generalization; see program.md).

Stages (mirrors dataset/dataset_generation.ipynb):
  1. binarize whole-tumor segs   2. extract tumor compartments (shape pool)
  3. sample healthy decoy masks + write the 5 output files per case.

Usage:
  python data/generate_masks.py \
      --input-root  <data>/BraTS-MET/MICCAI-LH-BraTS2025-MET-Challenge-Training \
      --output-root <data>/BraTS-Task04-derived/MET-inpainting \
      --work-dir    <data>/BraTS-Task04-derived/MET-inpainting/_cache \
      --samples-per-brain 1 --threads 16 --seed 2026

Smoke test on N cases (creates a temp symlink dir):
  python data/generate_masks.py --input-root <raw> --output-root /tmp/gen_test \
      --work-dir /tmp/gen_test/_cache --limit 6 --threads 4
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "maskgen"))
import include  # noqa: E402  (vendored challenge pipeline)


def make_subset_dir(input_root: Path, limit: int) -> Path:
    """Symlink the first `limit` case folders into a temp dir (for smoke tests / splits)."""
    cases = sorted([p for p in input_root.glob("*") if p.is_dir() and p.name.startswith("BraTS")])[:limit]
    tmp = Path(tempfile.mkdtemp(prefix="brats_subset_"))
    for c in cases:
        (tmp / c.name).symlink_to(c)
    print(f"[subset] linked {len(cases)} cases into {tmp}")
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path, help="dir for cache .gz files + chdir target")
    ap.add_argument("--samples-per-brain", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--limit", type=int, default=0, help=">0: only process first N cases (smoke test)")
    # generation hyperparams (challenge defaults)
    ap.add_argument("--tumor-dilation", type=float, default=5.0)
    ap.add_argument("--min-dist-to-tumor", type=float, default=5.0)
    ap.add_argument("--size-range-tol", type=float, default=0.1)
    ap.add_argument("--rand-points", type=int, default=2)
    ap.add_argument("--min-brain-intersection", type=float, default=0.75)
    ap.add_argument("--force-refresh", action="store_true")
    args = ap.parse_args()

    input_root = args.input_root
    if args.limit and args.limit > 0:
        input_root = make_subset_dir(args.input_root, args.limit)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.work_dir)  # cache .gz files (binarySegmentationMasks.gz, tumorCompartments.gz) land here
    print(f"[cfg] input={input_root}\n[cfg] output={args.output_root}\n[cfg] cwd(cache)={args.work_dir}")

    # 1. binarize whole-tumor segmentations
    tumorSegmentations = include.generateBinarySegmentationMasks(
        inputFolderRoot=input_root, relevantLabels=[1, 2, 3],
        segmentationMinSize=800, fillHoles=True,
        forceRefresh=args.force_refresh, threads=args.threads,
    )
    print(f"[stage1] binarized {len(tumorSegmentations.index)} brains")

    # 2. extract tumor compartments (shape pool for decoys)
    tumorCompartments = include.getTumorCompartments(
        tumorSegmentations, forceRefresh=args.force_refresh, threads=args.threads,
    )
    print(f"[stage2] {len(tumorCompartments.index)} compartments")

    # 3. sample healthy decoys + write t1n / t1n-voided / mask / mask-healthy / mask-unhealthy
    healthyMasks = include.getHealthyMasks(
        input_root, args.output_root, tumorSegmentations, tumorCompartments,
        samplesPerBrain=args.samples_per_brain,
        tumorDilationDistance=args.tumor_dilation,
        minDistanceToTumor=args.min_dist_to_tumor,
        sizeRangeTolerance=args.size_range_tol,
        randPointsN=args.rand_points,
        minimalBrainIntersection_p=args.min_brain_intersection,
        forceRefresh=args.force_refresh, threads=args.threads, seed=args.seed,
    )
    print(f"[stage3] wrote masks for {len(healthyMasks.index)} brains -> {args.output_root}")


if __name__ == "__main__":
    main()
