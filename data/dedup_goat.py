#!/usr/bin/env python
"""Content-fingerprint dedup of a pooled cohort (GoAT) against our GLI split.

GoAT re-packages BraTS gliomas (incl. our cases) under anonymized `BraTS-GoAT-NNNNN` ids, so we
cannot dedup by id. Instead fingerprint each t1n by a coarse block-mean signature (z-normalized)
and flag any pooled case whose best cosine match against the EXCLUDE set (our val ids — and
optionally train) exceeds a threshold: those are the same scan re-packaged → drop to avoid leakage.

Writes the KEEP id list (pooled cases with no near-duplicate in EXCLUDE). `--report` prints the
best-match similarity distribution so the threshold can be verified to sit in a clean bimodal gap
(true duplicates ~1.0, distinct brains lower) before trusting it.

  python data/dedup_goat.py --gli-root $DATA_GLI --pool-root <GoAT-src> \
      --exclude splits/val_ids.txt [--exclude splits/train_ids.txt] \
      --out splits/goat_keep_ids.txt --threshold 0.999 --report
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib


def signature(path, blocks=(24, 24, 31)):
    """Coarse z-normalized block-mean fingerprint of a t1n volume (robust to re-saving).
    Vectorized block-reduce: trim each axis to a multiple of its block count, reshape, mean."""
    v = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    f = [s // b for s, b in zip(v.shape, blocks)]                 # block size per axis
    v = v[:f[0]*blocks[0], :f[1]*blocks[1], :f[2]*blocks[2]]
    sig = v.reshape(blocks[0], f[0], blocks[1], f[1], blocks[2], f[2]).mean(axis=(1, 3, 5)).ravel()
    sig = sig - sig.mean()
    n = np.linalg.norm(sig)
    return sig / n if n > 0 else sig


def t1n_path(root, name):
    return Path(root) / name / f"{name}-t1n.nii.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gli-root", required=True)
    ap.add_argument("--pool-root", required=True)
    ap.add_argument("--exclude", action="append", required=True, help="split file(s) of GLI ids to exclude")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.999)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    excl_ids = []
    for f in args.exclude:
        excl_ids += [l.strip() for l in Path(f).read_text().splitlines() if l.strip()]
    print(f"computing {len(excl_ids)} exclude signatures (GLI)...", flush=True)
    E = np.stack([signature(t1n_path(args.gli_root, n)) for n in excl_ids])      # (Nex, D)

    pool = sorted(p.name for p in Path(args.pool_root).glob("BraTS-GoAT-*") if p.is_dir())
    print(f"matching {len(pool)} pooled cases...", flush=True)
    keep, best = [], []
    for k, name in enumerate(pool):
        s = signature(t1n_path(args.pool_root, name))
        bm = float((E @ s).max())                                                # best cosine vs exclude
        best.append(bm)
        if bm < args.threshold:
            keep.append(name)
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(pool)}", flush=True)

    Path(args.out).write_text("\n".join(keep) + "\n")
    best = np.array(best)
    dropped = len(pool) - len(keep)
    print(f"\nkept {len(keep)}, dropped {dropped} (near-dup of exclude) -> {args.out}")
    if args.report:
        qs = np.percentile(best, [0, 50, 90, 95, 99, 100])
        print("best-match cosine percentiles [0,50,90,95,99,100]:", np.round(qs, 5))
        print(f"cases in [{args.threshold-0.005:.3f}, {args.threshold:.3f}): "
              f"{int(((best >= args.threshold-0.005) & (best < args.threshold)).sum())} "
              "(near-threshold — inspect if nonzero to confirm the gap is clean)")


if __name__ == "__main__":
    main()
