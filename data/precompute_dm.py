"""Precompute the per-brain distance map + dilated tumor that OfficialSampler.sample_healthy computes
live (1.1s/item — the training-dataloader bottleneck). With these cached to disk the live sample drops
to ~0.1s/item (profiled). GLI/GoAT use the real mask-unhealthy tumor. Stored: dm(float16), tumor_d(packed).
HCP has no fixed tumor (fresh fake tumor per item) so it is skipped here and recomputed live.

  python data/precompute_dm.py --roots /tmp/gli_only /tmp/goat_new --out /home/maia-user/dm_cache \
      --tumor-dilation 5.0 --threads 24
"""
import argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np, nibabel as nib
from scipy import ndimage

_G = {}

def _one(args):
    d, name, out, dilation = args
    of = Path(out) / f"{name}.npz"
    if of.exists():
        return "skip"
    d = Path(d)
    tf = d / f"{name}-mask-unhealthy.nii.gz"; t1f = d / f"{name}-t1n.nii.gz"
    if not tf.exists() or not t1f.exists():
        return "miss"
    t1 = np.asarray(nib.load(str(t1f)).get_fdata(), dtype=np.float32)
    brain = t1 > 0.02 * t1.max()
    tumor = np.asarray(nib.load(str(tf)).get_fdata()) > 0.5
    dm = ndimage.distance_transform_edt(~tumor) - dilation          # EXACT match to sampler
    tumor_d = tumor | (dm < 0)
    dm[~brain] = 0
    # uint8 (clip [0,255]): the sampler only needs dm>min_dist and farthest-point; negatives (near tumor)
    # and background -> 0, correctly excluded. UNCOMPRESSED np.savez -> fast reads (decompress was slower
    # than recomputing). ~10MB/brain.
    dm_u8 = np.clip(dm, 0, 255).astype(np.uint8)
    np.savez(of, dm=dm_u8, tumor_d=np.packbits(tumor_d.ravel()), shape=np.array(tumor_d.shape))
    return "ok"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tumor-dilation", type=float, default=5.0)
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    work = []
    for r in a.roots:
        for p in sorted(Path(r).glob("*")):
            if p.is_dir():
                work.append((str(p), p.name, a.out, a.tumor_dilation))
    print(f"precomputing dm for {len(work)} brains -> {a.out}", flush=True)
    ok = 0
    with Pool(a.threads) as pool:
        for k, r in enumerate(pool.imap_unordered(_one, work, chunksize=4)):
            ok += (r == "ok")
            if (k + 1) % 200 == 0:
                print(f"{k+1}/{len(work)} ({ok} ok)", flush=True)
    print(f"DONE: {ok}/{len(work)}", flush=True)

if __name__ == "__main__":
    main()
