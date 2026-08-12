"""Offline: generate a BANK of N official-distribution healthy-decoy masks per brain, so training can
draw a FRESH one each epoch (the winner's mask-aug lever, val-matching distribution) WITHOUT the live
sampler's per-item cost. Stores only the small decoy masks (packed) per brain -> tiny disk vs writing
N voided volumes; the MaskBank dataloader reconstructs voided = t1*(~(tumor|decoy)) live.

  python data/gen_mask_bank.py --gli-root <GLI> --ids splits/train_ids.txt --n 40 \
      --out <bank_dir> --pool-cache <pool.pkl> --threads 16
"""
import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import nibabel as nib

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_sampler import build_official_pool, OfficialSampler

_G = {}


def _load(p):
    return np.asarray(nib.load(str(p)).get_fdata(), dtype=np.float32)


def _one(args):
    name, root, out_dir, n, seed = args
    root, out = Path(root), Path(out_dir)
    of = out / f"{name}.npz"
    if of.exists():
        return f"skip {name}"
    tf = root / name / f"{name}-t1n.nii.gz"
    uf = root / name / f"{name}-mask-unhealthy.nii.gz"
    if not tf.exists() or not uf.exists():
        return f"miss {name}"
    t1 = _load(tf); brain = t1 > 0.02 * t1.max(); tumor = _load(uf) > 0.5
    samp = OfficialSampler(_G["pool"], size_cdf=_G.get("size_cdf"), dm_dir=_G.get("dm_dir"))
    masks, empty = [], 0                                     # packed full-volume decoy masks
    for s in range(n):
        m = samp.sample_healthy(brain, tumor, np.random.default_rng(seed + s), brain_key=name)
        empty += int(not m.any())
        masks.append(np.packbits(m.ravel()))
    np.savez_compressed(of, masks=np.array(masks), shape=np.array(t1.shape))
    return f"ok {name} ({n} decoys, {empty} empty)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gli-root", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool-cache", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--size-cdf-npy", default=None,
                    help="npy of target decoy voxel-sizes (the eval size distribution). If set, decoys "
                         "are scaled to match it instead of the inverse-proportional official selection.")
    ap.add_argument("--dm-dir", default=None, help="precomputed distance maps (data/precompute_dm.py) to speed gen")
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    _G["pool"] = build_official_pool(a.pool_cache, a.gli_root, ids)
    if a.size_cdf_npy:
        _G["size_cdf"] = np.load(a.size_cdf_npy)
        print(f"size-matched: target CDF n={len(_G['size_cdf'])} median={int(np.median(_G['size_cdf']))}", flush=True)
    _G["dm_dir"] = a.dm_dir
    Path(a.out).mkdir(parents=True, exist_ok=True)
    work = [(n, a.gli_root, a.out, a.n, a.seed + i * 100) for i, n in enumerate(ids)]
    ok = 0
    with Pool(a.threads) as pool:
        for k, r in enumerate(pool.imap_unordered(_one, work, chunksize=4)):
            if r.startswith("ok"):
                ok += 1
            if (k + 1) % 100 == 0:
                print(f"{k+1}/{len(work)} ({ok} ok)", flush=True)
    print(f"DONE: {ok}/{len(work)} brains -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
