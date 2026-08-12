"""Retrieval index over a large donor pool — the core of the retrieval prior.

Memory-safe by construction: donor volumes are never all held in RAM.

  1. INDEX  — stream every donor brain once, store only a small low-res structural embedding
              (24x24x16, z-scored) plus its path.
  2. QUERY  — embed the target's VISIBLE (voided) tissue, take the top-`shortlist` donors by
              cosine similarity, load ONLY those, rerank by full-resolution MSE over the
              visible region, and fill the void from the best (or a top-K blend).

Donor pool = BraTS training cases (val split excluded, no leakage) plus 808 fully healthy HCP
brains in BraTS space, which are ideal donors because they carry no tumour to borrow. The
low-res embedding is a no-train stand-in for a learned encoder.

Used as a library by the gen_*.py scripts in this directory and by deploy/predict_merged.py.
"""
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import nibabel as nib
from scipy import ndimage

EMB = (24, 24, 16)            # low-res structural embedding grid


def load(p):
    return np.asarray(nib.load(str(p)).get_fdata(), dtype=np.float32)


def pct995(vol):
    nz = vol[vol > 0]
    return float(np.percentile(nz, 99.5)) if nz.size else 1.0


def embed(vol, scale):
    """Scale-normalized, z-scored low-res descriptor — robust to intensity, captures gross shape."""
    v = vol / (scale if scale > 1e-6 else 1.0)
    f = [EMB[i] / vol.shape[i] for i in range(3)]
    d = ndimage.zoom(v, f, order=1).ravel().astype(np.float32)
    return (d - d.mean()) / (d.std() + 1e-6)


def _index_one(path):
    try:
        v = load(path)
        return path, embed(v, pct995(v))
    except Exception:
        return path, None


def build_index(brats, hcp, exclude, cache, workers=8):
    if Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        print(f"index: loaded {len(z['paths'])} donors from cache")
        return list(z["paths"]), z["emb"]
    paths = []
    for c in sorted(Path(brats).glob("BraTS-*")):
        if c.name not in exclude:
            paths.append(str(c / f"{c.name}-t1n.nii.gz"))
    if hcp:
        for c in sorted(Path(hcp).glob("HCP-*")):
            paths.append(str(c / f"{c.name}-t1n.nii.gz"))
    paths = [p for p in paths if Path(p).exists()]
    print(f"index: embedding {len(paths)} donors ({workers} workers)...")
    embs, keep = [], []
    with Pool(workers) as pool:
        for i, (p, e) in enumerate(pool.imap_unordered(_index_one, paths, chunksize=8)):
            if e is not None:
                keep.append(p); embs.append(e)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(paths)}")
    emb = np.stack(embs).astype(np.float32)
    np.savez(cache, paths=np.array(keep), emb=emb)
    print(f"index: built {len(keep)} donors, dim={emb.shape[1]}")
    return keep, emb
