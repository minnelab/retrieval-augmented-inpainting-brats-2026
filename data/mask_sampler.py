"""On-the-fly tumor-shaped mask generation for healthy brains (HCP), for use during dataloading.

Build a pool of REAL tumor shapes from BraTS `-mask-unhealthy` once (cached to disk), then per call
stamp two voids into a brain: a larger *tumor-like* void (≈ mask-unhealthy sizes) and a smaller
*healthy-like* void (≈ mask-healthy sizes), placed ≥`min_dist` voxels apart — mirroring the BraTS
local-synthesis construction (see eda/FINDINGS.md). On a healthy brain BOTH voids cover real tissue
with known ground truth, so a training loss can supervise both (use their union as the loss mask),
unlike real BraTS where the tumor void is unscored.

Placement avoids the expensive distance transform (checks brain-intersection on the shape's bbox
sub-box) so it is cheap enough to run per sample in a DataLoader worker.
"""
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage


# ---------------------------------------------------------------------------------------------
# shape pool (built once from BraTS, cached)
# ---------------------------------------------------------------------------------------------
def build_shape_pool(brats_root, n_cases=200, min_size=2000, max_size=600000, seed=0):
    """Harvest tumor-compartment shapes + the real size distributions from BraTS."""
    rng = np.random.default_rng(seed)
    cases = sorted(glob.glob(f"{brats_root}/BraTS-*"))
    pick = rng.choice(len(cases), size=min(n_cases, len(cases)), replace=False)
    pool, healthy_sizes, tumor_sizes = [], [], []
    for idx in pick:
        c = cases[idx]; name = os.path.basename(c)
        uf = Path(c) / f"{name}-mask-unhealthy.nii.gz"
        if not uf.exists():
            continue
        uh = np.asarray(nib.load(str(uf)).get_fdata()) > 0.5
        tumor_sizes.append(int(uh.sum()))
        lab, n = ndimage.label(uh)
        for sl, i in zip(ndimage.find_objects(lab), range(1, n + 1)):
            crop = lab[sl] == i
            s = int(crop.sum())
            if min_size <= s <= max_size:
                pool.append(ndimage.binary_fill_holes(crop).copy())
        # the real healthy-decoy sizes (target distribution for the smaller void)
        for hf in glob.glob(f"{c}/{name}-mask-healthy*.nii.gz"):
            healthy_sizes.append(int((np.asarray(nib.load(hf).get_fdata()) > 0.5).sum()))
    return {"pool": pool,
            "healthy_sizes": np.array(healthy_sizes),
            "tumor_sizes": np.array(tumor_sizes)}


def load_or_build_pool(cache_path, brats_root=None, **kw):
    """Load a cached pool pickle, or build it from `brats_root` and cache it."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    if brats_root is None:
        raise FileNotFoundError(f"{cache_path} missing and no brats_root given to build it")
    pool = build_shape_pool(brats_root, **kw)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(pool, f)
    return pool


# ---------------------------------------------------------------------------------------------
# shape transforms
# ---------------------------------------------------------------------------------------------
def scale_to_volume(shape_arr, target_vol):
    """Isotropically zoom a binary shape so its voxel count ≈ target_vol (order=0, no interp)."""
    cur = int(shape_arr.sum())
    if cur == 0:
        return shape_arr
    factor = float(np.clip((target_vol / cur) ** (1.0 / 3.0), 0.4, 2.5))
    z = ndimage.zoom(shape_arr.astype(np.float32), factor, order=0, mode="grid-constant",
                     prefilter=False) > 0.5
    return ndimage.binary_fill_holes(z) if z.any() else shape_arr


def random_transform(shape_arr, rng):
    """Random mirror per axis + random rotation in two planes (order=0, no interpolation)."""
    fx, fy, fz = (rng.integers(0, 2, 3) * 2 - 1)
    a = shape_arr[::fx, ::fy, ::fz]
    a = ndimage.rotate(a, rng.uniform(0, 360), axes=(0, 1), reshape=True, order=0,
                       mode="grid-constant", prefilter=False)
    a = ndimage.rotate(a, rng.uniform(0, 360), axes=(1, 2), reshape=True, order=0,
                       mode="grid-constant", prefilter=False)
    return a


# ---------------------------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------------------------
class MaskSampler:
    """Stamp a (tumor-like, healthy-like) mask pair into a brain. Cheap enough for dataloading."""

    def __init__(self, pool, min_dist=5.0, min_intersection=0.75, min_size=2000,
                 healthy_frac_cap=0.7, transform_tries=8, place_tries=200):
        self.pool = pool["pool"]
        self.h_sizes = pool["healthy_sizes"]
        self.t_sizes = pool["tumor_sizes"]
        self.min_dist = float(min_dist)
        self.min_intersection = float(min_intersection)
        self.min_size = int(min_size)
        self.healthy_frac_cap = float(healthy_frac_cap)
        self.transform_tries = int(transform_tries)
        self.place_tries = int(place_tries)

    def _shape_for(self, target_vol, rng):
        return random_transform(scale_to_volume(self.pool[rng.integers(len(self.pool))],
                                                int(target_vol)), rng)

    def _place(self, brain, cand, shp, rng, avoid=None):
        """Try random centers; accept the first placement mostly inside the brain (and, if `avoid`
        given, not touching it). Only the bbox sub-box is examined per try → no full-volume alloc."""
        sh = np.array(shp.shape)
        ssum = int(shp.sum())
        if ssum == 0:
            return None
        for _ in range(self.place_tries):
            center = cand[rng.integers(len(cand))]
            lo = center - sh // 2
            hi = lo + sh
            if np.any(lo < 0) or np.any(hi > brain.shape):
                continue
            sub = brain[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            inter = np.logical_and(sub, shp)
            if inter.sum() / ssum < self.min_intersection:
                continue
            if avoid is not None and np.logical_and(avoid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]],
                                                    shp).any():
                continue
            if inter.sum() < self.min_size * 0.5:
                continue
            out = np.zeros(brain.shape, bool)
            out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = inter   # already clipped to brain
            return out
        return None

    def _stamp(self, brain, cand, target_vol, rng, avoid=None):
        for _ in range(self.transform_tries):
            m = self._place(brain, cand, self._shape_for(target_vol, rng), rng, avoid=avoid)
            if m is not None:
                return m
        return None

    def sample_healthy(self, brain, rng, avoid=None, size_range=None):
        """Stamp ONE healthy-like void into a (possibly tumor-bearing) brain, avoiding `avoid`
        (e.g. the dilated real tumor). `size_range=(lo,hi)` draws the target volume log-uniform in
        [lo,hi] — use a small range to MATCH the val healthy-decoy distribution (median ~17k).
        None → draw from the harvested healthy_sizes."""
        cand = np.argwhere(brain)
        if len(cand) == 0:
            return np.zeros(brain.shape, bool)
        if size_range is not None:
            lo, hi = size_range
            h_tgt = int(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            h_tgt = int(self.h_sizes[rng.integers(len(self.h_sizes))])
        m = self._stamp(brain, cand, h_tgt, rng, avoid=avoid)
        return m if m is not None else np.zeros(brain.shape, bool)

    def sample(self, brain, rng, large_frac=0.0):
        """Return (mask_tumor, mask_healthy) bool arrays — tumor-like larger, ≥min_dist apart.
        `large_frac` gently fattens the big-decoy tail: with probability `large_frac` this decoy's
        target volume is the MAX of 2 natural size draws (order statistic → skews large using ONLY
        real harvested sizes); otherwise a single natural draw. So most decoys stay natural and only
        a `large_frac` slice is upsized — a slight Q4 nudge, not a distribution shift. Tumor and
        healthy skew together so the healthy_frac_cap (healthy < 0.7·tumor) doesn't clamp the gain."""
        cand = np.argwhere(brain)
        z = np.zeros(brain.shape, bool)
        if len(cand) == 0:
            return z, z
        k = 2 if (large_frac > 0 and rng.random() < large_frac) else 1
        t_tgt = int(max(self.t_sizes[rng.integers(len(self.t_sizes))] for _ in range(k)))
        h_raw = int(max(self.h_sizes[rng.integers(len(self.h_sizes))] for _ in range(k)))
        h_tgt = int(min(h_raw, self.healthy_frac_cap * t_tgt))
        mt = self._stamp(brain, cand, t_tgt, rng)
        avoid = ndimage.binary_dilation(mt, iterations=int(round(self.min_dist))) if mt is not None else None
        mh = self._stamp(brain, cand, h_tgt, rng, avoid=avoid)
        return (mt if mt is not None else z), (mh if mh is not None else z)
