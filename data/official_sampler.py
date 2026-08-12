"""On-the-fly port of the OFFICIAL challenge healthy-mask sampler (data/maskgen/include.py
getHealthyMasks), so training can draw FRESH masks each epoch from the SAME distribution that made the
frozen GLI-x5 (which matches the val decoy distribution). Our earlier MaskSampler used a different
placement + rescaled sizes and REGRESSED the metric; this reproduces the official algorithm:

  - placement: distance-map biased AWAY from the tumor (sample randPointsN points where dist>minDist,
    take the farthest) — official sampleLocation.
  - shape: a real tumor compartment sampled inversely-proportional to the current tumor size (big tumor
    -> small decoy), with random flips + two random rotations, NO rescale — official sampleCompartment.
  - validity: in-bounds, no tumor overlap, >=minBrainIntersection with brain, >=minDist from tumor.

Speed: the official code does full-volume distance_transform_edt per mask in a resample loop (a batch
job). Here the per-brain distanceMap is CACHED and validity uses a cheap distanceMap lookup instead of
a fresh minDist transform — so it runs live in the dataloader.
"""
import pickle
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage


def _load(p):
    return np.asarray(nib.load(str(p)).get_fdata(), dtype=np.float32)


def build_official_pool(cache_path, gli_root, ids, brain_thr=0.0, min_comp=50):
    """Harvest tumor-compartment shapes (connected components of each brain's whole-tumor void) + their
    sizes-as-brain-fraction, matching the official pool. Pickle {compartments, sizes_frac}. Idempotent."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    root = Path(gli_root)
    comps, sizes = [], []
    for k, name in enumerate([i.strip() for i in ids if i.strip()]):
        uf = root / name / f"{name}-mask-unhealthy.nii.gz"
        tf = root / name / f"{name}-t1n.nii.gz"
        if not uf.exists() or not tf.exists():
            continue
        tumor = _load(uf) > 0.5
        brain_v = float((_load(tf) > brain_thr).sum())
        if brain_v == 0:
            continue
        lab, n = ndimage.label(tumor)
        objs = ndimage.find_objects(lab)                     # per-label bboxes (needs int label array)
        for j in range(1, n + 1):
            comp = lab[objs[j - 1]] == j                     # cropped component (bool)
            if comp.sum() < min_comp:
                continue
            comps.append(np.packbits(comp))                  # store packed + shape (memory-light)
            sizes.append((comp.shape, int(comp.sum()) / brain_v))
        if (k + 1) % 200 == 0:
            print(f"pool {k+1} brains, {len(comps)} compartments", flush=True)
    order = np.argsort([s[1] for s in sizes])
    pool = {"packed": [comps[i] for i in order],
            "shapes": [sizes[i][0] for i in order],
            "sizes_frac": np.array([sizes[i][1] for i in order], dtype=np.float64)}   # ascending
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(pool, f)
    print(f"built official pool: {len(pool['packed'])} compartments -> {cache_path}", flush=True)
    return pool


class OfficialSampler:
    """Faithful on-the-fly reproduction of the official getHealthyMasks sampler."""
    def __init__(self, pool, tumor_dilation=5.0, min_dist=5.0, size_range_tol=0.1,
                 rand_points=2, min_brain_intersection=0.75, max_tries=60, size_cdf=None, dm_dir=None):
        self.shapes = pool["shapes"]
        self.packed = pool["packed"]
        self.sizes = pool["sizes_frac"]                      # ascending compartment sizes (brain fraction)
        self.tumor_dilation = float(tumor_dilation)
        self.min_dist = float(min_dist)
        self.size_range_tol = float(size_range_tol)
        self.rand_points = int(rand_points)
        self.min_bi = float(min_brain_intersection)
        self.max_tries = int(max_tries)
        self._dm = {}                                        # per-brain (distanceMap, dilated tumor) cache
        self.dm_dir = Path(dm_dir) if dm_dir else None       # precomputed dm on disk (data/precompute_dm.py)
        # SIZE-MATCHED mode: our pool is whole-tumor UNION blobs (median ~56k vox, no small tail),
        # but the eval decoys come from multi-label sub-compartments (median ~19k, floor ~800). We don't
        # have the sub-labels, so instead of the inverse-proportional selection we pick ANY shape and
        # zoom-scale it to a target VOLUME drawn from the eval size distribution (size_cdf, in voxels).
        self.size_cdf = np.asarray(size_cdf, dtype=np.float64) if size_cdf is not None else None
        self._scdf_q25 = float(np.percentile(self.size_cdf, 25)) if self.size_cdf is not None else 0.0

    def _unpack(self, i):
        sh = self.shapes[i]
        n = sh[0] * sh[1] * sh[2]
        return np.unpackbits(self.packed[i])[:n].reshape(sh).astype(bool)

    def _sample_compartment(self, tumor_frac, rng, force_small=False):
        cnt = len(self.sizes)
        if self.size_cdf is not None:                        # SIZE-MATCHED: any shape (scaled below)
            idx = int(rng.random() * cnt)
        elif force_small:                                    # fallback: small compartments always place
            idx = int(rng.random() * max(1, cnt // 4))       # smallest quartile
        else:                                                # official: inverse-proportional size window
            bi = int(np.searchsorted(self.sizes, tumor_frac))
            target = cnt - bi
            fr = max(0, int(target - self.size_range_tol * cnt / 2))
            to = min(cnt - 1, int(target + self.size_range_tol * cnt / 2))
            idx = int(fr + rng.random() * max(1, to - fr))
        idx = min(idx, cnt - 1)
        comp = self._unpack(idx)
        if self.size_cdf is not None:                        # scale to a target volume drawn from eval CDF
            tgt = float(self.size_cdf[int(rng.random() * len(self.size_cdf))])
            if force_small:                                  # placement fallback: bias toward small
                tgt = min(tgt, self._scdf_q25)
            f = float(np.clip((tgt / max(1.0, comp.sum())) ** (1.0 / 3.0), 0.12, 3.0))
            if abs(f - 1.0) > 0.02:                          # order=1 + threshold: solid, no fragmentation
                comp = ndimage.zoom(comp.astype(np.float32), f, order=1, prefilter=False) > 0.5
                if not comp.any():                           # degenerate tiny scale -> keep a seed voxel
                    comp = self._unpack(idx)
        fx, fy, fz = (1 - 2 * (rng.random(3) < 0.5)).astype(int)   # ±1 (never 0 → no zero-step slice)
        comp = comp[::fx, ::fy, ::fz]
        a1, a2 = rng.random(2) * 360
        comp = ndimage.rotate(comp, a1, axes=(0, 1), reshape=True, mode="grid-constant", order=0, prefilter=False)
        comp = ndimage.rotate(comp, a2, axes=(1, 2), reshape=True, mode="grid-constant", order=0, prefilter=False)
        return comp

    def sample_healthy(self, brain, tumor, rng, brain_key=None):
        """Return ONE official-style healthy decoy mask (bool, brain-shaped) placed away from `tumor`."""
        shape = brain.shape
        if brain_key is not None and brain_key in self._dm:
            dm, tumor_d = self._dm[brain_key]
        else:
            dm = tumor_d = None
            if self.dm_dir is not None and brain_key is not None:   # precomputed dm on disk (fast path)
                p = self.dm_dir / f"{brain_key}.npz"
                if p.exists():
                    with np.load(p) as z:
                        sh = tuple(int(x) for x in z["shape"])
                        dm = z["dm"]                          # uint8; dm>min_dist / farthest-point work as-is
                        tumor_d = np.unpackbits(z["tumor_d"])[:int(np.prod(sh))].reshape(sh).astype(bool)
            if dm is None:                                    # compute live (e.g. HCP fresh fake tumor)
                dm = ndimage.distance_transform_edt(~tumor) - self.tumor_dilation
                tumor_d = tumor | (dm < 0)                    # dilated tumor (official)
                dm[~brain] = 0
            if brain_key is not None:
                if len(self._dm) > 32:
                    self._dm.clear()
                self._dm[brain_key] = (dm, tumor_d)
        valid = np.where(dm > self.min_dist)
        if len(valid[0]) == 0:
            return np.zeros(shape, bool)
        tumor_frac = float(tumor_d.sum()) / max(1.0, float(brain.sum()))
        matched = self.size_cdf is not None
        # SIZE-MATCHED mode forces a target volume onto every brain, so a given (large) decoy may not fit
        # at the farthest-from-tumor point. Instead of shrinking (force_small collapses the distribution),
        # try MANY candidate locations per compartment and keep the target size; the away-from-tumor
        # preference is retained by trying candidates in descending distance-to-tumor order.
        npts = 24 if matched else self.rand_points
        for attempt in range(self.max_tries):
            # official sampleLocation: randPointsN points, take the one farthest from tumor.
            # After half the tries fail (official mode only), drop the farthest-bias + force small.
            small = (not matched) and attempt >= self.max_tries // 2
            ri = np.round(rng.random(npts) * (len(valid[0]) - 1)).astype(int)
            pts = [(valid[0][i], valid[1][i], valid[2][i]) for i in ri]
            comp = self._sample_compartment(tumor_frac, rng, force_small=small)
            cs = comp.shape
            if matched:                                       # try each candidate, farthest-from-tumor first
                cand = sorted(pts, key=lambda p: dm[p], reverse=True)
            else:
                cand = [pts[0] if small else max(pts, key=lambda p: dm[p])]
            placed = False
            for loc in cand:
                lo = [int(loc[a] - cs[a] // 2) for a in range(3)]
                hi = [lo[a] + cs[a] for a in range(3)]
                if any(lo[a] < 0 for a in range(3)) or any(hi[a] > shape[a] for a in range(3)):
                    continue
                sub = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
                if np.logical_and(comp, tumor_d[sub]).any():                   # no tumor overlap
                    continue
                bi = np.logical_and(comp, brain[sub]).sum() / max(1, comp.sum())
                if bi < self.min_bi:                                          # brain intersection
                    continue
                if (dm[sub][comp] <= self.min_dist).any():                    # cheap min-dist-to-tumor
                    continue
                placed = True
                break
            if not placed:
                continue
            out = np.zeros(shape, bool); out[sub] = comp                       # valid placement found
            return out
        return np.zeros(shape, bool)
