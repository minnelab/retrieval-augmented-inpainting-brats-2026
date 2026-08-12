"""Dataset for the inpainting baseline.

Uses the PROVIDED files directly (no re-voiding), so the training input distribution matches
inference exactly:
  input  = t1n-voided (healthy + tumor already zeroed) + full mask channel
  target = t1n (ground truth)
  loss is computed only on mask-healthy (the scored region; the only place GT healthy exists)

Per-case intensity scale = 99.5th percentile of nonzero t1n-voided (EDA: intensity varies ~100x).
Training crops are centered on the healthy-mask centroid (EDA: ~95% of healthy bboxes fit 128^3).
"""
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from scipy import ndimage
from torch.utils.data import Dataset


def strong_augment(chans, t, h, mask_idx, rng=None):
    """In-place-ish intensity + rotation regularization (train only). `chans` = model-input channels
    (chans[0]=voided, chans[mask_idx]=mask, others intensity); t=target image, h=loss mask.
    Small random 3D rotation (image order-1, masks order-0; voided re-derived = t*(1-mask) to stay
    aligned) + gamma intensity jitter on the image channels. Returns (chans, t, h)."""
    rng = rng or np.random
    if rng.random() < 0.5:                                        # small rotation (head-tilt)
        ang = float(rng.uniform(-12, 12)); axes = tuple(int(a) for a in rng.choice(3, 2, replace=False))
        rk = dict(axes=axes, reshape=False, mode="nearest")
        t = ndimage.rotate(t, ang, order=1, **rk)
        h = ndimage.rotate(h, ang, order=0, **rk)
        for i in range(len(chans)):
            if i != 0:                                            # skip voided (re-derived below)
                chans[i] = ndimage.rotate(chans[i], ang, order=(0 if i == mask_idx else 1), **rk)
        chans[0] = t * (1 - chans[mask_idx])                      # voided = image with the void zeroed
    g = float(np.exp(rng.normal(0, 0.08)))                        # gamma intensity jitter
    t = np.clip(t, 0, None) ** g
    for i in range(len(chans)):
        if i != mask_idx:
            chans[i] = np.clip(chans[i], 0, None) ** g
    return chans, t, h


def load(p):
    return np.asarray(nib.load(str(p)).get_fdata(), dtype=np.float32)


def per_case_scale(voided):
    nz = voided[voided > 0]
    if nz.size == 0:
        return 1.0
    s = float(np.percentile(nz, 99.5))
    return s if s > 1e-6 else 1.0


def kmeans1d(x, k=3, iters=25):
    """1D k-means on values `x` → k ASCENDING centers (e.g. CSF<GM<WM). Quantile init + Lloyd. Used by
    the tissue-prior generator (assign visible-tissue intensity classes) and the tissue-atlas builder."""
    x = np.asarray(x, np.float32).ravel()
    if x.size == 0:
        return np.zeros(k, np.float32)
    c = np.quantile(x, np.linspace(0, 1, k + 2)[1:-1]).astype(np.float32)   # init at interior quantiles
    for _ in range(iters):
        lab = np.abs(x[:, None] - c[None, :]).argmin(1)
        newc = np.array([x[lab == j].mean() if np.any(lab == j) else c[j] for j in range(k)], np.float32)
        if np.allclose(newc, c):
            break
        c = newc
    return np.sort(c)


def centered_bbox(center, crop, shape):
    """Return slices for a `crop`-sized box centered at `center`, clamped to `shape`."""
    sl = []
    for c, cs, sh in zip(center, crop, shape):
        start = int(round(c - cs / 2))
        start = max(0, min(start, sh - cs)) if sh >= cs else 0
        sl.append((start, start + cs))
    return sl


def crop_bbox_containing(target, crop, shape, rng=None):
    """Slices for a `crop`-sized box that fully contains the `target` mask bbox.
    If rng given, the offset is randomized among valid positions (translation aug) so the
    target appears at varied positions in the crop -> matches sliding-window inference.
    Falls back to centering on the target when it doesn't fit / no rng.
    """
    coords = np.where(target)
    if len(coords[0]) == 0:
        return [(max(0, (sh - cs) // 2), max(0, (sh - cs) // 2) + cs) for cs, sh in zip(crop, shape)]
    sl = []
    for ax, (cs, sh) in enumerate(zip(crop, shape)):
        lo, hi = int(coords[ax].min()), int(coords[ax].max())
        if hi - lo + 1 >= cs or sh < cs:                      # target larger than crop -> center
            start = (lo + hi) // 2 - cs // 2
        else:
            lo_s, hi_s = max(0, hi - cs + 1), min(lo, sh - cs)  # range keeping target inside crop & volume
            start = int(rng.integers(lo_s, hi_s + 1)) if (rng is not None and hi_s > lo_s) else (lo + hi) // 2 - cs // 2
        sl.append((max(0, min(start, sh - cs)) if sh >= cs else 0,) )
        sl[-1] = (sl[-1][0], sl[-1][0] + cs)
    return sl


def crop_or_pad(arr, sl, crop):
    """Crop arr by sl (list of (start,stop)); pad with zeros if the volume is smaller than crop."""
    out = np.zeros(crop, dtype=arr.dtype)
    src = tuple(slice(s, e) for s, e in sl)
    piece = arr[src]
    out[tuple(slice(0, d) for d in piece.shape)] = piece
    return out


def pad_to(arr, shape):
    """Zero-pad a 3D array up to `shape` (each axis >= current). Canonical even-shape padding for
    whole-volume Stage-1 (e.g. 240x240x155 -> 240x240x160)."""
    return np.pad(arr, [(0, max(0, s - a)) for s, a in zip(shape, arr.shape)])


def resample_to(vol, target_shape, order=1):
    """Resample a 3D array to an EXACT target_shape (order=1 trilinear for intensity, order=0 nearest
    for masks). Guards zoom's +-1 rounding by padding/cropping to the exact shape."""
    out = ndimage.zoom(vol, [t / s for t, s in zip(target_shape, vol.shape)], order=order,
                       mode="nearest", prefilter=False)
    if out.shape != tuple(target_shape):
        out = pad_to(out, target_shape)[tuple(slice(0, t) for t in target_shape)]
    return out


def enumerate_items(roots, ids=None):
    """Build the training-item list across one or more data roots, expanding mask augmentation.

    A case folder may hold either the single canonical inpainting (`{name}-t1n-voided.nii.gz`,
    suffix "") or N mask-augmented samples (`{name}-t1n-voided-0000.nii.gz`, ...; getHealthyMasks
    samplesPerBrain>1). Each (folder, name, suffix) is one independent training example sharing the
    case's `{name}-t1n.nii.gz`. `ids` (if given) restricts to those case names — used so train/val
    never mix. Pooling cohorts (GLI-x5 + GoAT) = just pass multiple roots.
    """
    keep = set(i.strip() for i in ids if i.strip()) if ids is not None else None
    items = []
    for root in ([roots] if isinstance(roots, (str, Path)) else roots):
        cases = sorted(list(Path(root).glob("BraTS*")) + list(Path(root).glob("HCP*")))  # GLI/GoAT + frozen HCP
        for d in cases:
            if not d.is_dir() or (keep is not None and d.name not in keep):
                continue
            voided = sorted(d.glob(f"{d.name}-t1n-voided*.nii.gz"))
            for p in voided:
                stem = p.name[:-len(".nii.gz")]
                suffix = stem[len(f"{d.name}-t1n-voided"):]      # "" or "-0000"
                items.append((d, d.name, suffix))
    return items


class InpaintCrops(Dataset):
    def __init__(self, roots, ids, crop=(128, 128, 128), augment=True, cache=False, prior_root=None,
                 strong_aug=False):
        self.items = enumerate_items(roots, ids)
        self.crop = tuple(crop)
        self.augment = augment
        self.strong_aug = strong_aug          # + intensity(gamma) + small-rotation aug (regularize)
        self._cache = {} if cache else None   # in-memory raw-volume cache (use only for small sets)
        # each prior_root adds one channel; multiple priors (e.g. donor + tissue) stack as extra channels
        self.prior_roots = [Path(prior_root)] if prior_root else []

    def __len__(self):
        return len(self.items)

    def _load_raw(self, item):
        d, name, suffix = item
        key = (str(d), suffix)
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        priors = [load(pr / f"{name}{suffix}-t1n-inference.nii.gz") for pr in self.prior_roots]
        raw = (load(d / f"{name}-t1n-voided{suffix}.nii.gz"),
               load(d / f"{name}-t1n.nii.gz"),
               load(d / f"{name}-mask{suffix}.nii.gz") > 0.5,
               load(d / f"{name}-mask-healthy{suffix}.nii.gz") > 0.5,
               priors)
        if self._cache is not None:
            self._cache[key] = raw
        return raw

    def __getitem__(self, idx):
        voided, t1n, mask, healthy, priors = self._load_raw(self.items[idx])

        scale = per_case_scale(voided)
        # random crop containing the healthy region when augmenting (translation robustness so
        # sliding-window inference matches training); else center on it.
        rng = np.random.default_rng() if self.augment else None
        sl = crop_bbox_containing(healthy, self.crop, voided.shape, rng=rng)

        v = crop_or_pad(voided, sl, self.crop) / scale
        t = crop_or_pad(t1n, sl, self.crop) / scale
        m = crop_or_pad(mask.astype(np.float32), sl, self.crop)
        h = crop_or_pad(healthy.astype(np.float32), sl, self.crop)
        chans = [v, m]                             # [voided, mask]
        for prior in priors:                       # -> [voided, mask, donor(, tissue)]
            chans.append(crop_or_pad(prior, sl, self.crop) / scale)

        if self.augment:
            for ax in range(3):
                if np.random.rand() < 0.5:
                    t, h = np.flip(t, ax).copy(), np.flip(h, ax).copy()
                    chans = [np.flip(c, ax).copy() for c in chans]
            if self.strong_aug:                    # intensity(gamma) + small-rotation (mask idx = 1)
                chans, t, h = strong_augment([c.copy() for c in chans], t.copy(), h.copy(), 1)

        x = np.stack(chans, axis=0)                # (2 or 3, X, Y, Z); litdiff reads [:,:1]=v [:,1:2]=m [:,2:3]=prior
        y = t[None]                                # (1, X, Y, Z)
        hm = h[None]                               # healthy mask for loss
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(hm))


class DownsampledWhole(Dataset):
    """Stage-1 (coarse, GLOBAL context) source. The WHOLE voided volume + mask are resampled to a low
    resolution (default 96x96x64 from the padded 240x240x160 -> factor ~2.5) where the entire brain
    fits the UNet receptive field and large holes shrink. Emits the SAME (x=[voided,mask], y=t1n,
    loss_mask) tuple as InpaintCrops, so LitDiffusion(in_channels=24) trains completely unchanged.
    No cropping and no prior (Stage-1 is unconditioned)."""
    def __init__(self, roots, ids, low_shape=(96, 96, 64), pad_shape=(240, 240, 160),
                 augment=True, cache=False, strong_aug=False, ds_root=None):
        self.items = enumerate_items(roots, ids)
        self.low, self.pad = tuple(low_shape), tuple(pad_shape)
        self.augment, self.strong_aug = augment, strong_aug
        self._cache = {} if cache else None
        # ds_root: dir of PRECOMPUTED {name}{suffix}.npz (eda/gen_downsampled.py) — skips the slow
        # full-res load + zoom per item (~6s -> ~0.05s). Falls back to on-the-fly if None.
        self.ds_root = Path(ds_root) if ds_root else None

    def __len__(self):
        return len(self.items)

    def _load_low(self, item):
        d, name, suffix = item
        if self.ds_root is not None:
            z = np.load(self.ds_root / f"{name}{suffix}.npz")
            return (z["v"].astype(np.float32), z["t"].astype(np.float32),
                    z["m"].astype(np.float32), z["h"].astype(np.float32))
        key = (str(d), suffix)
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        voided = load(d / f"{name}-t1n-voided{suffix}.nii.gz"); scale = per_case_scale(voided)
        down = lambda a, o: resample_to(pad_to(a, self.pad), self.low, o)
        low = (down(voided, 1) / scale,
               down(load(d / f"{name}-t1n.nii.gz"), 1) / scale,
               (down(load(d / f"{name}-mask{suffix}.nii.gz") > 0.5, 0) > 0.5).astype(np.float32),
               (down(load(d / f"{name}-mask-healthy{suffix}.nii.gz") > 0.5, 0) > 0.5).astype(np.float32))
        if self._cache is not None:
            self._cache[key] = low
        return low

    def __getitem__(self, idx):
        v, t, m, h = self._load_low(self.items[idx])           # already low-res + normalized
        chans = [v, m]
        if self.augment:
            for ax in range(3):
                if np.random.rand() < 0.5:
                    t, h = np.flip(t, ax).copy(), np.flip(h, ax).copy()
                    chans = [np.flip(c, ax).copy() for c in chans]
            if self.strong_aug:
                chans, t, h = strong_augment([c.copy() for c in chans], t.copy(), h.copy(), 1)
        x = np.stack(chans, axis=0)
        return (torch.from_numpy(x), torch.from_numpy(t[None]), torch.from_numpy(h[None]))


class HCPOnTheFly(Dataset):
    """Healthy brains (HCP in BraTS space) with tumor-shaped voids generated ON THE FLY.

    Per item we stamp two real-tumor-shaped masks into the brain (a larger tumor-like + a smaller
    healthy-like void, >=5 vox apart) via a `data.mask_sampler.MaskSampler`, then void the union.
    Because the brain is entirely healthy, BOTH voids have ground truth -> the loss mask is their
    union. Returns the same (x=[voided, mask], y=t1n, loss_mask) tuple as InpaintCrops, so it is a
    drop-in training source poolable (ConcatDataset) with real BraTS cases. Fresh masks each epoch.
    """
    def __init__(self, data_root, ids, sampler, crop=(128, 128, 128), augment=True,
                 brain_thr=0.02, cache=False, strong_aug=False, decoy_large_frac=0.0):
        self.root = Path(data_root)
        self.ids = [i.strip() for i in ids if i.strip()]
        self.sampler = sampler
        self.crop = tuple(crop)
        self.augment = augment
        self.strong_aug = strong_aug
        self.brain_thr = brain_thr
        self.decoy_large_frac = float(decoy_large_frac)   # fraction of decoys upsized (large-decoy bias)
        self._cache = {} if cache else None   # cache raw t1n (read-heavy); masks still vary per epoch

    def __len__(self):
        return len(self.ids)

    def _t1(self, name):
        if self._cache is not None and name in self._cache:
            return self._cache[name]
        t1 = load(self.root / name / f"{name}-t1n.nii.gz")
        if self._cache is not None:
            self._cache[name] = t1
        return t1

    def __getitem__(self, idx):
        name = self.ids[idx]
        t1 = self._t1(name)
        rng = np.random.default_rng()                       # fresh randomness -> new masks each epoch
        brain = t1 > self.brain_thr * t1.max()
        m_tumor, m_healthy = self.sampler.sample(brain, rng, large_frac=self.decoy_large_frac)
        union = m_tumor | m_healthy
        voided = t1 * (~union)

        scale = per_case_scale(voided)
        crng = rng if self.augment else None
        sl = crop_bbox_containing(union, self.crop, t1.shape, rng=crng)
        v = crop_or_pad(voided, sl, self.crop) / scale
        t = crop_or_pad(t1, sl, self.crop) / scale
        m = crop_or_pad(union.astype(np.float32), sl, self.crop)     # mask channel = region to fill
        if self.augment:
            for ax in range(3):
                if np.random.rand() < 0.5:
                    v, t, m = (np.flip(a, ax).copy() for a in (v, t, m))
            if self.strong_aug:                    # loss mask = union = m (mask idx 1)
                (v, m), t, _ = strong_augment([v.copy(), m.copy()], t.copy(), m.copy(), 1)

        x = np.stack([v, m], axis=0)
        y = t[None]
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m[None]))   # loss on union


class OnTheFlyBigHole(Dataset):
    """On-the-fly masks with a BIG-hole-biased healthy decoy, to target the large-decoy score cap.

    Each epoch draws a FRESH healthy decoy whose size is log-uniform in `size_range` (biased large vs
    the natural ~17k median) -> more practice on the big holes that dominate the rank-sum. For real
    tumour-bearing brains (GLI/GoAT: `{name}-mask-unhealthy.nii.gz` present) the true tumour is the
    unscored part of the void and the LOSS is on the healthy decoy only; for tumour-free brains (HCP)
    a fake tumour is stamped and the loss is the union (all healthy GT). Same (x=[voided,mask], y, hm)
    tuple as InpaintCrops -> poolable, warm-startable base model, no prior channel."""
    def __init__(self, items, sampler, crop=(128, 128, 128), size_range=(25000, 120000),
                 augment=True, brain_thr=0.02, cache=False):
        self.items = list(items)                 # [(dir, name), ...] (repeat for oversample)
        self.sampler = sampler
        self.crop = tuple(crop)
        self.size_range = tuple(size_range)
        self.augment = augment
        self.brain_thr = brain_thr
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.items)

    def _t1(self, d, name):
        if self._cache is not None and name in self._cache:
            return self._cache[name]
        t1 = load(d / f"{name}-t1n.nii.gz")
        if self._cache is not None:
            self._cache[name] = t1
        return t1

    def __getitem__(self, idx):
        d, name = self.items[idx]; d = Path(d)
        t1 = self._t1(d, name)
        rng = np.random.default_rng()                       # fresh mask each epoch
        brain = t1 > self.brain_thr * t1.max()
        tf = d / f"{name}-mask-unhealthy.nii.gz"
        has_tumor = tf.exists()
        if has_tumor:
            tumor = load(tf) > 0.5
        else:                                               # HCP: stamp a fake tumour-shaped void
            cand = np.argwhere(brain)
            t_tgt = int(self.sampler.t_sizes[rng.integers(len(self.sampler.t_sizes))])
            tumor = self.sampler._stamp(brain, cand, t_tgt, rng)
            tumor = tumor if tumor is not None else np.zeros(brain.shape, bool)
        avoid = ndimage.binary_dilation(tumor, iterations=int(round(self.sampler.min_dist))) if tumor.any() else None
        decoy = self.sampler.sample_healthy(brain, rng, avoid=avoid, size_range=self.size_range)
        union = tumor | decoy
        voided = t1 * (~union)
        loss = decoy if has_tumor else union                # GLI/GoAT: score only the healthy decoy

        scale = per_case_scale(voided)
        crng = rng if self.augment else None
        sl = crop_bbox_containing(decoy if decoy.any() else union, self.crop, t1.shape, rng=crng)
        v = crop_or_pad(voided, sl, self.crop) / scale
        t = crop_or_pad(t1, sl, self.crop) / scale
        m = crop_or_pad(union.astype(np.float32), sl, self.crop)
        h = crop_or_pad(loss.astype(np.float32), sl, self.crop)
        if self.augment:
            for ax in range(3):
                if np.random.rand() < 0.5:
                    v, t, m, h = (np.flip(a, ax).copy() for a in (v, t, m, h))
        x = np.stack([v, m], axis=0)
        return (torch.from_numpy(x), torch.from_numpy(t[None]), torch.from_numpy(h[None]))


class OnTheFlyRetrHCP(Dataset):
    """HCP with FRESH on-the-fly masks + a blurred-retrieval prior built LIVE from a precomputed
    per-brain best donor (eda/gen_hcp_best_donor.py). Each epoch: sample a mask, paste the (fixed)
    best-donor tissue into the fresh hole, blur (sigma) -> the same blurred-retrieval prior schema as
    the frozen GLI priors, but with fresh masks. Returns [voided, mask, prior] (3ch), loss on the
    union -> poolable (ConcatDataset) with InpaintCrops(prior_root=...) for the retrieval model."""
    def __init__(self, data_root, ids, sampler, best_donor_npz, crop=(128, 128, 128), sigma=4.0,
                 augment=True, brain_thr=0.02, decoy_large_frac=0.0):
        self.root = Path(data_root)
        self.ids = [i.strip() for i in ids if i.strip()]
        self.sampler = sampler
        self.crop = tuple(crop)
        self.sigma = float(sigma)
        self.augment = augment
        self.brain_thr = brain_thr
        self.decoy_large_frac = float(decoy_large_frac)   # fraction of decoys upsized (large-decoy bias)
        z = np.load(best_donor_npz, allow_pickle=True)
        self.donor_path = {str(n): str(p) for n, p in zip(z["names"], z["donor_paths"])}
        self.donor_scale = {str(n): float(s) for n, s in zip(z["names"], z["donor_scales"])}
        self._t1c, self._dc = {}, {}          # bounded caches (per worker)

    def __len__(self):
        return len(self.ids)

    def _t1(self, name):
        if name not in self._t1c:
            if len(self._t1c) > 48:
                self._t1c.clear()
            self._t1c[name] = load(self.root / name / f"{name}-t1n.nii.gz")
        return self._t1c[name]

    def _donor(self, name):
        if name not in self._dc:
            if len(self._dc) > 24:
                self._dc.clear()
            self._dc[name] = load(Path(self.donor_path[name])) / self.donor_scale[name]  # normalized donor
        return self._dc[name]

    def __getitem__(self, idx):
        name = self.ids[idx]
        t1 = self._t1(name)
        rng = np.random.default_rng()
        brain = t1 > self.brain_thr * t1.max()
        m_tumor, m_healthy = self.sampler.sample(brain, rng, large_frac=self.decoy_large_frac)
        union = m_tumor | m_healthy
        voided = t1 * (~union)
        scale = per_case_scale(voided)
        retr = voided.copy()
        retr[union] = self._donor(name)[union].astype(np.float32) * scale        # hole <- donor tissue
        prior = ndimage.gaussian_filter(retr, self.sigma).astype(np.float32)      # blurred-retrieval prior
        crng = rng if self.augment else None
        sl = crop_bbox_containing(union, self.crop, t1.shape, rng=crng)
        v = crop_or_pad(voided, sl, self.crop) / scale
        t = crop_or_pad(t1, sl, self.crop) / scale
        m = crop_or_pad(union.astype(np.float32), sl, self.crop)
        pr = crop_or_pad(prior, sl, self.crop) / scale
        if self.augment:
            for ax in range(3):
                if np.random.rand() < 0.5:
                    v, t, m, pr = (np.flip(a, ax).copy() for a in (v, t, m, pr))
        x = np.stack([v, m, pr], axis=0)                                          # [voided, mask, prior]
        return (torch.from_numpy(x), torch.from_numpy(t[None]), torch.from_numpy(m[None]))   # loss on union


def read_ids(path):
    return Path(path).read_text().splitlines()


class OnTheFlyGLI(Dataset):
    """GLI (glioma) with FRESH on-the-fly synthetic HEALTHY decoys each epoch (mask augmentation, the
    2025-winner lever) + a merged-retrieval prior from a precomputed per-brain merged donor volume
    (eda/gen_merged_donor.py). Keeps the REAL tumor void (mask-unhealthy) and samples a fresh healthy
    decoy AVOIDING the dilated tumor (so its GT is genuinely healthy). Voids tumor|decoy, prior =
    merged donor pasted in (unblurred if sigma<=0). LOSS on the HEALTHY DECOY ONLY (tumor fill is
    unsupervised, matching the frozen GLI path + BraTS scoring where the tumor is unscored).
    Returns [voided, union_mask, prior], y=t1n, hm=decoy -> poolable with OnTheFlyRetrHCP/InpaintCrops."""
    def __init__(self, data_root, ids, sampler, merged_donor_npz, crop=(128, 128, 128), sigma=0.0,
                 augment=True, brain_thr=0.02, min_dist=5.0):
        self.root = Path(data_root)
        self.ids = [i.strip() for i in ids if i.strip()]
        self.sampler = sampler
        self.crop = tuple(crop)
        self.sigma = float(sigma)
        self.augment = augment
        self.brain_thr = brain_thr
        self.min_dist = float(min_dist)
        z = np.load(merged_donor_npz, allow_pickle=True)
        self.donor_path = {str(n): str(p) for n, p in zip(z["names"], z["donor_paths"])}
        self.donor_scale = {str(n): float(s) for n, s in zip(z["names"], z["donor_scales"])}
        self._t1c, self._tuc, self._dc = {}, {}, {}

    def __len__(self):
        return len(self.ids)

    def _t1(self, name):
        if name not in self._t1c:
            if len(self._t1c) > 48:
                self._t1c.clear()
            self._t1c[name] = load(self.root / name / f"{name}-t1n.nii.gz")
        return self._t1c[name]

    def _tumor(self, name):
        if name not in self._tuc:
            if len(self._tuc) > 48:
                self._tuc.clear()
            self._tuc[name] = load(self.root / name / f"{name}-mask-unhealthy.nii.gz") > 0.5
        return self._tuc[name]

    def _donor(self, name):
        if name not in self._dc:
            if len(self._dc) > 24:
                self._dc.clear()
            self._dc[name] = load(Path(self.donor_path[name])) / self.donor_scale[name]
        return self._dc[name]

    def __getitem__(self, idx):
        name = self.ids[idx]
        t1 = self._t1(name)
        tumor = self._tumor(name)
        rng = np.random.default_rng()
        brain = t1 > self.brain_thr * t1.max()
        avoid = ndimage.binary_dilation(tumor, iterations=int(round(self.min_dist)))
        healthy = None
        for _ in range(8):
            healthy = self.sampler.sample_healthy(brain, rng, avoid=avoid)
            if healthy.sum() > 50:
                break
        union = tumor | healthy
        voided = t1 * (~union)
        scale = per_case_scale(voided)
        retr = voided.copy()
        retr[union] = self._donor(name)[union].astype(np.float32) * scale
        prior = ndimage.gaussian_filter(retr, self.sigma).astype(np.float32) if self.sigma > 0 else retr
        crng = rng if self.augment else None
        sl = crop_bbox_containing(healthy, self.crop, t1.shape, rng=crng)
        v = crop_or_pad(voided, sl, self.crop) / scale
        t = crop_or_pad(t1, sl, self.crop) / scale
        m = crop_or_pad(union.astype(np.float32), sl, self.crop)
        h = crop_or_pad(healthy.astype(np.float32), sl, self.crop)
        pr = crop_or_pad(prior, sl, self.crop) / scale
        if self.augment:
            for ax in range(3):
                if np.random.rand() < 0.5:
                    v, t, m, h, pr = (np.flip(a, ax).copy() for a in (v, t, m, h, pr))
        x = np.stack([v, m, pr], axis=0)
        return (torch.from_numpy(x), torch.from_numpy(t[None]), torch.from_numpy(h[None]))
