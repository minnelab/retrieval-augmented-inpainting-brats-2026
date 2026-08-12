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


def load(p):
    return np.asarray(nib.load(str(p)).get_fdata(), dtype=np.float32)


def per_case_scale(voided):
    nz = voided[voided > 0]
    if nz.size == 0:
        return 1.0
    s = float(np.percentile(nz, 99.5))
    return s if s > 1e-6 else 1.0


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


class InpaintCrops(Dataset):
    def __init__(self, data_root, ids, crop=(128, 128, 128), augment=True, cache=False):
        self.root = Path(data_root)
        self.ids = [i.strip() for i in ids if i.strip()]
        self.crop = tuple(crop)
        self.augment = augment
        self._cache = {} if cache else None   # in-memory raw-volume cache (use only for small sets)

    def __len__(self):
        return len(self.ids)

    def _sample_paths(self, name, rng):
        """Paths to (voided, t1n, mask, healthy) for one sample.
        Mask-augmented datasets (data/generate_masks.py --samples-per-brain N>1) write one
        decoy per index as `{name}-mask-healthy-NNNN.nii.gz` (+ matching mask/voided). When
        present we pick one index (random under aug, else 0); otherwise fall back to the
        original single-mask filenames. t1n is shared across indices.
        """
        d = self.root / name
        variants = sorted(d.glob(f"{name}-mask-healthy-*.nii.gz"))
        if variants:
            i = int(rng.integers(len(variants))) if rng is not None else 0
            sfx = f"-{i:04d}"
        else:
            sfx = ""
        return (d / f"{name}-t1n-voided{sfx}.nii.gz", d / f"{name}-t1n.nii.gz",
                d / f"{name}-mask{sfx}.nii.gz", d / f"{name}-mask-healthy{sfx}.nii.gz")

    def _load_raw(self, name, rng):
        vp, tp, mp, hp = self._sample_paths(name, rng)
        key = (name, vp.name)
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        raw = (load(vp), load(tp), load(mp) > 0.5, load(hp) > 0.5)
        if self._cache is not None:
            self._cache[key] = raw
        return raw

    def __getitem__(self, idx):
        name = self.ids[idx]
        # one rng drives both the decoy-variant pick and the crop offset (aug only).
        rng = np.random.default_rng() if self.augment else None
        voided, t1n, mask, healthy = self._load_raw(name, rng)

        scale = per_case_scale(voided)
        # random crop containing the healthy region when augmenting (translation robustness so
        # sliding-window inference matches training); else center on it.
        sl = crop_bbox_containing(healthy, self.crop, voided.shape, rng=rng)

        v = crop_or_pad(voided, sl, self.crop) / scale
        t = crop_or_pad(t1n, sl, self.crop) / scale
        m = crop_or_pad(mask.astype(np.float32), sl, self.crop)
        h = crop_or_pad(healthy.astype(np.float32), sl, self.crop)

        if self.augment:
            for ax in range(3):
                if np.random.rand() < 0.5:
                    v, t, m, h = (np.flip(a, ax).copy() for a in (v, t, m, h))

        x = np.stack([v, m], axis=0)              # (2, X, Y, Z)
        y = t[None]                                # (1, X, Y, Z)
        hm = h[None]                               # healthy mask for loss
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(hm))


def winner_norm(t1n):
    """2025-winner normalization: clip to [0.1, 99.9] pct, /max -> [0,1], then *2-1 -> [-1,1]."""
    lo, hi = np.quantile(t1n, 0.001), np.quantile(t1n, 0.999)
    t = np.clip(t1n, lo, hi)
    mx = float(t.max())
    t = t / mx if mx > 1e-6 else t
    return (t * 2 - 1).astype(np.float32), mx     # also return max for denorm at inference


class InpaintWinner(Dataset):
    """Faithful 2025-winner training data path.

    Differences from InpaintCrops, all read from the winner's container code:
      - normalization: clip[0.1,99.9]/max then [-1,1] (vs our /99.5pct, unbounded);
      - voids ONLY the healthy decoy at train (tumor stays visible as context) -> input =
        norm(t1n*(1-healthy)) with the hole at -1; mask channel = healthy mask;
      - enumerates ALL N mask variants per brain (1 epoch sees every mask), not 1 random/brain;
      - no on-the-fly image flips (diversity comes from the N offline mirror/rotate masks).
    Target = norm(t1n); loss is on the healthy mask. Inference (infer.py --data-style winner)
    uses the union mask + both-voided input, matching the winner's train/infer setup.
    """
    def __init__(self, data_root, ids, crop=(208, 208, 144), augment=False, match_infer=False,
                 synth_sampler=None, synth_p=0.0, synth_size=(4000, 45000), brain_thr=0.02,
                 merged_donor_npz=None, prior_root=None):
        self.root = Path(data_root)
        self.crop = tuple(crop)
        self.augment = augment                     # on-the-fly flips + intensity jitter (regularize)
        # MERGED RETRIEVAL PRIOR (3rd input channel). Two mutually-exclusive sources:
        #   merged_donor_npz: per-brain (mask-independent) merged-top-K donor volume; pasted into THIS
        #     item's void -> native prior -> winner_norm (matches infer). Used for TRAIN (our masks).
        #   prior_root: a frozen dir of native {name}-t1n-inference{sfx}.nii.gz priors -> winner_norm.
        #     Used for VAL (merged_priors/val, built from the official voided).
        self.prior_root = Path(prior_root) if prior_root else None
        self.donor_map, self._dc = None, {}
        if merged_donor_npz:
            z = np.load(merged_donor_npz, allow_pickle=True)
            self.donor_map = {str(n): (str(p), float(s)) for n, p, s
                              in zip(z["names"], z["donor_paths"], z["donor_scales"])}
        # match_infer: void the UNION (tumor+decoy) with union mask channel -> matches the test-time
        # distribution (infer.py loads t1n-voided + union mask). Default False replicates the winner
        # (voids only the decoy, healthy mask channel) but mismatches inference.
        self.match_infer = match_infer
        # MASK AUGMENTATION (winner's headline lever): with prob synth_p, ignore the official decoy
        # and stamp a FRESH synthetic healthy void into real GLI healthy tissue (avoiding the real
        # tumor), sized log-uniform in synth_size to match val's smaller decoys. Void = synth∪tumor
        # (matches inference); loss is on the synthetic void (real T1 there = valid GT). Unbounded
        # mask variety per epoch -> closes the train/val mask-size gap.
        self.synth_sampler = synth_sampler
        self.synth_p = float(synth_p)
        self.synth_size = tuple(synth_size)
        self.brain_thr = float(brain_thr)
        self.items = []                            # (brain, variant-suffix)
        for i in [s.strip() for s in ids if s.strip()]:
            variants = sorted((self.root / i).glob(f"{i}-mask-healthy-*.nii.gz"))
            if variants:
                for v in variants:
                    self.items.append((i, v.name[len(i) + len("-mask-healthy"):-len(".nii.gz")]))
            else:
                self.items.append((i, ""))         # single-mask layout fallback

    def __len__(self):
        return len(self.items)

    def _donor(self, name):                        # per-brain merged donor volume (normalized), cached
        if name not in self._dc:
            if len(self._dc) > 24:
                self._dc.clear()
            p, s = self.donor_map[name]
            self._dc[name] = load(p) / (s if s > 1e-6 else 1.0)
        return self._dc[name]

    def __getitem__(self, idx):
        name, sfx = self.items[idx]
        d = self.root / name
        rng = np.random.default_rng()
        t1n = load(d / f"{name}-t1n.nii.gz")
        use_synth = self.synth_sampler is not None and rng.random() < self.synth_p
        if use_synth:                               # synthesize a fresh healthy void in real tissue
            tp = d / f"{name}-mask-unhealthy.nii.gz"
            tumor = (load(tp) > 0.5) if tp.exists() else np.zeros(t1n.shape, bool)
            brain = t1n > self.brain_thr * t1n.max()
            avoid = (ndimage.binary_dilation(tumor, iterations=int(round(self.synth_sampler.min_dist)))
                     if tumor.any() else None)      # keep the synthetic decoy ≥min_dist from the tumor
            healthy = self.synth_sampler.sample_healthy(brain, rng, avoid=avoid, size_range=self.synth_size)
            if healthy.sum() < 50:                  # sampler failed -> fall back to the official decoy
                use_synth = False
        if not use_synth:
            healthy = load(d / f"{name}-mask-healthy{sfx}.nii.gz") > 0.5
        # mask channel / void region
        if use_synth:
            void_mask = healthy | tumor             # void synth decoy + real tumor (matches inference)
        elif self.match_infer:
            mp = d / f"{name}-mask{sfx}.nii.gz"
            void_mask = (load(mp) > 0.5) if mp.exists() else healthy   # official union mask
        else:
            void_mask = healthy
        tn, _ = winner_norm(t1n)                    # [-1,1]
        # merged retrieval prior (native = t1n outside void, donor guess inside), winner_normed like
        # t1n so it matches the inference path (infer.py winner_norms the frozen native val prior).
        pn = None
        if self.donor_map is not None:              # TRAIN: paste per-brain donor into THIS void
            vsc = per_case_scale(t1n * (~void_mask))
            pv = t1n.copy(); pv[void_mask] = self._donor(name)[void_mask].astype(np.float32) * vsc
            pn, _ = winner_norm(pv)
        elif self.prior_root is not None:           # VAL: frozen native merged prior
            pv = load(self.prior_root / f"{name}-t1n-inference{sfx}.nii.gz")
            pn, _ = winner_norm(pv)
        sl = crop_bbox_containing(healthy, self.crop, t1n.shape, rng=rng)
        t = crop_or_pad(tn, sl, self.crop)         # target (normalized full)
        h = crop_or_pad(healthy.astype(np.float32), sl, self.crop)    # loss mask (always healthy)
        mk = crop_or_pad(void_mask.astype(np.float32), sl, self.crop)  # input mask channel / void
        pr = crop_or_pad(pn, sl, self.crop) if pn is not None else None  # prior channel [-1,1]
        if self.augment:
            for ax in range(3):                    # random flips (image+masks+prior together)
                if rng.random() < 0.5:
                    t, h, mk = np.flip(t, ax).copy(), np.flip(h, ax).copy(), np.flip(mk, ax).copy()
                    if pr is not None:
                        pr = np.flip(pr, ax).copy()
            g = float(np.exp(rng.normal(0, 0.05)))  # mild intensity jitter (gamma-ish on [0,1])
            t01 = np.clip((t + 1) / 2, 0, 1) ** g
            t = (t01 * 2 - 1).astype(np.float32)
        v = ((t + 1) / 2) * (1 - mk) * 2 - 1       # void the mask region -> -1 (matches infer)
        chans = [v, mk] if pr is None else [v, mk, pr]
        x = np.stack(chans, axis=0)                # [voided(-1 hole), mask(, prior)]
        return (torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(t[None].astype(np.float32)),
                torch.from_numpy(h[None].astype(np.float32)))   # loss on healthy mask


class HCPOnTheFly(Dataset):
    """Healthy HCP brains (in BraTS space) with tumor-shaped voids generated ON THE FLY.

    Per item, stamp two real-tumor-shaped voids (a larger tumor-like + a smaller healthy-like,
    >=5 vox apart) into the brain via a `data.mask_sampler.MaskSampler`, then void their union.
    The brain is entirely healthy so BOTH voids have ground truth -> loss mask = union (supervise
    both). On-the-fly generation gives unbounded mask variety per epoch with no files on disk.

    style="winner": output matches InpaintWinner ([-1,1] norm, hole at -1) so it pools directly with
    the BraTS winner-pipeline dataset. style="ours": /percentile [0,1] (InpaintCrops format).
    Returns (x=[voided, union_mask], y=t1n, union_mask) like the other datasets.
    """
    def __init__(self, data_root, ids, sampler, crop=(208, 208, 144), augment=True,
                 style="winner", brain_thr=0.02, cache=False, merged_donor_npz=None):
        self.root = Path(data_root)
        self.ids = [i.strip() for i in ids if i.strip()]
        self.sampler = sampler
        self.crop = tuple(crop)
        self.augment = augment
        self.style = style
        self.brain_thr = brain_thr
        self._cache = {} if cache else None
        self.donor_map, self._dc = None, {}         # per-brain merged donor -> retrieval prior channel
        if merged_donor_npz:
            z = np.load(merged_donor_npz, allow_pickle=True)
            self.donor_map = {str(n): (str(p), float(s)) for n, p, s
                              in zip(z["names"], z["donor_paths"], z["donor_scales"])}

    def _donor(self, name):
        if name not in self._dc:
            if len(self._dc) > 24:
                self._dc.clear()
            p, s = self.donor_map[name]
            self._dc[name] = load(p) / (s if s > 1e-6 else 1.0)
        return self._dc[name]

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
        m_tumor, m_healthy = self.sampler.sample(brain, rng)
        union = m_tumor | m_healthy
        crng = rng if self.augment else None
        sl = crop_bbox_containing(union, self.crop, t1.shape, rng=crng)
        m = crop_or_pad(union.astype(np.float32), sl, self.crop)

        pr = None
        if self.style == "winner":
            tn, _ = winner_norm(t1)                          # [-1,1] full
            t = crop_or_pad(tn, sl, self.crop)
            v = ((t + 1) / 2) * (1 - m) * 2 - 1              # void union -> -1 (matches InpaintWinner)
            if self.donor_map is not None:                   # merged donor pasted into the void
                vsc = per_case_scale(t1 * (~union))
                pv = t1.copy(); pv[union] = self._donor(name)[union].astype(np.float32) * vsc
                pn, _ = winner_norm(pv)
                pr = crop_or_pad(pn, sl, self.crop)
        else:
            scale = per_case_scale(t1 * (~union))
            t = crop_or_pad(t1, sl, self.crop) / scale
            v = crop_or_pad((t1 * (~union)), sl, self.crop) / scale

        if self.augment:
            for ax in range(3):
                if np.random.rand() < 0.5:
                    v, t, m = (np.flip(a, ax).copy() for a in (v, t, m))
                    if pr is not None:
                        pr = np.flip(pr, ax).copy()
        chans = [v, m] if pr is None else [v, m, pr]
        x = np.stack(chans, axis=0)
        return (torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(t[None].astype(np.float32)),
                torch.from_numpy(m[None].astype(np.float32)))


def read_ids(path):
    return Path(path).read_text().splitlines()
