"""Generate K healthy decoys per val case -> a multi-mask robustness eval set on held-out val-150.

We have full ground truth (t1n) for val-150, so any healthy decoy we void can be scored. For each id,
K times: sample a healthy decoy (MaskSampler: real BraTS shapes, size-matched to val, placed in
healthy tissue >=5 vox from the tumor), then write a standalone eval "case" {id}-m{k}/ with:
  -t1n.nii.gz          (ground truth, copied)
  -t1n-voided.nii.gz   (t1n with decoy∪tumor zeroed = model input)
  -mask.nii.gz         (decoy∪tumor union = the void the model fills)
  -mask-healthy.nii.gz (the decoy = the SCORED region)
infer.py then runs over these exactly like the official val, giving 150*K
samples -> a robust mean +/- std and mask-placement/size sensitivity (vs the single official decoy).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mask_sampler import load_or_build_pool, MaskSampler  # noqa: E402


def load(p):
    return np.asarray(nib.load(str(p)).get_fdata(), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="raw val cases (t1n + mask-unhealthy)")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--pool-cache", default="data/hcp_shape_pool.pkl")
    ap.add_argument("-k", "--per-brain", type=int, default=5)
    ap.add_argument("--size-lo", type=int, default=4000)
    ap.add_argument("--size-hi", type=int, default=45000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ids-out", default=None, help="write the generated {id}-m{k} id list here")
    a = ap.parse_args()

    pool = load_or_build_pool(a.pool_cache, brats_root=a.data_root)
    sampler = MaskSampler(pool)
    ids = [x.strip() for x in open(a.ids) if x.strip()]
    out = Path(a.out_root); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    written = []
    for cid in ids:
        d = Path(a.data_root) / cid
        t1_img = nib.load(str(d / f"{cid}-t1n.nii.gz"))
        t1 = np.asarray(t1_img.get_fdata(), dtype=np.float32)
        tumor = load(d / f"{cid}-mask-unhealthy.nii.gz") > 0.5
        brain = t1 > 0.02 * t1.max()
        avoid = ndimage.binary_dilation(tumor, iterations=5) if tumor.any() else None
        aff, hdr = t1_img.affine, t1_img.header
        for k in range(a.per_brain):
            decoy = sampler.sample_healthy(brain, rng, avoid=avoid, size_range=(a.size_lo, a.size_hi))
            if decoy.sum() < 50:
                continue
            union = decoy | tumor
            voided = t1 * (~union)
            name = f"{cid}-m{k}"
            cd = out / name; cd.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(t1, aff, hdr), cd / f"{name}-t1n.nii.gz")
            nib.save(nib.Nifti1Image(voided, aff, hdr), cd / f"{name}-t1n-voided.nii.gz")
            nib.save(nib.Nifti1Image(union.astype(np.uint8), aff, hdr), cd / f"{name}-mask.nii.gz")
            nib.save(nib.Nifti1Image(decoy.astype(np.uint8), aff, hdr), cd / f"{name}-mask-healthy.nii.gz")
            written.append(name)
        print(f"{cid}: {sum(1 for w in written if w.startswith(cid))} decoys", flush=True)

    if a.ids_out:
        Path(a.ids_out).write_text("\n".join(written) + "\n")
    print(f"wrote {len(written)} multi-mask eval cases ({len(ids)} brains x ~{a.per_brain}) -> {out}")


if __name__ == "__main__":
    main()
