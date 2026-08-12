"""Blur retrieval priors -> reliable coarse conditioning (the blurred-prior experiment).
Under the hole, raw retrieval corr is 0.52 but blurred (sigma~4) it is 0.93 — the low-freq is
trustworthy, the high-freq is wrong. So we condition on the BLURRED retrieval prior."""
import argparse, glob, os
from multiprocessing import Pool
import numpy as np, nibabel as nib
from scipy import ndimage

SIGMA = 4.0
DST = ""


def one(f):
    out = os.path.join(DST, os.path.basename(f))
    if os.path.exists(out):
        return 0
    img = nib.load(f)
    v = np.squeeze(np.asarray(img.get_fdata(), dtype=np.float32))
    vb = ndimage.gaussian_filter(v, SIGMA).astype(np.float32)
    nib.save(nib.Nifti1Image(vb, img.affine, img.header), out)
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)      # retr_priors/{train|val}
    ap.add_argument("--dst", required=True)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    SIGMA = a.sigma; DST = a.dst
    os.makedirs(DST, exist_ok=True)
    files = sorted(glob.glob(f"{a.src}/*.nii.gz"))
    with Pool(a.workers) as p:
        done = sum(p.map(one, files, chunksize=8))
    print(f"blurred {done} new (of {len(files)}) sigma={a.sigma} workers={a.workers} -> {DST}", flush=True)
