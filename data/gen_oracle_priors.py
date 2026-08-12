"""ORACLE prior: blurred GROUND-TRUTH T1 -> theoretical ceiling of a perfect coarse prior.
Decisive test of "priors closed vs retrieval-quality-limited": if even a perfect blurred-GT prior
does not lift val SSIM above the unconditioned 0.872, coarse priors are genuinely closed (the model
already reconstructs the low-freq from context). Mirrors retr_priors/ filenames so the loader is
unchanged. GT t1n is one file per case (x5 aug varies only the mask), so blur once, symlink the 5."""
import argparse, glob, os
from multiprocessing import Pool
import numpy as np, nibabel as nib
from scipy import ndimage

SIGMA = 4.0; GTROOT = ""; CACHE = ""


def case_of(fn, val):
    b = os.path.basename(fn).replace("-t1n-inference.nii.gz", "")
    if val:
        return b, b                      # val: prior name == case id
    return b.rsplit("-", 1)[0], b        # train: strip trailing -XXXX aug index


def blur_case(case):
    out = os.path.join(CACHE, case + ".nii.gz")
    if os.path.exists(out):
        return 0
    img = nib.load(os.path.join(GTROOT, case, case + "-t1n.nii.gz"))
    v = np.squeeze(np.asarray(img.get_fdata(), dtype=np.float32))
    vb = ndimage.gaussian_filter(v, SIGMA).astype(np.float32)
    nib.save(nib.Nifti1Image(vb, img.affine, img.header), out)
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)        # retr_priors/{train|val} to mirror names
    ap.add_argument("--gt-root", required=True)     # DATA_X5 (train) or DATA_GLI (val)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    SIGMA = a.sigma; GTROOT = a.gt_root
    dst = os.path.abspath(a.dst); CACHE = dst + "/_cache"
    os.makedirs(CACHE, exist_ok=True)
    files = sorted(glob.glob(f"{a.ref}/*-t1n-inference.nii.gz"))
    mapping = [case_of(f, a.val) for f in files]
    cases = sorted(set(c for c, _ in mapping))
    with Pool(a.workers) as p:
        done = sum(p.map(blur_case, cases, chunksize=4))
    n = 0
    for case, full in mapping:
        link = os.path.join(dst, full + "-t1n-inference.nii.gz")
        if not os.path.exists(link):
            os.symlink(os.path.join(CACHE, case + ".nii.gz"), link); n += 1
    print(f"oracle: blurred {done}/{len(cases)} cases, linked {n}/{len(files)} sigma={a.sigma} -> {dst}", flush=True)
