"""Generalized per-brain MERGED top-K retrieval donor VOLUME precompute (mask-independent) for ANY
cohort (GLI, GoAT, HCP), so on-the-fly fresh-mask training can paste a merged prior into any hole.
Same as gen_hcp_merged_donor but with --root/--prefix/--ids so it works for BraTS-GLI-*/BraTS-GoAT-*.
Inline embed (no torch), login-runnable. Writes merged volumes + npz manifest
{names, donor_paths, donor_scales(=1.0)} matching the OnTheFlyRetrHCP donor schema."""
import argparse, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np, nibabel as nib
from scipy import ndimage

EMB = (24, 24, 16)
G = {}


def load(p):
    return np.asarray(nib.load(str(p)).get_fdata(), dtype=np.float32)


def pct995(v):
    s = float(np.percentile(v, 99.5))
    return s if s > 1e-6 else 1.0


def embed(vol):
    s = pct995(vol); v = vol / s
    f = [EMB[i] / vol.shape[i] for i in range(3)]
    d = ndimage.zoom(v, f, order=1).ravel().astype(np.float32)
    return (d - d.mean()) / (d.std() + 1e-6)


def worker(name):
    try:
        K = G["k"]; sl = G["shortlist"]
        out = Path(G["vol"]) / (name + "-merged.nii.gz")
        if out.exists():
            return name
        img = nib.load(str(Path(G["root"]) / name / (name + "-t1n.nii.gz")))
        t1 = np.asarray(img.get_fdata(), dtype=np.float32)
        scale = pct995(t1); ctx = t1 > 0                     # whole-brain query (mask-independent)
        q = embed(t1); q = q / (np.linalg.norm(q) + 1e-6)
        order = np.argsort(-(G["embn"] @ q))
        tgt = (t1 / scale)[ctx].astype(np.float32)
        cands = []; seen = 0
        for j in order:
            if G["names"][j] == name:
                continue
            dn = load(G["paths"][j]); dn = dn / pct995(dn)
            mse = float(np.mean((tgt - dn[ctx].astype(np.float32)) ** 2))
            cands.append((mse, dn)); seen += 1
            if seen >= sl:
                break
        cands.sort(key=lambda c: c[0])
        merged = np.mean([c[1] for c in cands[:K]], axis=0).astype(np.float32)
        nib.save(nib.Nifti1Image(merged, img.affine, img.header), out)
        return name
    except Exception as e:
        print("ERR", name, repr(e), flush=True); return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--ids", default=None, help="optional id list to restrict to")
    ap.add_argument("--index", required=True)
    ap.add_argument("--vol-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--shortlist", type=int, default=15)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    idx = np.load(a.index, allow_pickle=True)
    G["paths"] = idx["paths"]; emb = idx["emb"]
    G["embn"] = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6)
    G["names"] = [Path(p).parent.name for p in idx["paths"]]
    G["root"] = a.root; G["vol"] = a.vol_dir; G["k"] = a.k; G["shortlist"] = a.shortlist
    os.makedirs(a.vol_dir, exist_ok=True)
    if a.ids:
        ids = [x.strip() for x in open(a.ids) if x.strip()]
    else:
        ids = sorted(p.name for p in Path(a.root).glob(a.prefix + "*") if p.is_dir())
    print("merged-donor for %d brains (prefix=%s) k=%d -> %s" % (len(ids), a.prefix, a.k, a.vol_dir), flush=True)
    done = []
    with Pool(a.workers) as p:
        for i, n in enumerate(p.imap_unordered(worker, ids, chunksize=2)):
            if n: done.append(n)
            if (i + 1) % 100 == 0:
                print("  %d/%d" % (i + 1, len(ids)), flush=True)
    names = sorted(done)
    paths = [str(Path(a.vol_dir) / (n + "-merged.nii.gz")) for n in names]
    np.savez(a.out, names=np.array(names), donor_paths=np.array(paths),
             donor_scales=np.ones(len(names), dtype=np.float32))
    print("wrote merged-donor manifest for %d brains -> %s" % (len(names), a.out), flush=True)


if __name__ == "__main__":
    main()
