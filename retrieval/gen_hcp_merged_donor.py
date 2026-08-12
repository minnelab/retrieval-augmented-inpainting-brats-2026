"""Precompute a per-HCP-brain MERGED top-K retrieval donor VOLUME (mask-independent) so the on-the-fly
HCP dataloader can paste a merged prior into any FRESH hole -- like eda/gen_hcp_best_donor.py but
averaging the top-K MSE-reranked donors instead of taking the single best. Inline embed (no torch),
runs on the login node. Writes merged volumes to --vol-dir and an npz manifest matching the
OnTheFlyRetrHCP donor schema {names, donor_paths(->merged files), donor_scales(=1.0, already normalized)}."""
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
        img = nib.load(str(Path(G["hcp"]) / name / (name + "-t1n.nii.gz")))
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
        merged = np.mean([c[1] for c in cands[:K]], axis=0).astype(np.float32)   # normalized units
        nib.save(nib.Nifti1Image(merged, img.affine, img.header), out)
        return name
    except Exception as e:
        print("ERR", name, repr(e), flush=True); return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hcp", required=True)
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
    G["hcp"] = a.hcp; G["vol"] = a.vol_dir; G["k"] = a.k; G["shortlist"] = a.shortlist
    os.makedirs(a.vol_dir, exist_ok=True)
    ids = sorted(p.name for p in Path(a.hcp).glob("HCP-*") if p.is_dir())
    print("merged-donor for %d HCP brains k=%d -> %s" % (len(ids), a.k, a.vol_dir), flush=True)
    done = []
    with Pool(a.workers) as p:
        for i, n in enumerate(p.imap_unordered(worker, ids, chunksize=2)):
            if n: done.append(n)
            if (i + 1) % 50 == 0:
                print("  %d/%d" % (i + 1, len(ids)), flush=True)
    names = sorted(done)
    paths = [str(Path(a.vol_dir) / (n + "-merged.nii.gz")) for n in names]
    scales = np.ones(len(names), dtype=np.float32)
    np.savez(a.out, names=np.array(names), donor_paths=np.array(paths), donor_scales=scales)
    print("wrote merged-donor manifest for %d brains -> %s" % (len(names), a.out), flush=True)


if __name__ == "__main__":
    main()
