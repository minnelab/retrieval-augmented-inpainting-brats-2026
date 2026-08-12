"""Generate MERGED top-K retrieval priors: fill each void with the AVERAGE of the K closest donors
(MSE-reranked) tissue. Same file schema as retr_priors ({name}{sfx}-t1n-inference.nii.gz). Inline
embed (no torch), parallel over cases."""
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
        root = Path(G["data"]); sfx = G["sfx"]; K = G["k"]; sl = G["shortlist"]
        out = Path(G["out"]) / (name + sfx + "-t1n-inference.nii.gz")
        if out.exists():
            return 0
        d = root / name
        vimg = nib.load(str(d / (name + "-t1n-voided" + sfx + ".nii.gz")))
        voided = np.asarray(vimg.get_fdata(), dtype=np.float32)
        void = load(d / (name + "-mask" + sfx + ".nii.gz")) > 0.5
        scale = pct995(voided); ctx = voided > 0
        q = embed(voided); q = q / (np.linalg.norm(q) + 1e-6)
        order = np.argsort(-(G["embn"] @ q))
        tgt = (voided / scale)[ctx].astype(np.float32)
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
        result = voided.copy()
        result[void] = (merged[void] * scale).astype(np.float32)
        nib.save(nib.Nifti1Image(result, vimg.affine, vimg.header), out)
        return 1
    except Exception as e:
        print("ERR", name, repr(e), flush=True); return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--index", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--shortlist", type=int, default=15)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    idx = np.load(a.index, allow_pickle=True)
    G["paths"] = idx["paths"]; emb = idx["emb"]
    G["embn"] = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6)
    G["names"] = [Path(p).parent.name for p in idx["paths"]]
    G["data"] = a.data_root; G["sfx"] = a.suffix; G["k"] = a.k
    G["shortlist"] = a.shortlist; G["out"] = a.out_dir
    os.makedirs(a.out_dir, exist_ok=True)
    ids = [x.strip() for x in open(a.ids) if x.strip()]
    print("merging %d cases sfx=[%s] k=%d -> %s" % (len(ids), a.suffix, a.k, a.out_dir), flush=True)
    with Pool(a.workers) as p:
        tot = 0
        for i, n in enumerate(p.imap_unordered(worker, ids, chunksize=4)):
            tot += n
            if (i + 1) % 150 == 0:
                print("  %d/%d (%d written)" % (i + 1, len(ids), tot), flush=True)
    print("done: %d priors -> %s" % (tot, a.out_dir), flush=True)


if __name__ == "__main__":
    main()
