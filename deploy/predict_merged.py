"""BraTS 2026 Task 4 submission entrypoint — MERGED-PRIOR wavelet-diffusion model.

Two-stage per case: (1) build the MERGED top-K retrieval prior — embed the voided brain, cosine-
shortlist a donor index, MSE-rerank, average the top-K donors' tissue into the hole; (2) run the
prior-conditioned diffusion model (concat prior channel) with n_avg + TTA, composited.

Needs, baked into the image: model.ckpt, the donor INDEX (embeddings + donor volume paths under
/app/donors), and the donor volume library itself. Donor I/O is threaded (not multiprocessing) so the
container needs NO large /dev/shm — the loads share the process's heap, and torch's few-step inference
does its batching on the GPU, not via DataLoader workers. (See DEPLOY notes for the measured shm need.)

Env knobs: INPUT_DIR /input, OUTPUT_DIR /output, CKPT /app/model.ckpt, INDEX /app/index.npz,
K (10), SHORTLIST (15), LOAD_THREADS (8); plus the usual ROI/STEPS/ADAPTIVE/N_AVG_MIN/MAX/BATCH_CAP.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data import load                        # noqa: E402
from infer import diffuse_volume             # noqa: E402
from litdiff import LitDiffusion             # noqa: E402

VOIDED_SUFFIX = "-t1n-voided.nii.gz"
MASK_SUFFIX = "-mask.nii.gz"
OUT_SUFFIX = "-t1n-inference.nii.gz"
EMB = (24, 24, 16)                           # structural embedding grid (matches gen_merged_priors)


def pct995(v):
    s = float(np.percentile(v, 99.5))
    return s if s > 1e-6 else 1.0


def embed(vol):
    s = pct995(vol); v = vol / s
    f = [EMB[i] / vol.shape[i] for i in range(3)]
    d = ndimage.zoom(v, f, order=1).ravel().astype(np.float32)
    return (d - d.mean()) / (d.std() + 1e-6)


def build_merged_prior(voided, void, embn, names, paths, self_name, k, shortlist, pool):
    """Merged top-K retrieval fill for one case (same math as retrieval/gen_merged_priors.worker)."""
    scale = pct995(voided); ctx = voided > 0
    q = embed(voided); q = q / (np.linalg.norm(q) + 1e-6)
    order = np.argsort(-(embn @ q))
    tgt = (voided / scale)[ctx].astype(np.float32)
    shortlist_idx = [j for j in order if names[j] != self_name][:shortlist]
    donors = list(pool.map(lambda j: load(paths[j]) / pct995(load(paths[j])), shortlist_idx))  # threaded I/O
    mses = [float(np.mean((tgt - dn[ctx].astype(np.float32)) ** 2)) for dn in donors]
    topk = [donors[i] for i in np.argsort(mses)[:k]]
    merged = np.mean(topk, axis=0).astype(np.float32)
    result = voided.copy()
    result[void] = (merged[void] * scale).astype(np.float32)
    return result


def main():
    in_dir = Path(os.environ.get("INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("OUTPUT_DIR", "/output"))
    ckpt = os.environ.get("CKPT", "/app/model.ckpt")
    index_path = os.environ.get("INDEX", "/app/index.npz")
    k = int(os.environ.get("K", "10")); shortlist = int(os.environ.get("SHORTLIST", "15"))
    load_threads = int(os.environ.get("LOAD_THREADS", "8"))
    roi = int(os.environ.get("ROI", "128,128,128").split(",")[0]); steps = int(os.environ.get("STEPS", "4"))
    n_avg = int(os.environ.get("N_AVG", "4")); tta = os.environ.get("TTA", "1") not in ("0", "false", "False", "")
    adaptive = os.environ.get("ADAPTIVE", "1") not in ("0", "false", "False", "")
    n_avg_min = int(os.environ.get("N_AVG_MIN", "8")); n_avg_max = int(os.environ.get("N_AVG_MAX", "16"))
    batch_cap = int(os.environ.get("BATCH_CAP", "32"))

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw = {"use_flash_attention": True} if device.type == "cuda" else {}
    model = LitDiffusion.load_from_checkpoint(ckpt, map_location=device, **kw).to(device).eval()

    idx = np.load(index_path, allow_pickle=True)
    paths = idx["paths"]; emb = idx["emb"]
    embn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6)   # ~74 MB in heap
    names = [Path(p).parent.name for p in paths]
    print(f"device={device} donors={len(paths)} k={k} shortlist={shortlist} threads={load_threads}", flush=True)

    voided_files = sorted(in_dir.rglob(f"*{VOIDED_SUFFIX}"))
    if not voided_files:
        sys.exit(f"no *{VOIDED_SUFFIX} under {in_dir}")
    print(f"found {len(voided_files)} case(s)", flush=True)

    with ThreadPoolExecutor(max_workers=load_threads) as pool:
        for i, vp in enumerate(voided_files):
            name = vp.name[: -len(VOIDED_SUFFIX)]
            mp = vp.with_name(f"{name}{MASK_SUFFIX}")
            if not mp.exists():
                print(f"WARN skip {name}: missing {mp.name}", flush=True); continue
            vimg = nib.load(str(vp)); voided = np.asarray(vimg.get_fdata(), dtype=np.float32)
            void = load(mp) > 0.5
            prior = build_merged_prior(voided, void, embn, names, paths, name, k, shortlist, pool)
            result = diffuse_volume(model, voided, void, prior=[prior], roi=roi, steps=steps,
                                    n_avg=n_avg, tta=tta, device=device, batch_cap=batch_cap,
                                    adaptive=adaptive, n_avg_min=n_avg_min, n_avg_max=n_avg_max)
            nib.save(nib.Nifti1Image(result, vimg.affine, vimg.header), out_dir / f"{name}{OUT_SUFFIX}")
            print(f"[{i + 1}/{len(voided_files)}] wrote {name}{OUT_SUFFIX}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
