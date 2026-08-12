# Building the inference containers

Two images. Both need weights, which are **not** in the repo — fetch them first:

    ./checkpoints/download.sh

Run all build commands from the repo root (the Dockerfiles expect it as the build context).

## 1. Unconditioned — `Dockerfile`

The challenge submission (`wdm_hcp2`, val-150 SSIM 0.87167 with TTA). Self-contained: no
retrieval, no donor library, so the image is just the model plus `src/diffusion/`.

    cp checkpoints/wdm_hcp2_best.ckpt deploy/model.ckpt
    docker build -f deploy/Dockerfile -t brats-inpainting:latest .

## 2. Merged retrieval prior — `Dockerfile.merged`

The best model (`merged_reg2_ep59`, 0.87861). The container gets **no network access**, so the
entire donor library — the ~1909 T1 volumes the retrieval searches over — has to be baked in.
`build_merged.sh` does that: it stages the checkpoint, copies each indexed donor's T1 into
`deploy/donors/`, rewrites the index paths to their in-container locations, then builds.

    SRC_INDEX=<retr_index_bratshcp.npz> \
    CKPT=checkpoints/wdm_merged_reg2_ep59.ckpt \
    GLI_ROOT=<brats_train_root> \
    HCP_ROOT=<hcp_brats_t1_root> \
    IMAGE=brats-inpainting-merged:latest \
    deploy/build_merged.sh

Expect a large image — the donor library dominates it.

Build the index with `retrieval/prior_retrieval.py` if you don't have one.

## Testing and export

    deploy/run_local.sh <input_dir> <output_dir> <image_tag>
    docker save <image_tag> | gzip > brats-inpainting-image.tar.gz

The container reads `/input` (per case: `<id>-t1n-voided.nii.gz` and `<id>-mask.nii.gz`) and
writes `<id>-t1n-inference.nii.gz` to `/output`. It runs with **no arguments** — that's the
leaderboard contract.

`deploy/model.ckpt`, `deploy/index.npz` and `deploy/donors/` are gitignored build inputs. A
fresh clone will not have them, and `docker build` will fail until you produce them as above.
