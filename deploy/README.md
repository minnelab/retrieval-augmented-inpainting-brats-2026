# Deployment — BraTS 2026 Task 4 submission container

Packages a trained checkpoint into the Docker image the challenge expects. **Deploy glue
only** — the network and the inference core live in `src/`; nothing here is model code.

## Container contract

The leaderboard runner mounts an input dir of cases and an output dir, then runs the image
with **no arguments**. For each case it finds

```
<case>-t1n-voided.nii.gz   +   <case>-mask.nii.gz
```

`predict.py` writes the composited inpainting (voided input outside the mask, model
prediction inside; original affine/header preserved) as

```
<case>-t1n-inference.nii.gz
```

Cases are discovered with `rglob("*-t1n-voided.nii.gz")`, so flat dirs and per-case
subfolders both work. GPU is used when available, otherwise it falls back to CPU.

Paths default to `/input`, `/output`, `/app/model.ckpt` and are overridable via the
`INPUT_DIR` / `OUTPUT_DIR` / `CKPT` env vars (and `ROI`, default `128,128,128`), or as
positional args `python predict.py [INPUT_DIR] [OUTPUT_DIR] [CKPT]`.

> Confirm the mount points the current Synapse Task 4 runner uses. `/input` + `/output`
> is the recent BraTS convention (used here as the default); some older years used
> `/app/data` + `/data/results`. If Task 4 differs, set `INPUT_DIR`/`OUTPUT_DIR` in the
> `Dockerfile` to match — no code change needed.

## Build

```bash
deploy/build.sh runs/<exp>/checkpoints/best.ckpt  brats-inpainting:latest
```

This copies the checkpoint to `deploy/model.ckpt` (gitignored) and bakes it into the image.

## Test locally (mimics the leaderboard invocation)

```bash
deploy/run_local.sh <input_dir> <output_dir> brats-inpainting:latest
```

where `<input_dir>` holds a few cases' `*-t1n-voided.nii.gz` + `*-mask.nii.gz`. Then score
the outputs against a local split to confirm parity before submitting.

## Apptainer / Singularity

Many HPC sites run Apptainer, not Docker. Build the Docker image somewhere with Docker (or pull
it), then convert and run:

```bash
apptainer build brats-inpainting.sif docker-daemon://brats-inpainting:latest
apptainer run --nv \
  -B <input_dir>:/input -B <output_dir>:/output \
  brats-inpainting.sif
```

## Submit

Push the image to the registry the challenge specifies and submit per the Synapse Task 4
instructions. The image is fully self-contained (weights baked in); the runner provides
only `t1n-voided` + `mask` per case.

## Files

- `Dockerfile` — torch/CUDA base + `src/` + `predict.py` + baked `model.ckpt`.
- `predict.py` — entrypoint: discover cases → infer → write `-t1n-inference.nii.gz`.
- `requirements.txt` — monai / lightning / nibabel on top of the base image's torch.
- `build.sh` / `run_local.sh` — build with a checkpoint / smoke-test like the leaderboard.
