# Retrieval-Augmented Wavelet Diffusion for Local Synthesis of Healthy Brain Tissue

Code accompanying our BraTS 2026 *Local Synthesis of Healthy Brain Tissue via Inpainting* entry.

Given a brain MRI with a masked-out region, the task is to synthesise plausible **healthy** tissue
inside the mask. We condition a 3D wavelet diffusion model on a **retrieval prior**: a fill
assembled from anatomically similar donor brains, found by nearest-neighbour search over a
low-resolution structural embedding. A supervised SegResNet baseline is included, with and without
the same prior.

## How the retrieval prior works

Three stages, implemented in [`retrieval/prior_retrieval.py`](retrieval/prior_retrieval.py):

1. **Index** — every donor brain is streamed once and reduced to a 24×24×16 z-scored structural
   embedding. Only the embedding and the path are kept, so a pool of ~1900 brains (BraTS train
   minus val, plus 808 healthy HCP subjects) indexes within 64 GB of RAM.
2. **Query** — the target's *visible* tissue is embedded the same way; donors are shortlisted by
   cosine similarity (default top-15).
3. **Rerank** — only the shortlist is loaded at full resolution and reranked by MSE over the
   visible region. The void is filled with the average of the top-K reranked donors (the "merged"
   prior). Because the full-resolution comparison touches only the shortlist, a query costs about
   two seconds per case.

The merged prior is mask-independent per donor brain, so a precomputed donor volume can fill any
fresh mask — which is what makes on-the-fly mask sampling during training possible.

## Layout

```
src/diffusion/     3D wavelet DDPM — wavelet.py (Haar DWT), litdiff.py, data.py,
                   train.py, infer.py
src/segresnet/     supervised MONAI SegResNet baseline
retrieval/         donor index, merged top-K prior, per-brain donor precompute
data/              official challenge mask generator (vendored), decoy samplers,
                   HCP preprocessing, prior generation
splits/            train / val id lists
deploy/            inference containers
checkpoints/       download script (weights live on the release)
docs/TRAINING.md   the configuration behind each released checkpoint
```

`src/diffusion/` and `src/segresnet/` each contain their own `data.py`, `train.py` and `infer.py`.
They are different code, not duplicates — the diffusion `data.py` has six dataset classes to the
SegResNet one's three. Both use flat sibling imports, so run them by path
(`python src/diffusion/train.py`) and the right modules resolve.

## Install

```bash
# install PyTorch first, matched to your CUDA — https://pytorch.org
pip install -r requirements.txt
```

> **`monai==1.5.2` is a hard pin.** MONAI changed `HaarDWT3D` in 1.6.0; the same diffusion
> checkpoint then reconstructs visibly worse, with no error raised.

## Get the weights

```bash
./checkpoints/download.sh
```

| checkpoint | model | prior |
|---|---|---|
| `wdm_merged_reg2_ep59.ckpt` | diffusion | ✅ — the submitted model |
| `wdm_merged_ep021.ckpt` | diffusion | ✅ — earlier epoch of the same run |
| `wdm_hcp3_cont_ep039.ckpt` | diffusion | — matched control for `ep021` |
| `wdm_hcp2_best.ckpt` | diffusion | — earlier, smaller training pool |
| `unet_segresnet_merged_best.ckpt` | SegResNet | ✅ — 3rd input channel, needs `--prior-root` |
| `unet_segresnet_noprior_ctrl.ckpt` | SegResNet | — matched control for the above |
| `unet_segresnet_cont_best.ckpt` | SegResNet | — the baseline |
| `unet_segresnet_cont3_best.ckpt` | SegResNet | — a further continuation of it |

## Inference

The container entrypoint runs retrieval and diffusion in one pass, with no precomputed priors: the
donor library is baked into the image, so it needs no network access. It takes **no arguments**
(the challenge contract) and is configured by environment variables — `INPUT_DIR` (`/input`),
`OUTPUT_DIR` (`/output`), `CKPT` (`/app/model.ckpt`), `INDEX` (`/app/index.npz`), plus `K` (10),
`SHORTLIST` (15) and the usual `ROI`/`STEPS`/`N_AVG_*` knobs:

```bash
deploy/run_local.sh <input_dir> <output_dir> brats-inpainting-merged:latest
```

Building it is described in [deploy/BUILD.md](deploy/BUILD.md).

To run the stages separately, without a baked donor library:

```bash
# 1. build the merged retrieval prior for a set of cases
python retrieval/gen_merged_priors.py \
    --ids splits/val_ids.txt --data-root <brats_root> \
    --index <donor_index.npz> --out-dir priors/val --k 10 --shortlist 15

# 2. prior-conditioned diffusion
python src/diffusion/infer.py --model diffusion \
    --ckpt checkpoints/wdm_merged_reg2_ep59.ckpt \
    --data-root <brats_root> --ids splits/val_ids.txt \
    --prior-root priors/val --out-dir preds/ \
    --roi 128 128 128 --steps 4 --n-avg 8 --tta
```

Drop `--prior-root` for an unconditioned checkpoint. `--n-avg` trades compute for quality and
`--tta` adds 8-flip test-time augmentation.

SegResNet uses its own entrypoint, and **`--data-style winner` is mandatory**:

```bash
python src/segresnet/infer.py --data-style winner \
    --ckpt checkpoints/unet_segresnet_cont_best.ckpt \
    --data-root <brats_root> --ids splits/val_ids.txt \
    --out-dir preds/ --roi 128 128 128 [--tta]
```

> All SegResNet checkpoints were trained with the "winner" [-1,1] normalisation, but the flag
> **defaults to `ours`** ([0,1]). Omitting it raises no error and produces broken output. Add
> `--prior-root` for `unet_segresnet_merged_best.ckpt`, which expects the retrieval prior as a
> third input channel.

Set `FLASH=0` to disable flash attention if your GPU or PyTorch build doesn't support it.

## Training

Prepare the inputs:

```bash
python data/preprocess_hcp.py     ...   # HCP T1s onto the BraTS grid (the donor pool)
python data/generate_masks.py     ...   # mask-augmented training copies
python retrieval/prior_retrieval.py ... # donor index
python retrieval/gen_merged_priors.py ... # merged priors for training and validation
```

Then train:

```bash
python src/diffusion/train.py --model diffusion \
    --data-root <brats_root> --train-ids splits/train_ids.txt \
    --val-ids splits/val_small_ids.txt --out-dir runs/merged \
    --prior-root <priors>/train --val-prior-root <priors>/val \
    --crop 128 128 128 --batch 4 --epochs 60 --lr 1e-5 --lr-schedule cosine \
    --timesteps 1000 --infer-steps 4 --ema 0.999
```

[docs/TRAINING.md](docs/TRAINING.md) gives the full configuration behind each released
checkpoint — data mix, learning rate, schedule, warm-start chain and loss weights.

## Licence and attribution

`data/maskgen/` is the BraTS 2026 challenge organisers' healthy-mask generator, vendored unchanged
and retaining its original licence.
