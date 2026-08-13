# Retrieval-Augmented Wavelet Diffusion for Local Synthesis of Healthy Brain Tissue

Code accompanying our BraTS 2026 *Local Synthesis of Healthy Brain Tissue via Inpainting* entry.

Given a brain MRI with a masked-out region, the task is to synthesise plausible healthy tissue
inside the mask. We condition a 3D wavelet diffusion model on a prior assembled from anatomically
similar donor brains.

## Install

```bash
# install PyTorch first, matched to your CUDA — https://pytorch.org
pip install -r requirements.txt
```

`monai==1.5.2` is a hard pin. MONAI changed `HaarDWT3D` in 1.6.0; the same checkpoint then
reconstructs visibly worse, with no error raised.

## Weights

```bash
./checkpoints/download.sh
```

`wdm_merged_reg2_ep59.ckpt` is the submitted model.

## Inference

The container entrypoint runs retrieval and diffusion in one pass; the donor library is baked into
the image, so it needs no network access. It takes no arguments and is configured by environment
variables — `INPUT_DIR`, `OUTPUT_DIR`, `CKPT`, `INDEX`, plus `K`, `SHORTLIST` and the usual
`ROI`/`STEPS`/`N_AVG_*`. See [deploy/BUILD.md](deploy/BUILD.md).

```bash
deploy/run_local.sh <input_dir> <output_dir> brats-inpainting-merged:latest
```

To run the stages separately:

```bash
python retrieval/gen_merged_priors.py \
    --ids splits/val_ids.txt --data-root <brats_root> \
    --index <donor_index.npz> --out-dir priors/val --k 10 --shortlist 15

python src/diffusion/infer.py --model diffusion \
    --ckpt checkpoints/wdm_merged_reg2_ep59.ckpt \
    --data-root <brats_root> --ids splits/val_ids.txt \
    --prior-root priors/val --out-dir preds/ \
    --roi 128 128 128 --steps 4 --n-avg 8 --tta
```

Drop `--prior-root` for an unconditioned checkpoint. `--n-avg` trades compute for quality; `--tta`
adds 8-flip test-time augmentation. `FLASH=0` disables flash attention.

## Training

Prepare the inputs:

```bash
python data/preprocess_hcp.py         # HCP T1s onto the BraTS grid (donor pool)
python data/generate_masks.py         # mask-augmented training copies
python retrieval/prior_retrieval.py   # donor index
python retrieval/gen_merged_priors.py # merged priors
```

Then:

```bash
python src/diffusion/train.py --model diffusion \
    --data-root <brats_root> --train-ids splits/train_ids.txt \
    --val-ids splits/val_small_ids.txt --out-dir runs/merged \
    --prior-root <priors>/train --val-prior-root <priors>/val \
    --crop 128 128 128 --batch 4 --epochs 60 --lr 1e-5 --lr-schedule cosine \
    --timesteps 1000 --infer-steps 4 --ema 0.999
```

[docs/TRAINING.md](docs/TRAINING.md) gives the configuration behind each released checkpoint.

## Layout

```
src/diffusion/   wavelet DDPM — wavelet.py, litdiff.py, data.py, train.py, infer.py
retrieval/       donor index, merged prior, per-brain donor precompute
data/            mask generation (vendored official generator), HCP preprocessing
splits/          id lists
deploy/          inference containers
```

## Attribution

`data/maskgen/` is the BraTS 2026 challenge organisers' healthy-mask generator, vendored unchanged
and retaining its original licence.
