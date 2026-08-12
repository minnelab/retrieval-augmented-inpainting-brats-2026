# Training and inference recipes

The exact configurations behind the released checkpoints. Paths are placeholders — set
them to your own data roots.

Common inputs:

- `<GLI>` — BraTS 2026 Task-4 training data
- `<GLI_X5>` — the same cases with 5 mask-augmented samples per brain (`data/generate_masks.py`)
- `<GOAT_X40>` — 100 additional non-glioma brains, 40 distinct decoy masks each
- `<HCP>` — 808 healthy HCP T1s on the BraTS grid (`data/preprocess_hcp.py`)
- `<PRIORS>/{train,val}` — merged retrieval priors (`retrieval/gen_merged_priors.py`)
- `<DONOR>.npz` — per-brain merged donor volumes (`retrieval/gen_hcp_merged_donor.py`)

All runs use `--crop 128 128 128 --batch 4`, cosine LR decay and EMA 0.999. Multi-GPU via
`--devices N`.

---

## Diffusion, merged retrieval prior — the submitted model

`wdm_merged_reg2_ep59.ckpt`. A 60-epoch continuation at low LR with weight
decay, warm-started from an earlier merged-prior checkpoint. `--hcp-sigma 0` keeps the HCP prior
unblurred so it matches the frozen merged priors — blurring destroys the mid-frequency detail that
makes the merged prior better than a single donor.

```bash
python src/diffusion/train.py --model diffusion \
    --init-ckpt <earlier merged-prior ckpt> \
    --data-root <GLI> --train-root <GLI_X5> <GOAT_X40> \
    --train-ids splits/train_ids.txt+splits/goat_new_ids.txt \
    --val-ids splits/val_small_ids.txt --out-dir runs/merged_reg2 \
    --prior-root <PRIORS>/train --val-prior-root <PRIORS>/val \
    --hcp-root <HCP> --hcp-oversample 5 --hcp-retr-donor <DONOR>.npz --hcp-sigma 0 \
    --epochs 60 --lr 1e-5 --lr-schedule cosine --weight-decay 1e-3 \
    --timesteps 1000 --infer-steps 4 --n-avg 1 \
    --w-wave 1.0 --w-recon 1.0 --w-ssim 0.0 --ema 0.999 --save-every 1
```

The training id list is `splits/train_ids.txt` concatenated with `splits/goat_new_ids.txt`.

## Diffusion, no prior — the baseline

`wdm_hcp3_cont_ep039.ckpt`. Same data scale, no prior channel.

```bash
python src/diffusion/train.py --model diffusion \
    --init-ckpt <earlier no-prior ckpt> \
    --data-root <GLI> --train-root <GLI_X5> <GOAT_X40> \
    --train-ids splits/train_ids.txt+splits/goat_new_ids.txt \
    --val-ids splits/val_small_ids.txt --out-dir runs/noprior \
    --hcp-root <HCP> --hcp-oversample 5 --decoy-large-frac 0.3 \
    --epochs 40 --lr 5e-5 --lr-schedule cosine \
    --timesteps 1000 --infer-steps 4 --n-avg 1 \
    --w-wave 1.0 --w-recon 1.0 --w-ssim 0.0 --ema 0.999 --save-every 1
```

## Diffusion, earlier unconditioned model

`wdm_hcp2_best.ckpt`. Smaller pool (`--hcp-oversample 3`, 20 decoys per
GoAT brain rather than 40), `--epochs 60 --lr 1e-4`, otherwise as above.

## SegResNet

`unet_segresnet_cont3_best.ckpt`. `--data-style winner` is mandatory at both training and inference; the flag defaults to
`ours` and the mismatch produces broken output rather than an error.

```bash
python src/segresnet/train.py --arch segresnet \
    --data-root <GLI> --train-root <GLI_X5> \
    --data-style winner --aug-flips --match-infer-void \
    --hcp-root <HCP> --hcp-pool-cache data/hcp_shape_pool.pkl \
    --train-ids splits/train_plus_goat_ids.txt --val-ids splits/val_small_ids.txt \
    --out-dir runs/segresnet --init-ckpt <earlier segresnet ckpt> \
    --w-l1 1.0 --w-mse 0.0 --w-ssim 1.0 --ssim-ds 2 \
    --lr 5e-5 --lr-schedule cosine --ema 0.999 --ckpt-monitor val_ssim --epochs 120
```

### SegResNet + retrieval prior

`unet_segresnet_merged_best.ckpt` and its matched control
`unet_segresnet_noprior_ctrl.ckpt`. Identical except for the prior channel, which is
what makes the pair a clean measurement of the prior's contribution. Add to the above:

```bash
    --merged-donor-npz <DONOR>.npz --val-prior-root <PRIORS>/val \
    --train-ids splits/train_ids.txt --lr 1e-4 --epochs 100
```

`train.py` expands the input convolution to 3 channels with a zero-initialised third channel, so
the prior starts as a no-op and the warm-start stays valid.

---

## Inference

Running the submitted model over a split:

```bash
python src/diffusion/infer.py --model diffusion \
    --ckpt checkpoints/wdm_merged_reg2_ep59.ckpt \
    --data-root <GLI> --ids splits/val_ids.txt --prior-root <PRIORS>/val \
    --out-dir preds_reg2_tta --roi 128 128 128 --steps 4 --n-avg 8 --tta
```

Drop `--prior-root` for the no-prior baseline. SegResNet uses
`src/segresnet/infer.py` with `--data-style winner`.

For the 750-decoy set, generate priors with
`retrieval/gen_merged_priors.py --ids splits/val750_ids.txt`, then run inference with
`--n-avg 8` and no TTA.
