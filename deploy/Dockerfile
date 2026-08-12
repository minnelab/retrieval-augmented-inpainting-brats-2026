# BraTS 2026 Task 4 (inpainting) submission image — wavelet-diffusion wdm_hcp2 + TTA.
# Best on our val-150: SSIM 0.872 (beats the 2025-winner container 0.865). Unconditioned model,
# so no tissue-prior pipeline is baked in — fully self-contained (weights via deploy/build.sh).
# Base provides torch 2.7.1 + CUDA 12.6 (matches the training env). If this tag is
# unavailable in your registry, swap it for the closest pytorch/pytorch CUDA runtime tag.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INPUT_DIR=/input \
    OUTPUT_DIR=/output \
    CKPT=/app/model.ckpt

WORKDIR /app

COPY deploy/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Network + inference core (no model code is duplicated — predict.py imports from here).
COPY src/diffusion/ /app/src/
COPY deploy/predict.py /app/predict.py

# Trained weights, baked in by deploy/build.sh (see README).
COPY deploy/model.ckpt /app/model.ckpt

ENTRYPOINT ["python", "/app/predict.py"]
