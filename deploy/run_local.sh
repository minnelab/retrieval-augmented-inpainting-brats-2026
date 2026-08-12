#!/usr/bin/env bash
# Smoke-test the image exactly as the leaderboard runs it: mount an input dir of cases
# (each with *-t1n-voided.nii.gz + *-mask.nii.gz) and an output dir, no CLI args.
#   deploy/run_local.sh <input_dir> <output_dir> [image_tag]
set -euo pipefail

IN="${1:?usage: deploy/run_local.sh <input_dir> <output_dir> [image_tag]}"
OUT="${2:?usage: deploy/run_local.sh <input_dir> <output_dir> [image_tag]}"
TAG="${3:-brats-inpainting:latest}"
mkdir -p "$OUT"

GPU=""
docker run --help 2>/dev/null | grep -q -- "--gpus" && command -v nvidia-smi >/dev/null 2>&1 && GPU="--gpus all"

docker run --rm $GPU \
  -v "$(realpath "$IN")":/input:ro \
  -v "$(realpath "$OUT")":/output \
  "$TAG"
echo "outputs in $OUT:"
ls -1 "$OUT"
