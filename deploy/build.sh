#!/usr/bin/env bash
# Build the submission image with a trained checkpoint baked in.
#   deploy/build.sh <path-to-checkpoint.ckpt> [image_tag]
set -euo pipefail

CKPT="${1:?usage: deploy/build.sh <path-to-checkpoint.ckpt> [image_tag]}"
TAG="${2:-brats-inpainting:latest}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }
cp "$CKPT" "$ROOT/deploy/model.ckpt"

# Build context = repo root (Dockerfile COPYs src/ and deploy/).
docker build -f "$ROOT/deploy/Dockerfile" -t "$TAG" "$ROOT"
echo "built $TAG (checkpoint: $CKPT)"
