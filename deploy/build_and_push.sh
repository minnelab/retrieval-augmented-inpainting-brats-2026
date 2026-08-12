#!/usr/bin/env bash
# Build the BraTS 2026 Task-4 submission image (wdm_hcp2 + TTA, val-SSIM 0.872) and push it to the
# Synapse Docker registry so it appears in your project's submission directory.
#
#   deploy/build_and_push.sh [TAG]            # TAG defaults to v1
#
# Prereqs on the build machine (needs Docker):
#
# What it does: fetch weights -> build for linux/amd64 -> docker login -> docker push. After it
# finishes, refresh the Synapse submission page, select the image, and click "Submit Selection".
set -euo pipefail

PROJECT="${SYNAPSE_PROJECT:-syn75822651}"
TAG="${1:-v1}"
IMAGE="docker.synapse.org/${PROJECT}/brats-inpainting:${TAG}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${CKPT:-$ROOT/deploy/model.ckpt}"

# 1. Trained weights are gitignored — fetch them first with checkpoints/download.sh,
#    or point CKPT at a checkpoint you already have.
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found at $CKPT" >&2
  echo "  run ./checkpoints/download.sh, or set CKPT=/path/to/model.ckpt" >&2
  exit 1
fi
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found at $CKPT" >&2; exit 1; }
[ "$CKPT" = "$ROOT/deploy/model.ckpt" ] || cp "$CKPT" "$ROOT/deploy/model.ckpt"  # Dockerfile COPYs deploy/model.ckpt
echo ">> checkpoint ready ($(du -h "$ROOT/deploy/model.ckpt" | cut -f1))"

# 2. Build for the evaluation platform (linux/amd64; works whether your host is x86 or Apple silicon).
echo ">> building $IMAGE (linux/amd64)"
docker buildx build --platform linux/amd64 -f "$ROOT/deploy/Dockerfile" -t "$IMAGE" "$ROOT" --load

# 3. Log in to the Synapse registry (password = your Synapse PAT, not your web password) and push.
echo ">> logging in to docker.synapse.org — use a Synapse PAT (Modify) as the password"
docker login docker.synapse.org${SYNAPSE_USER:+ --username "$SYNAPSE_USER"}
echo ">> pushing $IMAGE"
docker push "$IMAGE"

echo ""
echo "DONE: pushed $IMAGE"
echo "Next: refresh the Synapse submission page -> select this image -> Submit Selection -> Task 4 queue."
