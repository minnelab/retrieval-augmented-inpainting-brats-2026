#!/usr/bin/env bash
# Assemble + build the MERGED-PRIOR submission image. Because the container has ZERO network access,
# the donor library (the ~1909 t1n volumes the retrieval indexes) must be baked in. This script:
#   1. stages the model checkpoint,
#   2. copies each indexed donor's t1n into deploy/donors/<name>/<name>-t1n.nii.gz,
#   3. rewrites the index paths to the in-container location /app/donors/...,
#   4. docker build.
#
# Env: SRC_INDEX (source retr_index_bratshcp.npz), CKPT (merged model .ckpt),
#      GLI_ROOT + HCP_ROOT (where donor t1n volumes live), IMAGE (tag). Run from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
IMAGE="${IMAGE:-brats-task4-merged:latest}"
SRC_INDEX="${SRC_INDEX:?set SRC_INDEX=/path/retr_index_bratshcp.npz}"
CKPT="${CKPT:?set CKPT=/path/merged_ep21.ckpt}"
GLI_ROOT="${GLI_ROOT:?set GLI_ROOT}"; HCP_ROOT="${HCP_ROOT:?set HCP_ROOT}"

[ "$CKPT" -ef deploy/model.ckpt ] || cp "$CKPT" deploy/model.ckpt
rm -rf deploy/donors; mkdir -p deploy/donors

# Stage donors + rewrite index paths to /app/donors/<name>/<name>-t1n.nii.gz
python3 - "$SRC_INDEX" "$GLI_ROOT" "$HCP_ROOT" <<'PY'
import sys, shutil, numpy as np
from pathlib import Path
src, gli, hcp = sys.argv[1], sys.argv[2], sys.argv[3]
z = np.load(src, allow_pickle=True); paths = [str(p) for p in z["paths"]]
donors = Path("deploy/donors"); new = []
for p in paths:
    name = Path(p).parent.name
    root = gli if name.startswith("BraTS") else hcp
    srcv = Path(root) / name / f"{name}-t1n.nii.gz"
    dst = donors / name / f"{name}-t1n.nii.gz"; dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists(): shutil.copy(srcv, dst)
    new.append(f"/app/donors/{name}/{name}-t1n.nii.gz")
np.savez("deploy/index.npz", paths=np.array(new), emb=z["emb"])
print(f"staged {len(new)} donors -> deploy/donors ; index -> deploy/index.npz")
PY

echo ">>> donor library size:"; du -sh deploy/donors
docker build -f deploy/Dockerfile.merged -t "$IMAGE" .
echo ">>> built $IMAGE"
echo ">>> test:  docker run --rm --network none --gpus all \\"
echo "             -v \$PWD/IN:/input:ro -v \$PWD/OUT:/output:rw \\"
echo "             --memory=48G --shm-size=16G $IMAGE"
