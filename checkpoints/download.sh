#!/usr/bin/env bash
# Fetch the released checkpoints into this directory and verify their checksums.
#
#   ./download.sh              # everything
#   ./download.sh wdm_merged_reg2_ep59.ckpt
#
# Diffusion (3D wavelet DDPM):
#   wdm_merged_reg2_ep59.ckpt       + retrieval prior; the submitted model
#   wdm_merged_ep021.ckpt           + retrieval prior; earlier epoch of the same run
#   wdm_hcp3_cont_ep039.ckpt        unconditioned; matched control for ep021
#   wdm_hcp2_best.ckpt              unconditioned; earlier, smaller training pool
#
# SegResNet (supervised baseline; all need --data-style winner at inference):
#   unet_segresnet_merged_best.ckpt  + retrieval prior as a 3rd channel (needs --prior-root)
#   unet_segresnet_noprior_ctrl.ckpt matched no-prior control for the above
#   unet_segresnet_cont_best.ckpt    the baseline
#   unet_segresnet_cont3_best.ckpt   a further continuation of it
set -euo pipefail

REPO="minnelab/retrieval-augmented-inpainting-brats-2026"
TAG="${TAG:-v1.0}"
BASE="https://github.com/${REPO}/releases/download/${TAG}"
cd "$(dirname "$0")"

declare -A SHA=(
  [wdm_merged_reg2_ep59.ckpt]=25bee77c254c2eeb53f7b69b22ce88058e0ac8256a5c591b22f2628571ad1805
  [wdm_hcp3_cont_ep039.ckpt]=228cb91ce0445ee5f4dda8857745b086fba4f431f3381973179b80e0ba14e161
  [unet_segresnet_merged_best.ckpt]=aed3f43a5dba29b1d493721e598975bddea39387eb99a0418416c20cdd65eeb9
  [unet_segresnet_noprior_ctrl.ckpt]=dd9cea90c6c596c68ca08b5f78affa328a39981a6841b16c9f1ad77ad20d0d99
  [unet_segresnet_cont_best.ckpt]=79f6ddf6a7ac1f903de90ccbf72afb96b26ea8bedd46f6d7b90f08c2c8e092cc
  [unet_segresnet_cont3_best.ckpt]=6b422104661daace0a7abdac625719a1c2792634093d7cd73bf86c1a3f7d3b9d
  [wdm_hcp2_best.ckpt]=1421356c0b6118a395bf6a4f0d92a4df83526db8d5876f5cbe0001d861750f85
  [wdm_merged_ep021.ckpt]=033019f07bb24b4b210c034c7214dd0883e0ee5e91ee61d13dda068f20f4bae9
)

files=("$@")
[ ${#files[@]} -eq 0 ] && files=("${!SHA[@]}")

for f in "${files[@]}"; do
  want="${SHA[$f]:-}"
  if [ -z "$want" ]; then
    echo "unknown checkpoint: $f" >&2
    echo "known: ${!SHA[*]}" >&2
    exit 1
  fi

  if [ -f "$f" ] && [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$want" ]; then
    echo "ok (cached)  $f"
    continue
  fi

  echo "downloading  $f"
  curl -fL --retry 3 -o "$f.part" "$BASE/$f"

  got="$(sha256sum "$f.part" | cut -d' ' -f1)"
  if [ "$got" != "$want" ]; then
    echo "CHECKSUM MISMATCH for $f" >&2
    echo "  expected $want" >&2
    echo "  got      $got" >&2
    echo "  left the partial download at $f.part" >&2
    exit 1
  fi
  mv "$f.part" "$f"
  echo "ok           $f"
done

echo
echo "Reminder: loading these requires monai==1.5.2 (see ../requirements.txt)."
