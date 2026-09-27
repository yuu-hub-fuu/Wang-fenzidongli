#!/usr/bin/env bash
# EBA — Protenix-based; env per README, CUTLASS v3.5.1, EBA checkpoint, Protenix release data (CCD + MSAs).
source "$(dirname "$0")/common.sh"
REPO="$THIRD_PARTY/eba"; ENV=eba
clone_repo https://github.com/lujiarui/eba.git "$REPO" c1f314506f313c5b4b9913fc43f1071fe70d70ec
if ! env_exists $ENV; then
  $CONDA create -y -n $ENV python=3.11
  in_env $ENV pip install -e "$REPO"
fi
[ -d "$REPO/cutlass" ] || git clone -b v3.5.1 --depth 1 https://github.com/NVIDIA/cutlass.git "$REPO/cutlass"

download https://storage.googleapis.com/project_icml25_eba/release.pt "$REPO/release.pt"

DATA="$THIRD_PARTY/protenix_data"
if [ ! -f "$DATA/seq_to_pdb_index.json" ]; then
  download https://af3-dev.tos-cn-beijing.volces.com/release_data.tar.gz "$DATA/release_data.tar.gz"
  tar -xzf "$DATA/release_data.tar.gz" -C "$DATA" && rm "$DATA/release_data.tar.gz"
fi
echo "[setup] EBA ready. Released ATLAS samples (run --fetch): https://storage.googleapis.com/project_icml25_eba/artifacts.tar.gz"
