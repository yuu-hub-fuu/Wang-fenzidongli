#!/usr/bin/env bash
# BioMD proxy — BioKinema (same authors; BioMD itself is not publicly released).
source "$(dirname "$0")/common.sh"
REPO="$THIRD_PARTY/BioKinema"; ENV=biokinema
clone_repo https://github.com/IDEA-XL/BioKinema.git "$REPO" e03508e341b71d1a5f0fa533e46665943fd21791
if ! env_exists $ENV; then
  (cd "$REPO" && $CONDA env create -f environment.yml -n $ENV)
fi
hf_download fengb/BioKinema "$REPO/checkpoints" --include "BioKinema_atlas+misato+mdposit_sqrt.pt"
echo "[setup] BioKinema ready. Its custom kernels need CUTLASS_PATH / CUDA_HOME (options.cutlass_path / options.cuda_home)."
