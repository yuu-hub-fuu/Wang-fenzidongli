#!/usr/bin/env bash
# BioEmu — pip package pinned to the reviewed commit; weights download automatically from HuggingFace.
source "$(dirname "$0")/common.sh"
REPO="$THIRD_PARTY/bioemu"; ENV=bioemu
COMMIT=8babb71563e97dc1feb9e3e2fbc00a1224b6a3a7
clone_repo https://github.com/microsoft/bioemu.git "$REPO" $COMMIT
if ! env_exists $ENV; then
  $CONDA create -y -n $ENV python=3.10
  in_env $ENV pip install "$REPO[cuda]"
  # optional side-chain reconstruction (HPacker, needs conda on PATH): pip install "$REPO[md]"
fi
echo "[setup] BioEmu ready (model weights + ColabFold AF2 params are fetched on first use)."
