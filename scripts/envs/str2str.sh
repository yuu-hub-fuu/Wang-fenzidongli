#!/usr/bin/env bash
# Str2Str — environment.yml (+ pip install -e .), pretrained checkpoint from Google Drive.
source "$(dirname "$0")/common.sh"
REPO="$THIRD_PARTY/Str2Str"; ENV=str2str
clone_repo https://github.com/lujiarui/Str2Str.git "$REPO" 0b690e990e2a95c73766db9e053a0e0ae0b8b181
if ! env_exists $ENV; then
  (cd "$REPO" && $CONDA env create -f environment.yml -n $ENV)
  in_env $ENV pip install -e "$REPO"
fi
CKPT="$REPO/data/ckpt/pretrain.pth"
if [ ! -s "$CKPT" ]; then
  mkdir -p "$(dirname "$CKPT")"
  # https://drive.google.com/file/d/1YsvFXOpdst4QxK34GSWvLjgbvzUq4Ry8 (Str2Str README)
  in_env $ENV pip install gdown
  in_env $ENV gdown 1YsvFXOpdst4QxK34GSWvLjgbvzUq4Ry8 -O "$CKPT" || {
    echo "[setup] download the checkpoint manually to $CKPT"; exit 1; }
fi
echo "[setup] Str2Str ready."
