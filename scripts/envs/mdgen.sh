#!/usr/bin/env bash
# MDGen — env per README, model weights from https://huggingface.co/bjing-mit/mdgen.
source "$(dirname "$0")/common.sh"
REPO="$THIRD_PARTY/mdgen"; ENV=mdgen
clone_repo https://github.com/bjing2016/mdgen.git "$REPO" 81482a403b91c5a8437046da1d8d321ba97089cc
if ! env_exists $ENV; then
  $CONDA create -y -n $ENV python=3.9
  in_env $ENV pip install numpy==1.21.2 pandas==1.5.3
  in_env $ENV pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
  in_env $ENV pip install pytorch_lightning==2.0.4 mdtraj==1.9.9 biopython==1.79
  in_env $ENV pip install wandb dm-tree einops torchdiffeq fair-esm pyEMMA
  in_env $ENV pip install matplotlib==3.7.2 numpy==1.21.2 huggingface_hub
fi
hf_download bjing-mit/mdgen "$REPO/weights"
ls "$REPO/weights"
echo "[setup] MDGen ready; the ATLAS model is expected at weights/atlas.ckpt (edit configs/baselines/mdgen.yaml otherwise)."
