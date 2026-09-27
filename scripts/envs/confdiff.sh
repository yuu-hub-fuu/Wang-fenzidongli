#!/usr/bin/env bash
# ConfDiff-OF-r3-MD — env.yml + OpenFold, OpenFold params (finetuning_no_templ_ptm_1.pt), ConfDiff-MD checkpoints.
source "$(dirname "$0")/common.sh"
REPO="$THIRD_PARTY/ConfDiff"; ENV=confdiff
clone_repo https://github.com/bytedance/ConfDiff.git "$REPO" 9cfae1c14121e423d8d455d03506c7e8ee580e48

if ! env_exists $ENV; then
  (cd "$REPO" && $CONDA env create -f env.yml -n $ENV)
  [ -d "$REPO/openfold" ] || git clone https://github.com/aqlaboratory/openfold.git "$REPO/openfold"
  in_env $ENV pip install -e "$REPO/openfold"
fi

# OpenFold weights used for the sequence representation (needs git-lfs)
if [ ! -s "$REPO/pretrain_repr/openfold/openfold_params/finetuning_no_templ_ptm_1.pt" ]; then
  (cd "$REPO" && bash pretrain_repr/openfold/download_openfold_param.sh pretrain_repr/openfold)
fi
# ATLAS fine-tuned checkpoints (ConfDiff-ESM-r3-MD, ConfDiff-OF-r3-MD)
hf_download leowang17/ConfDiff "$REPO/checkpoints" --include "ConfDiff-MD/*"
echo "[setup] ConfDiff ready; set options.ckpt in configs/baselines/confdiff.yaml if auto-detection of OF-r3 fails:"
find "$REPO/checkpoints" -name '*.ckpt' | sed 's/^/  /'
