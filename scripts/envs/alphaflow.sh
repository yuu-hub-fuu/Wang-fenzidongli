#!/usr/bin/env bash
# ESMFlow-MD (Full / Distilled) — AlphaFlow repository, env per its README, weights from HuggingFace.
source "$(dirname "$0")/common.sh"
REPO="$THIRD_PARTY/alphaflow"; ENV=alphaflow
clone_repo https://github.com/bjing2016/alphaflow.git "$REPO" 0408d7c89dac444a43a9089d7427ce470b0a5e67

if ! env_exists $ENV; then
  $CONDA create -y -n $ENV python=3.9
  in_env $ENV pip install numpy==1.21.2 pandas==1.5.3
  in_env $ENV pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
  in_env $ENV pip install biopython==1.79 dm-tree==0.1.6 modelcif==0.7 ml-collections==0.1.0 scipy==1.7.1 absl-py einops
  in_env $ENV pip install pytorch_lightning==2.0.4 fair-esm mdtraj==1.9.9 wandb
  # OpenFold needs CUDA 11 (see AlphaFlow README for installing it inside the env)
  in_env $ENV pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@103d037'
fi

HF=https://huggingface.co/bjing-mit/alphaflow/resolve/main
for f in esmflow_md_base_202402.pt esmflow_md_distilled_202402.pt; do
  download "$HF/params/$f" "$REPO/params/$f"
done
echo "[setup] ESMFlow-MD ready. Released ensembles (used by 'run --fetch'): $HF/samples/esmflow_md_{base,distilled}_202402.zip"
