# Shared helpers for the per-baseline environment scripts.
# Usage: THIRD_PARTY=/path/to/third_party bash scripts/envs/<baseline>.sh
# Requires: git, conda (or mamba), curl; git-lfs / huggingface-cli / gdown where noted.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY="${THIRD_PARTY:-$ROOT/third_party}"
CONDA="${CONDA:-conda}"
mkdir -p "$THIRD_PARTY"

# clone_repo URL DIR COMMIT : clone (or reuse) and check out the commit the harness was written against
clone_repo() {
  local url=$1 dir=$2 commit=$3
  if [ ! -d "$dir/.git" ]; then
    git clone "$url" "$dir"
  fi
  if [ "${ATLAS_BENCH_UNPINNED:-0}" != "1" ]; then
    git -C "$dir" fetch --quiet origin "$commit" 2>/dev/null || true
    git -C "$dir" checkout --quiet "$commit"
  fi
  echo "[setup] $(basename "$dir") @ $(git -C "$dir" rev-parse --short HEAD)"
}

# download URL DEST : resumable download, skipped if DEST exists
download() {
  local url=$1 dest=$2
  mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ]; then echo "[setup] exists: $dest"; return 0; fi
  curl -fL --retry 5 --retry-delay 10 -C - -o "$dest.part" "$url" && mv "$dest.part" "$dest"
}

# hf_download REPO LOCAL_DIR [--include PATTERN ...]
hf_download() {
  local repo=$1 dir=$2; shift 2
  command -v huggingface-cli >/dev/null 2>&1 || pip install -q huggingface_hub
  huggingface-cli download "$repo" --local-dir "$dir" "$@"
}

env_exists() { $CONDA env list | awk '{print $1}' | grep -qx "$1"; }
in_env() { local env=$1; shift; $CONDA run --no-capture-output -n "$env" "$@"; }
