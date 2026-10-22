#!/usr/bin/env bash
# One-shot setup: the two conda environments the pipeline needs, the submodules and their patches.
#
#   ./setup.sh              # everything
#   ./setup.sh --skip-fastgs  # only the main environment (no surface reconstruction)
#
# Needs: conda, git, an NVIDIA GPU with a recent driver. Takes a while (downloads several GB and
# compiles the FastGS CUDA extensions). Safe to re-run: finished steps are skipped or redone in place.
set -euo pipefail
cd "$(dirname "$0")"

MAIN_ENV=${SNAP3D_ENV:-snap3d}
FASTGS_ENV=${SNAP3D_FASTGS_ENV:-fastgs}
SKIP_FASTGS=0
[ "${1:-}" = "--skip-fastgs" ] && SKIP_FASTGS=1

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { echo "error: $*" >&2; exit 1; }
have_env() { conda env list | awk '{print $1}' | grep -qx "$1"; }
in_env() { local env=$1; shift; conda run --no-capture-output -n "$env" "$@"; }

command -v conda >/dev/null || die "conda not found (install Miniconda/Anaconda first)"
command -v git >/dev/null || die "git not found"

say "1/4 main environment ($MAIN_ENV)"
if have_env "$MAIN_ENV"; then
  conda env update -n "$MAIN_ENV" -f environment.yml
else
  conda env create -n "$MAIN_ENV" -f environment.yml
fi
in_env "$MAIN_ENV" pip install -e ".[all]"

say "2/4 submodules and patches"
git submodule update --init third_party/FastGS
./third_party/patches/apply.sh || die "patches did not apply cleanly (see apply.sh --check)"

if [ "$SKIP_FASTGS" = 1 ]; then
  say "skipping the fastgs environment (--skip-fastgs)"
else
  say "3/4 FastGS environment ($FASTGS_ENV)"
  if have_env "$FASTGS_ENV"; then
    conda env update -n "$FASTGS_ENV" -f environment-fastgs.yml
  else
    conda env create -n "$FASTGS_ENV" -f environment-fastgs.yml
  fi
  FASTGS_PREFIX=$(conda run -n "$FASTGS_ENV" python -c 'import sys; print(sys.prefix)')

  say "4/4 FastGS CUDA extensions"
  # The extensions are built with torch 2.11 / CUDA 12.8, so they need an nvcc of that version.
  if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
    :
  elif [ -x /usr/local/cuda-12.8/bin/nvcc ]; then
    CUDA_HOME=/usr/local/cuda-12.8
  else
    echo "no CUDA 12.8 toolkit found; installing one into the $FASTGS_ENV environment"
    conda install -y -n "$FASTGS_ENV" -c "nvidia/label/cuda-12.8.1" cuda-toolkit
    CUDA_HOME=$FASTGS_PREFIX
  fi
  export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
  # Build only for this machine's GPU (e.g. 12.0 -> 12.0), which also keeps the build short.
  if arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1) && [ -n "$arch" ]; then
    export TORCH_CUDA_ARCH_LIST="$arch"
  fi
  for m in diff-gaussian-rasterization_fastgs diff-plane-rasterization fused-ssim simple-knn; do
    in_env "$FASTGS_ENV" pip install "third_party/FastGS/submodules/$m" --no-build-isolation --force-reinstall --no-deps
  done
  in_env "$FASTGS_ENV" python -c "import diff_plane_rasterization, diff_gaussian_rasterization_fastgs, fused_ssim, simple_knn; print('FastGS extensions import ok')"
fi

cat <<MSG

Done. Next:
  conda activate $MAIN_ENV
  hf auth login        # SAM3 is gated: request access to facebook/sam3 first
  python -m pipeline.orchestrator.runner --config pipeline/orchestrator/configs/photo_pipeline.yaml \\
      --input /path/to/photos --text "the object"
MSG
