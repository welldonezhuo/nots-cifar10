#!/bin/bash
#SBATCH --job-name=flux-dev
#SBATCH --partition=scavenge
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Current directory: $(pwd)"
echo "GPU info:"
nvidia-smi || true

module purge

# 用真正的 scratch/work 目录，不要再用字面量 __SCRATCH__
if [ -n "${SCRATCH:-}" ]; then
    BASE_DIR="$SCRATCH/flux_${SLURM_JOB_ID}"
elif [ -d "/tmp/$USER" ]; then
    BASE_DIR="/tmp/$USER/flux_${SLURM_JOB_ID}"
else
    BASE_DIR="/tmp/flux_${USER}_${SLURM_JOB_ID}"
fi

CACHE_ROOT="$BASE_DIR/cache"
TMP_ROOT="$BASE_DIR/tmp"
OUT_ROOT="$BASE_DIR/output"
VENV_DIR="$BASE_DIR/venv"

mkdir -p "$CACHE_ROOT" "$TMP_ROOT" "$OUT_ROOT"

export HF_HOME="$CACHE_ROOT/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TMPDIR="$TMP_ROOT"
export TEMP="$TMP_ROOT"
export TMP="$TMP_ROOT"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"

mkdir -p \
    "$HF_HOME" \
    "$HF_HUB_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$HUGGINGFACE_HUB_CACHE" \
    "$XDG_CACHE_HOME" \
    "$PIP_CACHE_DIR"

echo "BASE_DIR=$BASE_DIR"
echo "HF_HOME=$HF_HOME"
echo "TMPDIR=$TMPDIR"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir diffusers transformers accelerate sentencepiece protobuf safetensors huggingface_hub

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

srun --unbuffered python -u flux_generate.py \
    --prompt "A real polar bear wearing a red wool hat and a fabric bowtie, carrying a light wood board with the text 'comp646' on it, standing in front of a red brick wall, ultra photorealistic, professional wildlife photography, detailed fur texture, natural lighting, shallow depth of field, DSLR camera, 85mm lens, high dynamic range, sharp focus" \
    --output "$OUT_ROOT/flux-dev.png"

echo "Output saved to: $OUT_ROOT/flux-dev.png"
echo "Job finished at $(date)"