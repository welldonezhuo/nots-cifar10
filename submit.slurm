#!/bin/bash
#SBATCH --job-name=flux-dev
#SBATCH --partition=scavenge
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=00:30:00
#SBATCH --output=__SCRATCH__/flux_%j.out
#SBATCH --error=__SCRATCH__/flux_%j.err

echo "Job started on $(hostname) at $(date)"
echo "Current directory: $(pwd)"
echo "GPU info:"
nvidia-smi

module purge

# Install packages in a self-contained venv, following the teacher's style
VENV_DIR=$HOME/venvs/flux
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --no-cache-dir --upgrade pip
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install --no-cache-dir diffusers transformers accelerate sentencepiece protobuf safetensors huggingface_hub
else
    source "$VENV_DIR/bin/activate"
fi

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# If you have a Hugging Face token, put it here
# export HF_TOKEN=your_huggingface_token_here

srun python flux_generate.py \
    --prompt "A real polar bear wearing a red wool hat and a fabric bowtie, carrying a light wood board with the text 'comp646' on it, standing in front of a red brick wall, ultra photorealistic, professional wildlife photography, detailed fur texture, natural lighting, shallow depth of field, DSLR camera, 85mm lens, high dynamic range, sharp focus" \
    --output __SCRATCH__/flux-dev.png

echo "Job finished at $(date)"