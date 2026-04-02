#!/bin/bash
#SBATCH --job-name=cifar10-cnn
#SBATCH --partition=scavenge
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=00:30:00
#SBATCH --output=__SCRATCH__/cifar10_job/cifar10_%j.out
#SBATCH --error=__SCRATCH__/cifar10_job/cifar10_%j.err

echo "Job started on $(hostname) at $(date)"
echo "GPU info:"
nvidia-smi

module purge

# Install PyTorch and torchvision via pip in a self-contained venv
VENV_DIR=__SCRATCH__/venvs/cifar10
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install --no-cache-dir tensorboard matplotlib
else
    source "$VENV_DIR/bin/activate"
fi

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "torchvision: $(python -c 'import torchvision; print(torchvision.__version__)')"

srun python train_cifar10.py \
    --epochs 50 \
    --batch-size 128 \
    --lr 0.1 \
    --data-dir __SCRATCH__/cifar10_data \
    --output-dir __SCRATCH__/cifar10_output \
    --log-dir __SCRATCH__/cifar10_logs

echo "Job finished at $(date)"
