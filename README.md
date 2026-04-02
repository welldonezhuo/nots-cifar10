# Training CNNs on the Rice NOTS Cluster

This repository provides a complete example of training a CNN on CIFAR-10 using the [NOTS cluster](https://kb.rice.edu/147970) at Rice University. It includes a training script, a SLURM job script, and helper scripts for job management and monitoring with TensorBoard.

## CIFAR-10

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) is a standard image classification benchmark of 60,000 32x32 color images split into 50,000 training and 10,000 test images across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset is automatically downloaded on the first run.

## Prerequisites

1. **Rice NetID** with an active NOTS account
2. **SSH access** to `nots.rice.edu` (passwordless via SSH keys recommended)
3. **Rice VPN** (Cisco AnyConnect) if connecting from off-campus

Set up passwordless SSH (one-time):
```bash
# If you don't have an SSH key yet, generate one first:
ssh-keygen -t ed25519
# Press Enter to accept the default file location (~/.ssh/id_ed25519)
# You can set a passphrase or leave it empty for convenience

# Then copy your public key to NOTS:
ssh-copy-id YOUR_NETID@nots.rice.edu
# Enter your Rice password when prompted — this is the last time you'll need it
```

You can verify it worked by running `ssh YOUR_NETID@nots.rice.edu` — it should log in without asking for a password.

## Quick Start

```bash
# 1. Edit config.sh and set your Rice NetID
# 2. Submit a training job
./launch.sh

# 3. Monitor progress
./status.sh

# 4. (Optional) Watch live training curves
./tensorboard.sh
# Then open http://localhost:6006 in your browser

# 5. Download trained model weights when done
./download.sh
```

## File Descriptions

### Training Code

#### `train_cifar10.py`
The main training script. Trains a 6-layer CNN (3 convolutional blocks with BatchNorm + MaxPool, followed by fully connected layers) on CIFAR-10.

**Architecture:**
- Conv blocks: 64 -> 128 -> 256 channels, each with two Conv2d + BatchNorm + ReLU layers and a MaxPool2d
- Classifier: Linear(4096, 512) -> ReLU -> Dropout(0.5) -> Linear(512, 10)

**Training details:**
- Optimizer: SGD with momentum 0.9 and weight decay 5e-4
- Scheduler: Cosine annealing over the full training run
- Data augmentation: Random crop (32x32 with padding 4) and random horizontal flip
- Expected test accuracy: ~92-93%

**Command-line arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 128 | Mini-batch size |
| `--lr` | 0.1 | Initial learning rate |
| `--data-dir` | `./data` | Where to download/read CIFAR-10 |
| `--output-dir` | `./output` | Where to save model weights |
| `--log-dir` | `./logs` | Where to write TensorBoard logs |

**TensorBoard logging:**
- **Loss** (train and test on the same chart)
- **Accuracy** (train and test on the same chart)
- **Learning rate** schedule
- **Predictions** (every 10 epochs): A figure showing 5 fixed test images with their true labels alongside horizontal bar charts of the predicted class probabilities for all 10 CIFAR-10 classes. Green bars indicate the true class; red bars indicate an incorrect top prediction.

**Saved outputs:**
- `best_model.pth` — weights from the epoch with the highest test accuracy
- `final_model.pth` — weights from the last epoch

### SLURM Job Script

#### `submit.slurm`
The SLURM batch script that runs on the cluster. Handles environment setup and launches training.

**What it does:**
1. Requests 1 GPU (A40 by default), 4 CPUs, 64 GB RAM, and 30 minutes of wall time on the `scavenge` partition
2. Creates a Python virtual environment on `/scratch/` (first run only) with PyTorch, torchvision, TensorBoard, and matplotlib installed via pip
3. Launches `train_cifar10.py` with `srun`

**Key SLURM parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `--partition` | `scavenge` | Uses idle condo nodes; max 1 hour wall time |
| `--account` | `commons` | Default account for all users |
| `--gres` | `gpu:ampere:1` | Requests one A40 GPU |
| `--time` | `00:30:00` | Maximum wall time |
| `--cpus-per-task` | `4` | CPU cores for data loading |
| `--mem-per-cpu` | `16G` | Memory per CPU core |

**Available GPU types on NOTS** (change the `--gres` line):
```bash
#SBATCH --gres=gpu:volta:1      # Tesla V100 (32 GB)
#SBATCH --gres=gpu:ampere:1     # NVIDIA A40 (48 GB)
#SBATCH --gres=gpu:lovelace:1   # NVIDIA L40S (48 GB)
#SBATCH --gres=gpu:1            # Any available GPU
```

**Available partitions:**
| Partition | Max Wall Time | Use Case |
|-----------|--------------|----------|
| `commons` | 24 hours | General purpose (default) |
| `scavenge` | 1 hour | Short jobs on idle condo nodes |
| `long` | 72 hours | Multi-day training runs |
| `debug` | 30 minutes | Quick tests and interactive work |

**Virtual environment:** The venv is created at `/scratch/YOUR_NETID/venvs/cifar10` on the first run. Subsequent jobs reuse it. If you need to add new packages or rebuild the environment, use `./launch.sh --fresh`.

**Why pip instead of system modules?** The PyTorch modules installed on NOTS are built against MPI libraries (e.g., `libpsm2`, `libibverbs`) that are not available on all node types, particularly `scavenge` nodes. Installing PyTorch via pip from the official PyTorch wheel index avoids these shared library issues.

### Helper Scripts

All helper scripts run from your **local machine** and communicate with NOTS over SSH.

#### `launch.sh`
Uploads `submit.slurm` and `train_cifar10.py` to the cluster and submits the job.

```bash
./launch.sh          # Submit job (reuses existing venv)
./launch.sh --fresh  # Delete and recreate the venv before submitting
```

The job ID is saved on the remote server so that `status.sh`, `stop.sh`, and `download.sh` can find it automatically.

#### `status.sh`
Shows human-readable job status: whether the job is pending, running, or completed, which node it's on, elapsed time, the last 5 lines of training output, and any errors.

```bash
./status.sh
```

Example output:
```
Job ID: 8015922

Status:    RUNNING
Node:      bc13u30n1
Partition: scavenge
Elapsed:   12:34 / 00:30:00

Latest output:
Epoch 25/50 | Train Loss: 0.2341 | Test Loss: 0.3812 | Train Acc: 91.23% | Test Acc: 89.45%
```

#### `stop.sh`
Cancels the currently running job.

```bash
./stop.sh
```

#### `tensorboard.sh`
Opens an SSH tunnel and starts TensorBoard on the NOTS login node, allowing you to view live training curves at `http://localhost:6006` in your local browser.

```bash
./tensorboard.sh
# Open http://localhost:6006
# Press Ctrl+C to stop
```

This uses only two persistent SSH connections (one tunnel, one for TensorBoard). It cleans up both when you press Ctrl+C.

**Note:** The venv must already exist on the cluster (i.e., at least one job must have run) before TensorBoard can start.

#### `download.sh`
Downloads the trained model weights (`best_model.pth` and `final_model.pth`) from the cluster to a local `./output/` directory.

```bash
./download.sh
```

## Customization for Your Own Project

### Step 1: Set Your NetID

Edit `config.sh` and replace `vo9` with your Rice NetID:

```bash
NETID="your_netid_here"
```

All scripts read from this file — no other changes needed.

### Step 2: Use Your Own Dataset

CIFAR-10 is downloaded automatically by `torchvision`. If you are using a different dataset, you need to upload it to `/scratch/YOUR_NETID/` on the cluster before submitting your job:

```bash
scp -r /path/to/your/dataset YOUR_NETID@nots.rice.edu:/scratch/YOUR_NETID/my_dataset
```

Then update the `--data-dir` argument in `submit.slurm` to point to that path. For large datasets, consider uploading a tarball and extracting it on the cluster to avoid slow transfers of many small files:

```bash
tar czf my_dataset.tar.gz my_dataset/
scp my_dataset.tar.gz YOUR_NETID@nots.rice.edu:/scratch/YOUR_NETID/
ssh YOUR_NETID@nots.rice.edu "cd /scratch/YOUR_NETID && tar xzf my_dataset.tar.gz"
```

### Step 3: Modify the Training Script

Replace `train_cifar10.py` with your own training code. Make sure it:
- Accepts `--data-dir`, `--output-dir`, and `--log-dir` arguments
- Saves model weights to `--output-dir`
- Writes TensorBoard logs to `--log-dir` (if you want TensorBoard support)

### Step 4: Adjust SLURM Resources

Edit `submit.slurm` to match your project's needs:
- **More time:** Change `--time` (e.g., `04:00:00` for 4 hours) and `--partition` (use `commons` for up to 24h, `long` for up to 72h)
- **More GPUs:** Change `--gres=gpu:ampere:2` and update your training script for multi-GPU (e.g., `DistributedDataParallel`)
- **More memory:** Increase `--mem-per-cpu` or add more `--cpus-per-task`
- **Additional pip packages:** Add them to the `pip install` line in `submit.slurm`, then run `./launch.sh --fresh`

### Step 5: Update the Training Command

Edit the `srun python ...` line at the bottom of `submit.slurm` to match your script's filename and arguments.

## Troubleshooting

**Job stuck in PENDING state:**
- Check reason with `./status.sh` or `squeue -u YOUR_NETID`
- `(Resources)` — waiting for a node; try a different GPU type or partition
- `(ReqNodeNotAvail)` — requested node type is unavailable; switch GPU type in `submit.slurm`

**`ModuleNotFoundError` or missing shared library errors:**
- Run `./launch.sh --fresh` to rebuild the virtual environment
- Make sure `submit.slurm` has the necessary `pip install` commands

**Disk quota exceeded:**
- All data and environments should be on `/scratch/`, not `/home/`. Check that your paths in `submit.slurm` point to `/scratch/YOUR_NETID/`
- `/scratch/` has much more space than `/home/` but files may be purged after extended inactivity

**SSH connection refused or timed out:**
- Make sure you're on the Rice VPN if off-campus
- Avoid scripts that open many rapid SSH connections — this can trigger the campus firewall. All helper scripts in this repo are designed to minimize SSH connections

**TensorBoard not showing data:**
- Make sure at least one epoch has completed (check with `./status.sh`)
- Try refreshing the browser or clicking the reload button in TensorBoard

## Useful SLURM Commands

```bash
# Check your jobs
squeue -u YOUR_NETID

# See available GPUs
sinfo -o '%P %N %G %t' -p scavenge

# See detailed node info
scontrol show node NODENAME

# Check available partitions and limits
sacctmgr show assoc cluster=nots user=YOUR_NETID

# Cancel a job
scancel JOB_ID

# Check past job history
sacct -u YOUR_NETID --format=JobID,JobName,Partition,State,Elapsed,ExitCode
```

## Cluster Resources

The NOTS cluster has the following GPU types:

| GPU | Memory | Nodes | GPUs/Node | Total |
|-----|--------|-------|-----------|-------|
| NVIDIA L40S | 48 GB | 12 | 4 | 48 |
| Tesla V100 | 32 GB | 16 | 2 | 32 |
| NVIDIA A40 | 48 GB | 4 | 4-8 | 28 |

For more information, see the [CRC NOTS documentation](https://kb.rice.edu/147970) or contact the Rice helpdesk at helpdesk@rice.edu.
