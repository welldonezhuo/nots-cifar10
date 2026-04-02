#!/bin/bash
# Upload files to NOTS and submit the CIFAR-10 training job.
# Usage: ./launch.sh [--fresh]  (--fresh recreates the venv)

source "$(dirname "$0")/config.sh"
REMOTE_DIR=$SCRATCH/cifar10_job
VENV_DIR=$SCRATCH/venvs/cifar10
JOB_FILE=$REMOTE_DIR/.jobid

if [ "$1" = "--fresh" ]; then
    echo "Removing existing venv ..."
    ssh $REMOTE "rm -rf $VENV_DIR"
fi

echo "Uploading files ..."
ssh $REMOTE "mkdir -p $REMOTE_DIR"
# Substitute __SCRATCH__ placeholder with the actual path before uploading
sed "s|__SCRATCH__|$SCRATCH|g" submit.slurm | ssh $REMOTE "cat > $REMOTE_DIR/submit.slurm"
scp train_cifar10.py $REMOTE:$REMOTE_DIR/

echo "Submitting job ..."
JOB_ID=$(ssh $REMOTE "cd $REMOTE_DIR && sbatch --parsable submit.slurm")

if [ -z "$JOB_ID" ]; then
    echo "Failed to submit job."
    exit 1
fi

ssh $REMOTE "echo $JOB_ID > $JOB_FILE"
echo "Job submitted: $JOB_ID"
echo "Run ./status.sh to check progress, ./stop.sh to cancel."
