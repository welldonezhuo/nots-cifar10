#!/bin/bash
# Cancel the current CIFAR-10 training job on NOTS.
# Usage: ./stop.sh

source "$(dirname "$0")/config.sh"
REMOTE_DIR=$SCRATCH/cifar10_job
JOB_FILE=$REMOTE_DIR/.jobid

JOB_ID=$(ssh $REMOTE "cat $JOB_FILE 2>/dev/null")

if [ -z "$JOB_ID" ]; then
    echo "No job ID found. Nothing to cancel."
    exit 1
fi

ssh $REMOTE "scancel $JOB_ID && rm -f $JOB_FILE"
echo "Job $JOB_ID cancelled."
