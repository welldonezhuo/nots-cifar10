#!/bin/bash
# Download trained model weights from NOTS to local machine.
# Usage: ./download.sh

source "$(dirname "$0")/config.sh"
REMOTE_OUTPUT=$SCRATCH/cifar10_output
LOCAL_OUTPUT=./output

mkdir -p "$LOCAL_OUTPUT"

echo "Downloading model weights ..."
scp $REMOTE:$REMOTE_OUTPUT/best_model.pth $REMOTE:$REMOTE_OUTPUT/final_model.pth "$LOCAL_OUTPUT/" 2>&1

if [ $? -eq 0 ]; then
    echo "Downloaded to $LOCAL_OUTPUT/:"
    ls -lh "$LOCAL_OUTPUT"/*.pth
else
    echo "Download failed. Check if training has completed (./status.sh)."
fi
