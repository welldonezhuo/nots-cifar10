#!/bin/bash
# Start TensorBoard on NOTS and open it locally via SSH tunnel.
# Usage: ./tensorboard.sh
# Press Ctrl+C to stop.

source "$(dirname "$0")/config.sh"
LOG_DIR=$SCRATCH/cifar10_logs
PORT=6006

cleanup() {
    echo ""
    echo "Shutting down tunnel and TensorBoard ..."
    kill $TUNNEL_PID 2>/dev/null
    ssh $REMOTE "pkill -f 'tensorboard.*--port $PORT'" 2>/dev/null
    exit 0
}
trap cleanup INT TERM

echo "Starting SSH tunnel (localhost:$PORT -> NOTS:$PORT) ..."
ssh -N -L $PORT:localhost:$PORT $REMOTE &
TUNNEL_PID=$!
sleep 2

if ! kill -0 $TUNNEL_PID 2>/dev/null; then
    echo "Failed to establish SSH tunnel."
    exit 1
fi

echo "Starting TensorBoard on NOTS ..."
echo "Open http://localhost:$PORT in your browser."
echo "Press Ctrl+C to stop."
echo "---"
ssh $REMOTE "source $SCRATCH/venvs/cifar10/bin/activate && tensorboard --logdir $LOG_DIR --port $PORT --bind_all 2>&1"

cleanup
