#!/bin/bash
# Show human-friendly status of the current CIFAR-10 training job on NOTS.
# Usage: ./status.sh

source "$(dirname "$0")/config.sh"
REMOTE_DIR=$SCRATCH/cifar10_job
JOB_FILE=$REMOTE_DIR/.jobid

JOB_ID=$(ssh $REMOTE "cat $JOB_FILE 2>/dev/null")

if [ -z "$JOB_ID" ]; then
    echo "No active job found."
    exit 0
fi

# Get job info in one SSH call
INFO=$(ssh $REMOTE "
    # Job status from squeue
    STATUS=\$(squeue -j $JOB_ID -h -o '%T|%M|%N|%P|%l|%R' 2>/dev/null)

    if [ -z \"\$STATUS\" ]; then
        # Job no longer in queue — check if it completed
        STATE=\$(sacct -j $JOB_ID --format=State,Elapsed,ExitCode -n -P 2>/dev/null | head -1)
        if [ -n \"\$STATE\" ]; then
            echo \"FINISHED|\$STATE\"
        else
            echo \"UNKNOWN\"
        fi
    else
        echo \"ACTIVE|\$STATUS\"
    fi

    # Last few lines of output
    OUTFILE=$REMOTE_DIR/cifar10_${JOB_ID}.out
    if [ -f \"\$OUTFILE\" ]; then
        echo \"---OUTPUT---\"
        tail -5 \"\$OUTFILE\"
    fi

    # Check for errors
    ERRFILE=$REMOTE_DIR/cifar10_${JOB_ID}.err
    if [ -f \"\$ERRFILE\" ] && [ -s \"\$ERRFILE\" ]; then
        # Only show non-pip-notice errors
        ERRS=\$(grep -v '^\[notice\]' \"\$ERRFILE\" | grep -v '^WARNING' | tail -3)
        if [ -n \"\$ERRS\" ]; then
            echo \"---ERRORS---\"
            echo \"\$ERRS\"
        fi
    fi
")

echo "Job ID: $JOB_ID"
echo ""

# Parse status line
STATUS_LINE=$(echo "$INFO" | head -1)

if [[ "$STATUS_LINE" == ACTIVE* ]]; then
    IFS='|' read -r _ STATE ELAPSED NODE PARTITION TIMELIMIT REASON <<< "$STATUS_LINE"
    echo "Status:    $STATE"
    echo "Node:      $NODE"
    echo "Partition: $PARTITION"
    echo "Elapsed:   $ELAPSED / $TIMELIMIT"
elif [[ "$STATUS_LINE" == FINISHED* ]]; then
    IFS='|' read -r _ STATE ELAPSED EXITCODE <<< "$STATUS_LINE"
    echo "Status:    COMPLETED ($STATE)"
    echo "Elapsed:   $ELAPSED"
    echo "Exit code: $EXITCODE"
else
    echo "Status:    Unknown (job may have been cancelled)"
fi

# Show output
OUTPUT=$(echo "$INFO" | sed -n '/---OUTPUT---/,/---ERRORS---/{ /---OUTPUT---/d; /---ERRORS---/d; p }')
if [ -z "$OUTPUT" ]; then
    OUTPUT=$(echo "$INFO" | sed -n '/---OUTPUT---/,${ /---OUTPUT---/d; p }')
fi
if [ -n "$OUTPUT" ]; then
    echo ""
    echo "Latest output:"
    echo "$OUTPUT"
fi

# Show errors
ERRORS=$(echo "$INFO" | sed -n '/---ERRORS---/,${ /---ERRORS---/d; p }')
if [ -n "$ERRORS" ]; then
    echo ""
    echo "Errors:"
    echo "$ERRORS"
fi
