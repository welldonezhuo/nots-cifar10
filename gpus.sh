#!/bin/bash
# Show GPU usage across the NOTS cluster: who is using what, and what's free.
# Usage: ./gpus.sh

source "$(dirname "$0")/config.sh"

echo "Querying NOTS cluster GPU status..."
echo ""

ssh $REMOTE 'bash -s' << 'SCRIPT'
NODES_FILE=$(mktemp)
JOBS_FILE=$(mktemp)
trap "rm -f $NODES_FILE $JOBS_FILE" EXIT

# 1) Extract GPU nodes from scontrol: node, total_gpus, gpu_type, partitions, state
#    Uses CfgTRES for total count and Gres for GPU type
scontrol show nodes 2>/dev/null | awk '
    BEGIN { node=""; gtype=""; gtotal=0; parts=""; state="" }
    /^NodeName=/ {
        if (node != "" && gtotal > 0)
            print node, gtotal, gtype, parts, state
        node = ""; gtype = ""; gtotal = 0; parts = ""; state = ""
        split($1, a, "="); node = a[2]
    }
    /Gres=gpu:/ {
        # e.g. Gres=gpu:lovelace:4(S:0-1)
        sub(/.*Gres=/, ""); sub(/\(.*/, ""); sub(/ .*/, "")
        n = split($0, f, ":")
        if (n >= 3) gtype = f[2]; else gtype = "gpu"
    }
    /CfgTRES=.*gres\/gpu=/ {
        # e.g. gres/gpu=4
        s = $0; sub(/.*gres\/gpu=/, "", s); sub(/,.*/, "", s)
        gtotal = int(s)
    }
    /Partitions=/ {
        sub(/.*Partitions=/, ""); sub(/ .*/, ""); parts = $0
    }
    /State=/ {
        sub(/.*State=/, ""); sub(/ .*/, ""); state = $0
    }
    END {
        if (node != "" && gtotal > 0)
            print node, gtotal, gtype, parts, state
    }
' | sort > "$NODES_FILE"

# 2) Get running GPU jobs: node, user, gpu_count, job_name
squeue --state=RUNNING -o "%u|%N|%b|%j" --noheader 2>/dev/null | grep -i gpu | while IFS='|' read -r user nodelist gres jobname; do
    # Parse count from e.g. "gres/gpu:lovelace:4" or "gres/gpu:1"
    count=$(echo "$gres" | rev | cut -d: -f1 | rev)
    expanded=$(scontrol show hostnames "$nodelist" 2>/dev/null)
    [ -z "$expanded" ] && expanded="$nodelist"
    for n in $expanded; do
        echo "$n $user $count $jobname"
    done
done > "$JOBS_FILE"

# 3) Print summary with colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'

echo "======================================"
echo "  NOTS Cluster GPU Summary"
echo "======================================"
echo ""

total_gpus=0
total_used=0

while read -r node total gtype partitions state; do
    # Sum GPU usage on this node
    used=0
    node_users=""
    while read -r jnode juser jcount jname; do
        [ "$jnode" != "$node" ] && continue
        used=$((used + jcount))
        node_users="${node_users}    ${juser} (${jcount} GPU, ${jname})\n"
    done < "$JOBS_FILE"

    free=$((total - used))
    [ $free -lt 0 ] && free=0
    total_gpus=$((total_gpus + total))
    total_used=$((total_used + used))

    if [ $used -eq 0 ]; then
        color=$GREEN
        tag="FREE"
    elif [ $free -gt 0 ]; then
        color=$YELLOW
        tag="PARTIAL"
    else
        color=$RED
        tag="FULL"
    fi

    printf "%-15s ${color}%-10s${RESET} %d/%d free  %-10s %s\n" \
        "$node" "[$tag]" "$free" "$total" "$gtype" "$partitions"
    if [ -n "$node_users" ]; then
        printf "$node_users"
    fi
done < "$NODES_FILE"

total_free=$((total_gpus - total_used))
echo ""
echo "======================================"
printf "Total: %d GPUs  |  ${RED}%d used${RESET}  |  ${GREEN}%d free${RESET}\n" "$total_gpus" "$total_used" "$total_free"
echo "======================================"
SCRIPT
