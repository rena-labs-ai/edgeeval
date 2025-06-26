#!/bin/bash

RESULTS_DIR=$1
if [ -z "$RESULTS_DIR" ]; then
    echo "Usage: $0 <results_dir>"
    exit 1
fi

tmux new-session -d "sudo pcm-memory 0.05 -s -csv=${RESULTS_DIR}/cpu-mem-bw.csv"

while ! pgrep "pcm-memory" > /dev/null; do
    sleep 1
done