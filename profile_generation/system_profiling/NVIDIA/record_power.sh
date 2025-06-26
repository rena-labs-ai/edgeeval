#!/bin/bash

RESULTS_DIR=$1

if [ -z "$RESULTS_DIR" ]; then
    echo "Usage: $0 <results_dir>"
    exit 1
fi

tmux new-session -d sudo python3 record_power.py -o ${RESULTS_DIR}/power_data.csv

# Check if the command is running
while ! pgrep "record_power" > /dev/null; do
    sleep 1
done

power_pid=`pgrep -fo record_power`

echo "Power monitoring started with PID: $power_pid"