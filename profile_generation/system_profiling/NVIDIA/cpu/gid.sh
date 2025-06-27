#!/bin/bash

if [ $# -ne 4 ]; then
    echo "Usage: $0 <Results Directory>"
    exit 1
fi
GID=$1
INTERVAL=$2
RESULTS_DIR=$3
GNAME=$4

# Check if the results directory exists
if [ ! -d $RESULTS_DIR ]; then
    echo "Results directory not found!"
    # exit 1
    mkdir -p $RESULTS_DIR
fi

get_pid_cpu(){
    local pid=$1
    local cpu_line=$(cat /proc/$pid/stat 2>/dev/null) || return 1
    local utime=$(echo $cpu_line | awk '{print $14}')
    local stime=$(echo $cpu_line | awk '{print $15}')
    # local cutime=$(echo $cpu_line | awk '{print $16}')
    # local cstime=$(echo $cpu_line | awk '{print $17}')
    
    local total_time=$((utime + stime))
    # echo "$utime $stime $cutime $cstime $total_time"
    echo "$total_time"
}

get_gid_cpu(){
    local PID_LIST=$(pgrep -g $GID)
    local total_time=0
    for pid in $PID_LIST; do
        total_time=$((total_time + $(get_pid_cpu $pid)))
    done
    echo "$total_time"
}

get_total_cpu(){
    # Read /proc/stat
    local cpu_line=$(cat /proc/stat | grep "^cpu ")
    local user=$(echo $cpu_line | awk '{print $2}')
    local nice=$(echo $cpu_line | awk '{print $3}')
    local system=$(echo $cpu_line | awk '{print $4}')
    local idle=$(echo $cpu_line | awk '{print $5}')
    local iowait=$(echo $cpu_line | awk '{print $6}')
    local irq=$(echo $cpu_line | awk '{print $7}')
    local softirq=$(echo $cpu_line | awk '{print $8}')

    # Calculate total
    # local total=$((user + nice + system + idle))
    local total=$((user + nice + system + idle + iowait + irq + softirq))
    # echo "$user $nice $system $idle $total"
    echo "$total"
}


previous_time=$(date +%s%N)
previous_gid=$(get_gid_cpu)
previous_total=$(get_total_cpu)

while true; do
    # Sleep for a millisecond
    sleep $INTERVAL

    # Get current time and CPU data
    current_time=$(date +%s%N)
    current_gid=$(get_gid_cpu)
    current_total=$(get_total_cpu)
    
    delta_gid=$((current_gid - previous_gid))
    delta_total=$((current_total - previous_total))

    # Calculate CPU usage
    cpu_usage=$((100 * delta_gid / delta_total))
    if [ $cpu_usage -gt 100 ]; then
        cpu_usage=100
    fi

    # Log the results
    echo "$(date +%Y-%m-%d_%H:%M:%S.%N) | CPU: ${cpu_usage}" >> $RESULTS_DIR/cpu_${GNAME}_gid_usage.log

    # Update previous values
    previous_time=$current_time
    previous_total=$current_total
    previous_gid=$current_gid

done
