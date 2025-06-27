#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: $0 <Results Directory>"
    exit 1
fi
PID=$1
INTERVAL=$2
RESULTS_DIR=$3

# Check if the results directory exists
if [ ! -d $RESULTS_DIR ]; then
    echo "Results directory not found!"
    exit 1
fi

get_pid_cpu(){
    local cpu_line=$(cat /proc/$PID/stat)
    local utime=$(echo $cpu_line | awk '{print $14}')
    local stime=$(echo $cpu_line | awk '{print $15}')
    local cutime=$(echo $cpu_line | awk '{print $16}')
    local cstime=$(echo $cpu_line | awk '{print $17}')
    
    local total_time=$((utime + stime + cutime + cstime))
    # echo "$utime $stime $cutime $cstime $total_time"
    echo "$total_time"
}

get_total_cpu(){
    # Read /proc/stat
    local cpu_line=$(cat /proc/stat | grep "^cpu ")
    local user=$(echo $cpu_line | awk '{print $2}')
    local nice=$(echo $cpu_line | awk '{print $3}')
    local system=$(echo $cpu_line | awk '{print $4}')
    local idle=$(echo $cpu_line | awk '{print $5}')

    # Calculate total
    local total=$((user + nice + system + idle))
    # echo "$user $nice $system $idle $total"
    echo "$total"
}


previous_time=$(date +%s%N)
previous_total=$(get_total_cpu)
previous_pid=$(get_pid_cpu)

while true; do
    # Sleep for a millisecond
    sleep $INTERVAL

    # Get current time and CPU data
    current_time=$(date +%s%N)
    current_total=$(get_total_cpu)
    current_pid=$(get_pid_cpu)
    
    delta_total=$((current_total - previous_total))
    delta_pid=$((current_pid - previous_pid))

    # Calculate CPU usage
    cpu_usage=$((100 * delta_pid / delta_total))

    # Log the results
    echo "$(date +%Y-%m-%d_%H:%M:%S.%N) | CPU: ${cpu_usage}" >> $RESULTS_DIR/cpu_pid_usage.log

    # Update previous values
    previous_time=$current_time
    previous_total=$current_total
    previous_pid=$current_pid

done
