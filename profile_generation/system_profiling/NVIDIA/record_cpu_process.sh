#!/bin/bash

# Script to calculate CPU utilization for a given PID
# Usage: ./cpu_util.sh <PID> [interval_seconds]

# Check if PID is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <PID> [interval_seconds]"
    echo "Example: $0 1234 1"
    exit 1
fi

PID=$1
INTERVAL=${2:-1}  # Default to 1 second if not specified

# Check if PID exists
if [ ! -d "/proc/$PID" ]; then
    echo "Error: Process $PID does not exist"
    exit 1
fi

# Function to get CPU stats for a process
get_cpu_stats() {
    local pid=$1
    if [ ! -f "/proc/$pid/stat" ]; then
        echo "Error: Cannot read /proc/$pid/stat"
        return 1
    fi
    
    # Read the stat file and extract relevant fields
    local stat_line=$(cat /proc/$pid/stat)
    
    # Parse the stat line (handling process names with spaces/parentheses)
    # Use awk to properly parse the fields after the command name in parentheses
    local stats=$(echo "$stat_line" | awk '{
        # Find the last closing parenthesis to handle command names with spaces
        for(i=1; i<=NF; i++) {
            if($i ~ /\)$/) {
                start_field = i + 1
                break
            }
        }
        # Print relevant fields: state, utime, stime (adjusting for 0-based indexing)
        print $(start_field + 11), $(start_field + 12)  # utime, stime
    }')
    
    echo $stats
}

# Function to get system uptime in centiseconds (for calculating elapsed time)
get_uptime_cs() {
    awk '{print int($1 * 100)}' /proc/uptime
}

# Get number of CPU cores
CPU_CORES=$(nproc)

# Get clock ticks per second
CLOCK_TICKS=$(getconf CLK_TCK)

echo "Monitoring PID $PID for $INTERVAL second(s)..."
echo "CPU cores: $CPU_CORES, Clock ticks per second: $CLOCK_TICKS"
echo

# First snapshot
STATS1=$(get_cpu_stats $PID)
if [ $? -ne 0 ]; then
    exit 1
fi

UPTIME1=$(get_uptime_cs)
UTIME1=$(echo $STATS1 | awk '{print $1}')
STIME1=$(echo $STATS1 | awk '{print $2}')
TOTAL_TIME1=$((UTIME1 + STIME1))

echo "Taking first measurement..."
echo "  User time: $UTIME1 ticks"
echo "  System time: $STIME1 ticks"
echo "  Total time: $TOTAL_TIME1 ticks"

# Wait for the specified interval
sleep $INTERVAL

# Check if process still exists
if [ ! -d "/proc/$PID" ]; then
    echo "Process $PID terminated during monitoring"
    exit 1
fi

# Second snapshot
STATS2=$(get_cpu_stats $PID)
if [ $? -ne 0 ]; then
    exit 1
fi

UPTIME2=$(get_uptime_cs)
UTIME2=$(echo $STATS2 | awk '{print $1}')
STIME2=$(echo $STATS2 | awk '{print $2}')
TOTAL_TIME2=$((UTIME2 + STIME2))

echo
echo "Taking second measurement..."
echo "  User time: $UTIME2 ticks"
echo "  System time: $STIME2 ticks"
echo "  Total time: $TOTAL_TIME2 ticks"

# Calculate differences
TIME_DIFF=$((TOTAL_TIME2 - TOTAL_TIME1))
UPTIME_DIFF=$((UPTIME2 - UPTIME1))

echo
echo "Calculations:"
echo "  CPU time difference: $TIME_DIFF ticks"
echo "  Uptime difference: $UPTIME_DIFF centiseconds"

# Calculate CPU utilization percentage
# Formula: (cpu_time_diff / clock_ticks) / (uptime_diff / 100) * 100
# Simplified: (cpu_time_diff * 100) / (uptime_diff * clock_ticks / 100)
# Further simplified: (cpu_time_diff * 10000) / (uptime_diff * clock_ticks)

if [ $UPTIME_DIFF -eq 0 ]; then
    echo "Error: No time elapsed between measurements"
    exit 1
fi

# Use awk for floating point calculation
CPU_PERCENT=$(awk "BEGIN {
    cpu_usage = ($TIME_DIFF * 100) / ($UPTIME_DIFF * $CLOCK_TICKS / 100)
    printf \"%.2f\", cpu_usage
}")

echo
echo "=== RESULT ==="
echo "Process $PID CPU utilization: ${CPU_PERCENT}%"

# Get process name for reference
PROCESS_NAME=$(cat /proc/$PID/comm 2>/dev/null || echo "unknown")
echo "Process name: $PROCESS_NAME"

# Optional: Show comparison with system load
echo
echo "=== SYSTEM CONTEXT ==="
echo "System load average: $(cat /proc/loadavg | awk '{print $1, $2, $3}')"
echo "Total CPU cores: $CPU_CORES"