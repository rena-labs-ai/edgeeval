### Background scripts for monitoring system-level metrics on NVIDIA GPU based systems

Running scripts in the background:

```
# Record CPU utilization
./get_cpu_usage.sh ${RESULTS_DIR} &
sleep 1
cpu_usage_pid=`pgrep "get_cpu_usage"`

# Record CPU Memory bandwidth
./record_cpu_compute.sh
cpu_mem_bw_pid=`pgrep "pcm-memory"`

# Record GPU compute and mem utilization
./record_gpu_mem_compute.sh ${RESULTS_DIR} &
sleep 1
gpu_utilization_pid=`pgrep "dcgmi"`

# Record power utilization
./record_power.sh
power_pid=`pgrep -fo record_power`

```

Killing background scripts:
```
kill -9 $cpu_usage_pid
sudo kill -9 $cpu_mem_bw_pid
kill -9 $gpu_utilization_pid
sudo kill -9 $power_pid
```
