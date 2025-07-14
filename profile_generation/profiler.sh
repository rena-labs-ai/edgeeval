#!/bin/bash

browserd_path=/home/zhan/rena-labs/rena-core/rena-browserd
trace_path=./trace.json
config_path=/home/zhan/rena-labs/rena-core/examples/use-cases/filesystem-manipulation/browserd.toml
eval_path=/home/zhan/rena-labs/rena-core/examples/use-cases/filesystem-manipulation/eval.toml
results_dir=./results/filesystem-manipulation

python3 profiler.py \
    --browserd_path $browserd_path \
    --trace_path $trace_path \
    --config_path $config_path \
    --eval_path $eval_path \
    --results_dir $results_dir