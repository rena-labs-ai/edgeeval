#!/bin/bash

# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json


export TOKENIZER_NAME="tinyllama" 
export SERVER_ADDR="127.0.0.1"
export SERVER_PORT="11434"

export SHAREGPT_PATH="$PWD/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
export API_ENDPOINT="ollama"

sudo python3 -m macos_bench \
  --tokenizer $TOKENIZER_NAME \
  --api-endpoint $API_ENDPOINT \
  --dataset sharegpt \
  --dataset-path $SHAREGPT_PATH \
  --num-request 5 \
  --num-gpus 1 \
  --num-concurrent-requests 1,2,4 \
  --host $SERVER_ADDR \
  --port $SERVER_PORT || echo "Benchmark exited with error"

echo "Benchmark completed"
