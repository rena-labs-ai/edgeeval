#!/bin/bash

# Uncomment if you have access to the dataset
# export HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXX"
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

export MODEL_NAME="mlc-ai/phi-2-q4f16_1-MLC"
export MODEL_PATH="HF://${MODEL_NAME}"
export TOKENIZER_NAME="mlc-ai/phi-2-q4f16_1-MLC"

export SERVER_ADDR="127.0.0.1"
export SERVER_PORT="8000"

export SHAREGPT_PATH="$PWD/ShareGPT_V3_unfiltered_cleaned_split.json"
export API_ENDPOINT="mlc"

sudo python3 -m mlc_llm.bench \
  --api-endpoint $API_ENDPOINT \
  --dataset sharegpt \
  --dataset-path $SHAREGPT_PATH \
  --tokenizer $TOKENIZER_NAME \
  --num-request 5 \
  --num-gpus 1 \
  --num-concurrent-requests 1 \
  --temperature 0.6 \
  --top-p 0.9 \
  --ignore-eos \
  --host $SERVER_ADDR \
  --port $SERVER_PORT || echo "Benchmark exited with error"

echo "Benchmark completed"
