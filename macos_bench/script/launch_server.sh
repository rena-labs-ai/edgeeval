export MODEL_NAME="mlc-ai/Llama-3.2-3B-Instruct-q4f16_0-MLC"
export MODEL_PATH="HF://${MODEL_NAME}"

export SERVER_ADDR="127.0.0.1"
export SERVER_PORT="8000"

echo "Starting MLC LLM server (Metal backend)..."
python3 -m mlc_llm serve $MODEL_PATH \
  --mode server \
  --host $SERVER_ADDR \
  --port $SERVER_PORT \
  --device metal \
  --prefix-cache-mode radix \
  --enable-debug
