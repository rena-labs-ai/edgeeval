export MODEL_NAME="mlc-ai/phi-2-q4f16_1-MLC"
export MODEL_PATH="HF://${MODEL_NAME}"

export SERVER_ADDR="127.0.0.1"
export SERVER_PORT="8000"

# === [3] Launch MLC-LLM Server ===
echo "[2/5] Starting MLC LLM server (Metal backend)..."
python3 -m mlc_llm serve $MODEL_PATH \
  --mode server \
  --host $SERVER_ADDR \
  --port $SERVER_PORT \
  --device metal \
  --prefix-cache-mode disable \
  --enable-debug
