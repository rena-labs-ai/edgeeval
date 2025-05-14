import mlx.core as mx
import mlx.utils as mutils
import json
import time
import os
from mlx_lm import load

def save_trace_to_json(trace_dict, filename="trace.json"):
    try:
        mutils.export_trace(trace_dict, filename)
        print(f"Profiler trace saved to {filename}. Open with chrome://tracing or Perfetto UI.")
    except AttributeError:
        print(f"mlx.utils.export_trace not found. Saving a simplified trace to {filename}.")
        with open(filename, 'w') as f:
            json.dump(list(trace_dict.values()), f)
        print(f"Profiler trace (simplified/raw) saved to {filename}.")


if __name__ == "__main__":
    model_name = "mlx-community/Phi-3-mini-4k-instruct-4bit-no-q-embed"

    prompt_text = "What are the main components of an MLX application?"
    max_tokens_to_generate = 50

    print(f"Loading model: {model_name}...")
    try:
        model, tokenizer = load(model_name, tokenizer_config={})
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        print("Please ensure the model identifier is correct and you have internet access.")
        print("You might need to use a model explicitly converted to MLX format.")
        exit(1)
    
    print("Model and tokenizer loaded.")

    input_ids = tokenizer.encode(prompt_text, return_tensors="np")
    input_ids = mx.array(input_ids)

    print(f"Input token IDs shape: {input_ids.shape}")

    print("Running a warm-up iteration...")
    try:
        _ = model(input_ids)
        mx.eval(_)
    except Exception as e:
        print(f"Error during warm-up: {e}")
        print("The model might require specific input structures (e.g., attention_mask).")
        print("For simplicity, this example attempts a direct forward pass.")

    print("Warm-up complete.")

    trace_file_path = "traces/mlx_capture.gputrace"
    metal_trace_directory = os.path.dirname(trace_file_path)

    print(f"\nEnabling MLX Metal GPU trace capture. Trace will be saved as: {trace_file_path}")
    try:
        if metal_trace_directory:
            os.makedirs(metal_trace_directory, exist_ok=True)
        mx.metal.start_capture(trace_file_path)

        print(f"Running model inference for profiling with input shape {input_ids.shape}...")
        start_time = time.perf_counter()
        
        try:
            outputs = model(input_ids)
            mx.eval(outputs)
        except Exception as e:
            print(f"Error during profiled model execution: {e}")
            print("This model might require more specific inputs for a direct forward pass.")
            print("Profiling will capture events up to this point if any occurred.")
            outputs = None

        end_time = time.perf_counter()
        print(f"Model execution (wall time): {end_time - start_time:.4f} seconds")

        mx.metal.stop_capture()
        print(f"MLX Metal GPU trace capture disabled. Trace saved as {trace_file_path}")
        print("This trace can be opened with Xcode Instruments.")

    except AttributeError:
        print("\nmlx.metal.start_capture or mx.metal.stop_capture not found.")
        print("This Metal-specific GPU tracing may require a newer MLX version or may not be available on your system (e.g., non-Apple Silicon).")
        print("The general MLX profiler (mx.profiler_start/stop) might still be an option if needed.\n")
        if 'outputs' not in locals(): outputs = None 
    except Exception as e:
        print(f"An error occurred during Metal GPU trace capture: {e}")
        print("Ensure you are running on Apple Silicon with Metal.")
        print("IMPORTANT: To enable Metal GPU trace capture, you MUST run your script with the environment variable MTL_CAPTURE_ENABLED=1.")
        print("If issues persist, consider if MLX was built from source with MLX_METAL_DEBUG=ON for more detailed trace labels (see MLX documentation).")
        try:
            mx.metal.stop_capture()
            print("Attempted to stop Metal capture due to an error.")
        except Exception as e_stop:
            print(f"Note: Failed to explicitly stop Metal capture after initial error (this might be expected): {e_stop}")
        if 'outputs' not in locals(): outputs = None

    print("\n--- MLX Memory Info (if applicable) ---")
    try:
        print(f"  Active memory: {mx.get_active_memory() / (1024*1024):.2f} MB")
        print(f"  Peak memory: {mx.get_peak_memory() / (1024*1024):.2f} MB")
        print(f"  Cache memory (device buffer cache): {mx.get_cache_memory() / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"  Could not retrieve Metal memory info: {e}")
        print("  (This is expected if not running on Apple Silicon with Metal, or if MLX version differs)")

    print("\nProfiling finished.")
    if outputs is not None:
        print(f"Sample output logits shape: {outputs.shape}")
