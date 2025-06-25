import mlx.core as mx
from mlx_lm import load, generate

def load_model(model_path: str):
    """
    Loads a language model and its tokenizer from the given path.

    Args:
        model_path: The path to the MLX model (e.g., a Hugging Face model identifier
                    or a local path to the model directory).

    Returns:
        A tuple containing the loaded model and tokenizer.
        Returns (None, None) if loading fails.
    """
    try:
        model, tokenizer = load(model_path)
        print(f"Successfully loaded model and tokenizer from: {model_path}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None

if __name__ == "__main__":
    model_name = "mlx-community/Phi-3-mini-4k-instruct-8bit"

    print(f"Attempting to load model: {model_name}")
    model, tokenizer = load_model(model_name)

    if model and tokenizer:
        print("\nModel and tokenizer loaded successfully.")

        prompt = "Write a short story about a robot who dreams of becoming a chef."
        print(f"\nGenerating text for prompt: '{prompt}'")
        
        response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=100)
        
        print("\n--- Generated Text ---")
        print(response)
        print("----------------------")
        
        if hasattr(model, 'config'):
            print(f"\nModel config keys: {model.config.keys()}")
        
        print(f"\nTokenizer type: {type(tokenizer)}")
        if hasattr(tokenizer, 'vocab_size'):
             print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    else:
        print(f"Failed to load the model: {model_name}")
