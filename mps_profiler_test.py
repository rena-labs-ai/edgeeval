import torch
import torch.nn as nn
import torch.profiler
import os
import tempfile
import shutil
import time

class TestModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def run_kineto_profiler(model, dummy_input, device, base_log_dir="mps_profiler_logs"):
    model.to(device)
    dummy_input = dummy_input.to(device)

    kineto_log_dir = os.path.join(base_log_dir, "kineto_trace")
    if os.path.exists(kineto_log_dir):
        shutil.rmtree(kineto_log_dir)
    os.makedirs(kineto_log_dir, exist_ok=True)
    
    tensorboard_trace_path = os.path.join(kineto_log_dir, "tensorboard_trace")
    chrome_trace_path = os.path.join(kineto_log_dir, "mps_model_chrome_trace.json")

    print(f"\n### Running Kineto ###\n")
    print(f"TensorBoard saved at: {tensorboard_trace_path}")
    print(f"Chrome trace saved at: {chrome_trace_path}")

    print("Warming up...")
    for _ in range(3):
        _ = model(dummy_input)
    if device.type == 'mps':
        torch.mps.synchronize()

    with torch.profiler.profile(
        activities=None,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_trace_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i in range(4): 
            with torch.profiler.record_function("model_iteration"): 
                with torch.profiler.record_function("model_inference"): 
                    output = model(dummy_input)
                with torch.profiler.record_function("custom_operation_after_inference"):
                    _ = output.mean() * 2.0 
            prof.step() 

    print("\n### Kineto Results ###\n")
    sort_key = "self_mps_time_total" if "self_mps_time_total" in prof.key_averages()[0].key else "self_cpu_time_total"
    try:
        print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
    except Exception as e:
        print(f"Could not print table sorted by '{sort_key}', trying 'self_cpu_time_total': {e}")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

    print(f"\nChrome trace saved to: {chrome_trace_path}")
    print(f"To view, open Chrome and navigate to chrome://tracing, then load the file.")
    print(f"To view with TensorBoard: tensorboard --logdir {tensorboard_trace_path}")

def generate_instruments_trace(model, dummy_input, device):
    print("\n### Generating MPS trace for XCode Instruments ###\n")
    if device.type != 'mps':
        print("No metal device found.")
        return

    model.to(device)
    dummy_input = dummy_input.to(device)

    try:
        trace_file = os.path.join("mps_profiler_outputs", "mps_trace.json")
        os.makedirs(os.path.dirname(trace_file), exist_ok=True)
        
        print(f"Starting MPS System Profiler...")
        print(f"Trace will be saved to: {trace_file}")
        
        torch.mps.profiler.start(mode="interval,event")
        print("MPS System Profiler Started.")
        
        with torch.no_grad():
            for i in range(10):
                print(f"Running iteration {i+1}/10...")
                batch = torch.cat([dummy_input] * 2, dim=0)
                _ = model(batch)
                torch.mps.synchronize()
                time.sleep(0.1)

        torch.mps.synchronize()
        
        trace_data = torch.mps.profiler.stop()
        with open(trace_file, 'w') as f:
            f.write(trace_data)
            
        print("MPS System Profiler Stopped.")
        print(f"\nTrace saved to: {os.path.abspath(trace_file)}")
        print("\nTo view the trace:")
        print("1. Open Xcode")
        print("2. Go to Xcode → Open Developer Tool → Instruments")
        print("3. In Instruments, select 'Metal System Trace' template")
        print("4. Click 'Choose File' and select the trace file")

    except RuntimeError as e:
        print(f"Failed to run MPS System Profiler: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during Instruments trace generation: {e}")

if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        print("MPS not available on this system.")
        if not torch.backends.mps.is_built():
            print("PyTorch was not built with MPS support. Exiting.")
        else:
            print("MPS is built, but no MPS device found. Exiting.")
        exit()
    else:
        print("MPS is available.")
        device = torch.device("mps")

    print(f"Using device: {device}")

    input_size = 512
    hidden_size = 1024
    output_size = 256
    batch_size = 128

    model = TestModel(input_size, hidden_size, output_size)
    dummy_input = torch.randn(batch_size, input_size)
    
    base_log_dir = "mps_profiler_outputs"
    if os.path.exists(base_log_dir):
        shutil.rmtree(base_log_dir)
    os.makedirs(base_log_dir, exist_ok=True)
    print(f"All profiler outputs will be saved under ./{base_log_dir}/")

    print(f"\nRunning profiler tests with model: {model.__class__.__name__}")

    run_kineto_profiler(model, dummy_input, device, base_log_dir=base_log_dir)

    generate_instruments_trace(model, dummy_input, device)

    print("\nMPS profiler tests completed.")
