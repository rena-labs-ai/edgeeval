# MacOS Inference Engine Benchmark

A benchmarking tool for evaluating inference engine performance on MacOS systems. This tool provides comprehensive metrics collection and analysis capabilities for measuring the performance of inference engines under various workloads.

## Features

- Synthetic and file-based dataset generation
- Concurrent request processing with rate limiting
- System metrics collection (CPU, memory, disk I/O, network I/O)
- Process-specific metrics monitoring
- Warmup phase support
- Detailed latency statistics
- JSON output format for results

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage with synthetic dataset:
```bash
python -m macos_bench \
    --input-length 100 \
    --output-length 100 \
    --num-requests 1000 \
    --concurrent-requests 4 \
    --num-warmup 10
```

Using a file-based dataset:
```bash
python -m macos_bench \
    --dataset-type file \
    --dataset-file path/to/dataset.json \
    --num-requests 1000 \
    --concurrent-requests 4
```

Monitor a specific process:
```bash
python -m macos_bench \
    --pid <process_id> \
    --num-requests 1000
```

### Command Line Arguments

#### Dataset Configuration
- `--input-length`: Average input length (default: 100)
- `--output-length`: Average output length (default: 100)
- `--input-length-std`: Input length standard deviation (default: 0.0)
- `--output-length-std`: Output length standard deviation (default: 0.0)
- `--num-requests`: Number of requests to benchmark (default: 100)
- `--dataset-type`: Type of dataset to use (choices: "synthetic", "file", default: "synthetic")
- `--dataset-file`: Path to dataset file (required for file dataset)

#### Benchmark Configuration
- `--num-warmup`: Number of warmup requests (default: 10)
- `--concurrent-requests`: Number of concurrent requests (default: 1)
- `--request-rate`: Request rate in requests per second
- `--pid`: Process ID to monitor
- `--seed`: Random seed for reproducibility

#### Output Configuration
- `--output`: Output file path (default: "benchmark_results.json")

## Output Format

The benchmark results are saved in JSON format with the following structure:

```json
{
  "config": {
    // Command line arguments used
  },
  "metrics": {
    // System metrics summary
    "cpu_percent": {
      "mean": 0.0,
      "std": 0.0,
      "min": 0.0,
      "max": 0.0,
      "p50": 0.0,
      "p95": 0.0,
      "p99": 0.0
    },
    // ... other metrics
  },
  "latency": {
    "mean": 0.0,
    "min": 0.0,
    "max": 0.0,
    "p50": 0.0,
    "p95": 0.0,
    "p99": 0.0
  },
  "total_requests": 0,
  "successful_requests": 0
}
```

## Dataset Format

When using a file-based dataset, the input file should be a JSON file with the following format:

```json
[
  {
    "input": "input text 1",
    "output": "output text 1"
  },
  {
    "input": "input text 2",
    "output": "output text 2"
  }
]
```
