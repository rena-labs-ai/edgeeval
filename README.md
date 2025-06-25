# Agentic Workflow Profiling

A comprehensive profiling toolkit for analyzing performance metrics and resource utilization in agentic AI workflows across different inference backends and hardware platforms.

## Features

- **Multi-backend Support**: Profile workflows running on Ollama, vLLM, and other inference backends
- **Cross-platform Monitoring**: Automated system-level metrics collection for Apple Silicon and NVIDIA GPU platforms
- **Real-time Data Collection**: Background monitoring of CPU, memory, GPU utilization, and inference latencies
- **Automated Visualization**: Post-processing scripts that generate insightful performance plots and reports
- **Seamless Integration**: Easy integration with Rena Runtime through simple configuration flags

## Repository Structure

```
profile_generation/           # Runtime profiling data collection
├── inference_backend/        # Backend-specific profiling modules
│   ├── Ollama/              # Ollama inference backend profiling
│   └── vLLM/                # vLLM inference backend profiling
└── system_profiling/         # System-level metrics monitoring
    ├── Apple_Silicon/        # macOS/Apple Silicon monitoring scripts
    └── NVIDIA/               # NVIDIA GPU monitoring utilities

profile_processing/           # Data analysis and visualization
├── data_processors/          # Raw data processing utilities
├── visualization/            # Plot generation and reporting tools
└── templates/                # Report templates and configurations
```

## Integration with Rena Runtime

**Enable Profiling**: Add the `--profile` flag when running Rena Runtime

### How It Works

1. **Automatic Platform Detection**: The system automatically detects whether you're running on Apple Silicon or NVIDIA hardware
2. **Backend Configuration**: Based on your `config.json` in rena runtime, the profiler selects the appropriate backend module from `profile_generation/inference_backend/`
3. **Background Monitoring**: System-level monitoring scripts start automatically, tracking:
   - CPU and memory usage
   - GPU utilization
   - Power consumption
4. **Data Collection**: All metrics are collected in structured formats (CSV/JSON) for easy analysis
5. **Post-processing**: Automated generation of performance visualizations and summary reports
