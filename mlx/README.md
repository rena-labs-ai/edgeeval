# MLX GPU Roofline Analysis Tools

This directory contains tools for analyzing MLX GPU performance using roofline model analysis with real GPU kernels extracted from MLX traces.

## What This Does

Extracts actual GPU kernels that were executed during MLX inference and analyzes their arithmetic intensity and performance characteristics using the roofline model.

## Key Files

### Core Tools
- **real_kernel_extractor.py** - Main tool that extracts real GPU kernels from .gputrace files
- **plot_roofline.py** - Creates roofline visualizations from extracted kernel data  
- **mlx_profiler.py** - Generates MLX GPU traces for analysis

### Generated Data
- **traces/mlx_capture.gputrace** - Raw MLX GPU trace (2.3GB)
- **traces/real_kernels.csv** - Extracted kernel performance data
- **traces/real_kernels_report.txt** - Detailed analysis report

## Quick Start

### 1. Generate GPU Trace

```bash
python3 mlx/mlx_profiler.py
```

### 2. Extract Kernels From Trace

```bash
python3 mlx/gpu_trace_kernels.py
```

### 3. Plot Roofline 

```bash
python3 mlx/plot_roofline.py
```