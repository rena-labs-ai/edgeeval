#!/usr/bin/env python3
"""
Roofline Plotter for GPU Trace Data

Reads CSV output from gpu_trace_kernels.py and creates roofline plots.
Handles matplotlib dependency issues gracefully with ASCII fallback.
"""

import os
import csv
import sys
import math
from pathlib import Path
import random

def check_matplotlib():
    """Check if matplotlib is available and working"""
    try:
        # Set backend before importing
        os.environ['MPLBACKEND'] = 'Agg'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        return True, plt, np
    except Exception as e:
        print(f"⚠️  Matplotlib not available: {e}")
        return False, None, None

def read_trace_data(csv_path="mlx/traces/gpu_trace_kernels.csv"):
    """Read the CSV data from kernel analysis"""
    if not Path(csv_path).exists():
        print(f"❌ CSV file not found: {csv_path}")
        print("💡 Run gpu_trace_kernels.py first to generate the data")
        return None
    
    operations = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                operations.append({
                    "name": row["Kernel"],
                    "type": row["Type"], 
                    "precision": row["Precision"],
                    "arithmetic_intensity": float(row["Arithmetic_Intensity"]),
                    "performance": float(row["Performance_GFLOPS"]) * 1e9  # Convert to FLOPS/sec
                })
        
        print(f"📊 Loaded {len(operations)} kernel operations from {csv_path}")
        
        # Group by unique kernel names and get representative values
        unique_kernels = {}
        for op in operations:
            kernel_name = op["name"]
            if kernel_name not in unique_kernels:
                unique_kernels[kernel_name] = {
                    "name": kernel_name,
                    "type": op["type"],
                    "precision": op["precision"],
                    "arithmetic_intensity": op["arithmetic_intensity"],
                    "performance": op["performance"],
                    "count": 1
                }
            else:
                # Take average performance for duplicate kernels
                existing = unique_kernels[kernel_name]
                existing["count"] += 1
                existing["performance"] = (existing["performance"] + op["performance"]) / 2
        
        # Convert back to list and sort by arithmetic intensity (most interesting first)
        unique_ops = list(unique_kernels.values())
        unique_ops.sort(key=lambda x: x["arithmetic_intensity"], reverse=True)
        
        # Select top 10 most diverse/interesting kernels
        selected_ops = select_diverse_kernels(unique_ops, limit=10)
        
        print(f"📊 Selected {len(selected_ops)} unique kernels from {len(unique_ops)} total unique kernels")
        for op in selected_ops:
            print(f"  {op['name']} ({op['type']}) - {op['count']} instances")
        
        return selected_ops
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None

def select_diverse_kernels(kernels, limit=10):
    """Select a diverse set of kernels for visualization"""
    if len(kernels) <= limit:
        return kernels
    
    selected = []
    type_counts = {}
    
    # Priority order: get the best examples of each type
    type_priority = ['matrix_multiply', 'convolution', 'activation', 'normalization', 'reduction', 'elementwise']
    
    # First pass: get the best kernel of each type
    for kernel_type in type_priority:
        for kernel in kernels:
            if kernel['type'] == kernel_type and kernel_type not in type_counts:
                selected.append(kernel)
                type_counts[kernel_type] = 1
                break
        if len(selected) >= limit:
            break
    
    # Second pass: fill remaining slots with highest arithmetic intensity kernels
    remaining_slots = limit - len(selected)
    if remaining_slots > 0:
        # Get kernels not already selected
        selected_names = {k['name'] for k in selected}
        remaining_kernels = [k for k in kernels if k['name'] not in selected_names]
        
        # Add the most interesting remaining kernels
        for kernel in remaining_kernels[:remaining_slots]:
            selected.append(kernel)
    
    return selected

def create_ascii_roofline(operations):
    """Create ASCII-based roofline visualization"""
    print("\n📈 ASCII ROOFLINE VISUALIZATION")
    print("=" * 60)
    
    # M1 GPU specs
    bandwidth = 68.25e9  # bytes/sec
    peak_flops = 2.6e12  # FLOPS/sec
    
    # Create ASCII plot dimensions
    width = 60
    height = 20
    
    # Log scale ranges - adjust based on actual data
    ai_values = [op["arithmetic_intensity"] for op in operations]
    perf_values = [op["performance"] for op in operations]
    
    ai_min = min(0.1, min(ai_values) * 0.5)
    ai_max = max(10000, max(ai_values) * 2)
    perf_min = max(1e3, min([p for p in perf_values if p > 0]) * 0.1)  # Use actual minimum > 0
    perf_max = min(peak_flops * 2, max(perf_values) * 5)  # Give more room at top
    
    def log_scale(value, min_val, max_val, size):
        if value <= 0:
            return 0
        log_val = math.log10(value)
        log_min = math.log10(min_val)
        log_max = math.log10(max_val)
        return int((log_val - log_min) / (log_max - log_min) * size)
    
    # Initialize plot grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Draw roofline
    for x in range(width):
        ai = ai_min * (ai_max / ai_min) ** (x / width)
        
        # Memory bound line
        mem_perf = bandwidth * ai
        mem_y = height - 1 - log_scale(mem_perf, perf_min, perf_max, height)
        if 0 <= mem_y < height:
            grid[mem_y][x] = '-'
        
        # Compute bound line  
        comp_perf = peak_flops
        comp_y = height - 1 - log_scale(comp_perf, perf_min, perf_max, height)
        if 0 <= comp_y < height:
            grid[comp_y][x] = '-'
        
        # Roofline (minimum of both)
        roofline_perf = min(mem_perf, comp_perf)
        roof_y = height - 1 - log_scale(roofline_perf, perf_min, perf_max, height)
        if 0 <= roof_y < height:
            grid[roof_y][x] = '█'
    
    # Plot operations
    op_chars = ['●', '◆', '▲', '■', '♦', '▼', '★', '◇', '☯', '⬢']
    type_chars = {'matrix_multiply': '●', 'elementwise': '◆', 'reduction': '▲', 'convolution': '♦', 'activation': '⬢', 'normalization': '▼'}
    
    for i, op in enumerate(operations):
        ai = op["arithmetic_intensity"]
        perf = op["performance"]
        
        # No jitter needed since we're only showing unique kernels
        x = log_scale(ai, ai_min, ai_max, width)
        y = height - 1 - log_scale(perf, perf_min, perf_max, height)
        
        if 0 <= x < width and 0 <= y < height:
            # Use type-specific character if available, otherwise use index-based
            char = type_chars.get(op["type"], op_chars[i % len(op_chars)])
            grid[y][x] = char
    
    # Print the grid
    print(f"Performance (FLOPS/sec) vs Arithmetic Intensity (FLOPS/byte)")
    print(f"Peak: {peak_flops/1e12:.1f} TFLOPS | Bandwidth: {bandwidth/1e9:.1f} GB/s")
    print()
    
    # Y-axis labels
    for y in range(height):
        perf_val = perf_max * (perf_min / perf_max) ** (y / height)
        if y % 3 == 0:  # Show every 3rd label for better readability
            if perf_val >= 1e9:
                label = f"{perf_val/1e9:>6.1f}G"
            elif perf_val >= 1e6:
                label = f"{perf_val/1e6:>6.1f}M"
            else:
                label = f"{perf_val/1e3:>6.1f}K"
        else:
            label = "       "
        print(f"{label} |{''.join(grid[y])}")
    
    # X-axis
    print("       " + "+" + "-" * (width-2) + "+")
    x_labels = []
    for i in range(0, width, 10):
        ai_val = ai_min * (ai_max / ai_min) ** (i / width)
        x_labels.append(f"{ai_val:>8.1f}")
    print("        " + "".join(x_labels))
    print("        Arithmetic Intensity (FLOPS/byte)")
    
    # Legend
    print(f"\n📋 LEGEND:")
    print(f"█ = Roofline  - = Limits")
    for i, op in enumerate(operations):
        char = type_chars.get(op["type"], op_chars[i % len(op_chars)])
        print(f"{char} = {op['name']} ({op['type']})")

def create_roofline_plot(operations, output_path="mlx/traces/roofline_plot.png"):
    """Create roofline plot with the trace data"""
    
    has_matplotlib, plt, np = check_matplotlib()
    if not has_matplotlib:
        print("❌ Cannot create plot without matplotlib")
        print("💡 Try: pip install matplotlib")
        print("📊 Falling back to ASCII visualization...")
        create_ascii_roofline(operations)
        return False
    
    print("📈 Creating roofline plot...")
    
    # M1 GPU specs
    bandwidth = 68.25e9  # bytes/sec
    peak_flops = 2.6e12  # FLOPS/sec
    
    # Extract data for plotting
    ai_values = [op["arithmetic_intensity"] for op in operations]
    perf_values = [op["performance"] for op in operations]
    names = [op["name"] for op in operations]
    types = [op["type"] for op in operations]
    
    # Create theoretical roofline
    ai_range = np.logspace(-2, 4, 1000)
    memory_bound = bandwidth * ai_range
    compute_bound = np.full_like(ai_range, peak_flops)
    roofline = np.minimum(memory_bound, compute_bound)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot roofline limits
    ax.loglog(ai_range, memory_bound, '--', linewidth=2, 
             label=f'Memory Bound ({bandwidth/1e9:.1f} GB/s)', color='red', alpha=0.8)
    ax.loglog(ai_range, compute_bound, '--', linewidth=2,
             label=f'Compute Bound ({peak_flops/1e12:.1f} TFLOPS)', color='blue', alpha=0.8)
    ax.loglog(ai_range, roofline, '-', linewidth=3, 
             label='Roofline', color='black')
    
    # Color mapping for operation types
    type_colors = {
        'matrix_multiply': '#2E8B57',  # Sea green
        'elementwise': '#FF8C00',     # Dark orange
        'reduction': '#8A2BE2',       # Blue violet
        'unknown': '#696969'          # Dim gray
    }
    
    # Plot operations with jitter for overlapping points
    for ai, perf, name, op_type in zip(ai_values, perf_values, names, types):
        color = type_colors.get(op_type, '#696969')
        
        # Add small jitter to avoid perfect overlaps (only for visualization)
        ai_jitter = ai * (0.95 + random.random() * 0.1)  # ±5% jitter
        perf_jitter = perf * (0.95 + random.random() * 0.1)  # ±5% jitter
        
        # Use different marker sizes for different operation types
        if op_type == 'matrix_multiply':
            marker_size = 12
            marker = 'o'
        elif op_type == 'elementwise':
            marker_size = 10
            marker = 's'  # Square
        elif op_type == 'reduction':
            marker_size = 8
            marker = '^'  # Triangle
        else:
            marker_size = 10
            marker = 'o'
        
        ax.loglog(ai_jitter, perf_jitter, marker, markersize=marker_size, color=color, 
                 alpha=0.8, markeredgecolor='black', markeredgewidth=1)
        
        # Add labels with better positioning
        offset_x = 15 if ai > 10 else 10
        offset_y = 10
        ax.annotate(name, (ai_jitter, perf_jitter), xytext=(offset_x, offset_y), 
                   textcoords='offset points', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   ha='left')
    
    # Create custom legend
    legend_elements = []
    
    # Add roofline elements
    legend_elements.extend([
        plt.Line2D([0], [0], linestyle='-', color='black', linewidth=3, label='Roofline'),
        plt.Line2D([0], [0], linestyle='--', color='red', linewidth=2, label=f'Memory Bound ({bandwidth/1e9:.1f} GB/s)'),
        plt.Line2D([0], [0], linestyle='--', color='blue', linewidth=2, label=f'Compute Bound ({peak_flops/1e12:.1f} TFLOPS)')
    ])
    
    # Add actual kernel names instead of operation types
    plotted_kernels = set()
    for name, op_type in zip(names, types):
        if name not in plotted_kernels:
            color = type_colors.get(op_type, '#696969')
            
            # Use appropriate marker based on type
            if op_type == 'matrix_multiply':
                marker = 'o'
            elif op_type == 'elementwise':
                marker = 's'
            elif op_type == 'reduction':
                marker = '^'
            else:
                marker = 'o'
            
            legend_elements.append(
                plt.Line2D([0], [0], marker=marker, color='w', 
                          markerfacecolor=color, markersize=8, 
                          markeredgecolor='black', markeredgewidth=1,
                          label=name, linestyle='None')
            )
            plotted_kernels.add(name)
    
    ax.legend(handles=legend_elements, fontsize=9, loc='lower right', ncol=1)
    
    # Formatting
    ax.set_xlabel('Arithmetic Intensity (FLOPS/byte)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (FLOPS/sec)', fontsize=14, fontweight='bold')
    ax.set_title('MLX GPU trace Roofline Analysis', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid and styling
    ax.grid(True, which="both", alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, which="minor", alpha=0.1, linestyle='-', linewidth=0.3)
    
    # Set axis limits with some padding
    ax.set_xlim(0.01, 10000)
    ax.set_ylim(1e5, peak_flops * 3)
    
    # Add efficiency lines as background guides
    for efficiency in [0.1, 1, 10, 50]:  # 0.1%, 1%, 10%, 50% efficiency
        eff_line = roofline * (efficiency / 100)
        ax.loglog(ai_range, eff_line, ':', alpha=0.3, color='gray', linewidth=1)
        # Add efficiency labels
        if efficiency >= 1:
            ax.text(1000, eff_line[800], f'{efficiency}%', fontsize=8, alpha=0.6, color='gray')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Roofline plot saved: {output_path}")
    return True

def print_analysis_summary(operations):
    """Print a summary of the analysis"""
    print("\n📊 ROOFLINE ANALYSIS SUMMARY")
    print("=" * 50)
    
    # M1 specs for calculations
    bandwidth = 68.25e9
    peak_flops = 2.6e12
    
    total_ops = len(operations)
    memory_bound_ops = 0
    compute_bound_ops = 0
    
    print(f"{'Operation':<15} {'AI (FLOPS/byte)':<15} {'Perf (GFLOPS)':<15} {'Efficiency':<12} {'Bottleneck'}")
    print("-" * 75)
    
    for op in operations:
        ai = op["arithmetic_intensity"]
        perf = op["performance"]
        
        # Calculate bottleneck and efficiency
        memory_peak = bandwidth * ai
        theoretical_peak = min(memory_peak, peak_flops)
        efficiency = (perf / theoretical_peak) * 100 if theoretical_peak > 0 else 0
        
        is_memory_bound = memory_peak < peak_flops
        bottleneck = "Memory" if is_memory_bound else "Compute"
        
        if is_memory_bound:
            memory_bound_ops += 1
        else:
            compute_bound_ops += 1
        
        print(f"{op['name']:<15} {ai:<15.1f} {perf/1e9:<15.3f} {efficiency:<11.1f}% {bottleneck}")
    
    print("-" * 75)
    print(f"Total operations: {total_ops}")
    print(f"Memory-bound: {memory_bound_ops}")
    print(f"Compute-bound: {compute_bound_ops}")
    
    avg_efficiency = sum([
        (op["performance"] / min(bandwidth * op["arithmetic_intensity"], peak_flops)) * 100
        for op in operations if op["performance"] > 0
    ]) / len([op for op in operations if op["performance"] > 0])
    
    print(f"Average efficiency: {avg_efficiency:.2f}%")
    
    if avg_efficiency < 10:
        print(f"\n⚠️  LOW EFFICIENCY WARNING!")
        print("Consider optimizing memory access patterns and kernel utilization.")

def main():
    """Main function"""
    print("🚀 ROOFLINE PLOTTER")
    print("=" * 40)
    
    # Check for CSV file
    csv_path = "mlx/traces/gpu_trace_kernels.csv"
    operations = read_trace_data(csv_path)
    
    if not operations:
        return
    
    # Print summary
    print_analysis_summary(operations)
    
    # Create plot
    success = create_roofline_plot(operations)
    
    if success:
        print("\n🎉 SUCCESS!")
        print("📈 Roofline plot created from GPU trace data!")
        print("📁 Check mlx/traces/roofline_plot.png")
    else:
        print("\n💡 ASCII visualization shown above")
        print("📊 Install matplotlib for high-quality plots")
        print("📁 Data available in mlx/traces/gpu_trace_kernels.csv")

if __name__ == "__main__":
    main() 