#!/usr/bin/env python3
"""
MLX Execution Flow Analyzer

Analyzes GPU trace execution flow without external dependencies.
Maps operations back to source code context and identifies bottlenecks.
Outputs complete detailed analysis to text files.
"""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

def analyze_execution_flow():
    """Analyze execution flow from timeline data"""
    print("MLX EXECUTION FLOW ANALYSIS")
    print("=" * 60)
    
    timeline_file = Path("mlx/traces/execution_timeline.json")
    if not timeline_file.exists():
        print("ERROR: Timeline data not found. Run trace_flow_analyzer.py first.")
        return
    
    # Load timeline data
    with open(timeline_file, 'r') as f:
        timeline_data = json.load(f)
    
    events = timeline_data['events']
    print(f"Analyzing {len(events)} execution events...")
    print("")
    
    # Generate comprehensive report
    report_lines = generate_comprehensive_report(events, timeline_data)
    
    # Write detailed analysis to file
    output_file = "mlx/traces/complete_execution_analysis.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Complete analysis written to: {output_file}")
    
    # Generate and write pipeline graphic
    pipeline_lines = generate_pipeline_graphic(events, timeline_data)
    pipeline_file = "mlx/traces/execution_flow_graphic.txt"
    with open(pipeline_file, 'w') as f:
        f.write('\n'.join(pipeline_lines))
    
    print(f"Pipeline flow graphic written to: {pipeline_file}")
    
    # Also print summary to console
    print_summary(events)

def generate_comprehensive_report(events, timeline_data):
    """Generate complete comprehensive report with all details"""
    report = []
    
    # Header
    report.append("MLX GPU EXECUTION FLOW ANALYSIS")
    report.append("=" * 80)
    report.append(f"Total Events: {len(events)}")
    report.append(f"Total Duration: {timeline_data.get('total_duration_ms', 0):.2f}ms")
    report.append("")
    
    # Sort events by timestamp
    sorted_events = sorted(events, key=lambda x: x['timestamp_ms'])
    
    # 1. COMPLETE EXECUTION TIMELINE
    report.append("COMPLETE EXECUTION TIMELINE")
    report.append("-" * 80)
    
    for i, event in enumerate(sorted_events):
        relative_time = (event['timestamp_ms'] - sorted_events[0]['timestamp_ms'])
        duration_ms = event['duration_ms']
        source_context = event.get('source_context', 'Unknown')
        
        report.append(f"{i+1:4d}. {relative_time:8.1f}ms | {event['name']:<50} | {duration_ms:6.1f}ms | {source_context}")
    
    report.append("")
    
    # 2. EXECUTION SEQUENCE & SOURCE MAPPING  
    report.append("EXECUTION SEQUENCE & SOURCE MAPPING")
    report.append("-" * 80)
    
    operation_groups = group_related_operations(sorted_events)
    
    for i, group in enumerate(operation_groups):
        report.append(f"\n{i+1:2d}. Operation Group: {group['type']} ({group['count']} events)")
        report.append(f"    Time: {group['start_time']:.1f}ms - {group['end_time']:.1f}ms ({group['duration']:.1f}ms)")
        report.append(f"    Source Context: {group['source_context']}")
        report.append(f"    All Operations:")
        
        for j, op_name in enumerate(group['operation_names']):
            report.append(f"      {j+1:3d}. {op_name}")
    
    report.append("")
    
    # 3. EXECUTION PATTERNS & DEPENDENCIES
    report.append("EXECUTION PATTERNS & DEPENDENCIES")
    report.append("-" * 80)
    
    patterns = identify_execution_patterns(sorted_events)
    
    for pattern_name, pattern_info in patterns.items():
        report.append(f"\nPattern: {pattern_name}")
        report.append(f"  Frequency: {pattern_info['frequency']} times")
        report.append(f"  Average Duration: {pattern_info['avg_duration']:.1f}ms")
        report.append(f"  Example: {pattern_info['example']}")
        
        # Show all examples for this pattern
        report.append(f"  All Examples:")
        for j, example in enumerate(pattern_info.get('all_examples', [pattern_info['example']])):
            report.append(f"    {j+1:2d}. {example}")
    
    report.append("")
    
    # 4. PERFORMANCE BOTTLENECKS & OPTIMIZATION OPPORTUNITIES
    report.append("PERFORMANCE BOTTLENECKS & OPTIMIZATION OPPORTUNITIES")
    report.append("-" * 80)
    
    bottlenecks = identify_bottlenecks_with_context(sorted_events)
    
    for i, bottleneck in enumerate(bottlenecks):
        report.append(f"\n{i+1:2d}. {bottleneck['name']} ({bottleneck['duration']:.1f}ms)")
        report.append(f"    Likely Source: {bottleneck['source_context']}")
        report.append(f"    Optimization: {bottleneck['optimization_hint']}")
        report.append(f"    Impact: {bottleneck['impact']}")
    
    report.append("")
    
    # 5. CRITICAL PATH TIMELINE
    report.append("CRITICAL PATH TIMELINE")
    report.append("-" * 80)
    
    timeline_lines = create_critical_path_timeline_text(sorted_events)
    report.extend(timeline_lines)
    
    report.append("")
    
    # 6. COMPLETE SOURCE CODE MAPPING SUMMARY
    report.append("COMPLETE SOURCE CODE MAPPING SUMMARY")
    report.append("-" * 80)
    
    source_contexts = defaultdict(lambda: {'count': 0, 'total_duration': 0, 'operations': []})
    
    for event in sorted_events:
        context = event.get('source_context', 'Unknown')
        source_contexts[context]['count'] += 1
        source_contexts[context]['total_duration'] += event['duration_ms']
        source_contexts[context]['operations'].append(event['name'])
    
    # Sort by total duration
    sorted_contexts = sorted(source_contexts.items(), 
                           key=lambda x: x[1]['total_duration'], reverse=True)
    
    for context, data in sorted_contexts:
        avg_duration = data['total_duration'] / data['count']
        report.append(f"\nSource Context: {context}")
        report.append(f"  Operations: {data['count']} | Total: {data['total_duration']:.1f}ms | Average: {avg_duration:.1f}ms")
        report.append(f"  All Operations:")
        
        for j, operation in enumerate(data['operations']):
            report.append(f"    {j+1:3d}. {operation}")
    
    report.append("")
    
    # 7. OPERATION TYPE BREAKDOWN
    report.append("OPERATION TYPE BREAKDOWN")
    report.append("-" * 80)
    
    type_stats = {}
    for event in sorted_events:
        event_type = categorize_operation(event)
        if event_type not in type_stats:
            type_stats[event_type] = {'count': 0, 'total_duration': 0, 'operations': []}
        
        type_stats[event_type]['count'] += 1
        type_stats[event_type]['total_duration'] += event['duration_ms']
        type_stats[event_type]['operations'].append(event['name'])
    
    for op_type, stats in sorted(type_stats.items(), key=lambda x: x[1]['total_duration'], reverse=True):
        avg_duration = stats['total_duration'] / stats['count']
        report.append(f"\n{op_type}:")
        report.append(f"  Count: {stats['count']} operations")
        report.append(f"  Total Duration: {stats['total_duration']:.1f}ms")
        report.append(f"  Average Duration: {avg_duration:.1f}ms")
        report.append(f"  All Operations:")
        
        for j, operation in enumerate(stats['operations']):
            report.append(f"    {j+1:3d}. {operation}")
    
    report.append("")
    
    # 8. BOTTLENECKS IF AVAILABLE
    if 'bottlenecks' in timeline_data:
        report.append("IDENTIFIED BOTTLENECKS")
        report.append("-" * 80)
        for i, bottleneck in enumerate(timeline_data['bottlenecks']):
            report.append(f"{i+1:2d}. {bottleneck}")
    
    return report

def group_related_operations(events):
    """Group related operations together"""
    groups = []
    current_group = None
    
    for event in events:
        event_type = categorize_operation(event)
        
        # Start new group if type changes or significant time gap
        if (current_group is None or 
            current_group['type'] != event_type or
            event['timestamp_ms'] - current_group['end_time'] > 10):  # 10ms gap threshold
            
            if current_group:
                groups.append(current_group)
            
            current_group = {
                'type': event_type,
                'start_time': event['timestamp_ms'],
                'end_time': event['timestamp_ms'] + event['duration_ms'],
                'duration': event['duration_ms'],
                'count': 1,
                'operation_names': [event['name']],
                'source_context': event.get('source_context', 'Unknown')
            }
        else:
            # Add to current group
            current_group['end_time'] = max(current_group['end_time'], 
                                          event['timestamp_ms'] + event['duration_ms'])
            current_group['duration'] += event['duration_ms']
            current_group['count'] += 1
            current_group['operation_names'].append(event['name'])
    
    if current_group:
        groups.append(current_group)
    
    return groups

def categorize_operation(event):
    """Categorize operation type"""
    name = event['name'].lower()
    event_type = event['type']
    
    if event_type == 'buffer_allocation':
        return 'Memory Allocation'
    elif event_type == 'memory_transfer':
        return 'Memory Transfer'
    elif 'gemm' in name or 'matmul' in name:
        return 'Matrix Multiplication'
    elif 'conv' in name:
        return 'Convolution'
    elif 'reduce' in name:
        return 'Reduction'
    elif 'softmax' in name or 'gelu' in name:
        return 'Activation'
    elif 'norm' in name:
        return 'Normalization'
    else:
        return 'Other GPU Operation'

def identify_execution_patterns(events):
    """Identify common execution patterns"""
    patterns = defaultdict(lambda: {'frequency': 0, 'total_duration': 0, 'examples': []})
    
    # Look for sequential patterns
    for i in range(len(events) - 1):
        current = categorize_operation(events[i])
        next_op = categorize_operation(events[i + 1])
        
        pattern_name = f"{current} -> {next_op}"
        patterns[pattern_name]['frequency'] += 1
        patterns[pattern_name]['total_duration'] += events[i]['duration_ms'] + events[i + 1]['duration_ms']
        patterns[pattern_name]['examples'].append(f"{events[i]['name']} -> {events[i + 1]['name']}")
    
    # Calculate averages and return top patterns
    result = {}
    for pattern, data in patterns.items():
        if data['frequency'] >= 2:  # Only patterns that occur 2+ times
            result[pattern] = {
                'frequency': data['frequency'],
                'avg_duration': data['total_duration'] / data['frequency'],
                'example': data['examples'][0],
                'all_examples': data['examples']
            }
    
    # Sort by frequency
    return dict(sorted(result.items(), key=lambda x: x[1]['frequency'], reverse=True))

def identify_bottlenecks_with_context(events):
    """Identify bottlenecks with source code context"""
    # Sort by duration to find longest operations
    long_operations = sorted(events, key=lambda x: x['duration_ms'], reverse=True)
    
    bottlenecks = []
    for event in long_operations:
        operation_type = categorize_operation(event)
        
        # Determine optimization hints based on operation type
        if 'Matrix Multiplication' in operation_type:
            optimization = "Consider optimizing matrix sizes, using mixed precision, or kernel fusion"
        elif 'Convolution' in operation_type:
            optimization = "Try different kernel sizes, stride optimizations, or grouped convolutions"
        elif 'Memory' in operation_type:
            optimization = "Batch memory operations, use pinned memory, or optimize buffer reuse"
        elif 'Reduction' in operation_type:
            optimization = "Consider parallel reduction algorithms or different reduction patterns"
        else:
            optimization = "Profile individual operation characteristics and memory access patterns"
        
        # Calculate impact
        total_time = sum(e['duration_ms'] for e in events)
        impact_percent = (event['duration_ms'] / total_time) * 100
        
        bottlenecks.append({
            'name': event['name'],
            'duration': event['duration_ms'],
            'source_context': event.get('source_context', 'Unknown MLX operation'),
            'optimization_hint': optimization,
            'impact': f"{impact_percent:.1f}% of total execution time"
        })
    
    return bottlenecks

def create_critical_path_timeline_text(events):
    """Create ASCII timeline of critical path"""
    lines = []
    
    # Find the longest operations for critical path
    critical_events = sorted(events, key=lambda x: x['duration_ms'], reverse=True)[:20]
    critical_events.sort(key=lambda x: x['timestamp_ms'])
    
    if not critical_events:
        lines.append("No critical events found")
        return lines
    
    # Calculate timeline parameters
    start_time = critical_events[0]['timestamp_ms']
    end_time = max(e['timestamp_ms'] + e['duration_ms'] for e in critical_events)
    total_duration = end_time - start_time
    
    timeline_width = 60
    
    lines.append(f"Critical Path Duration: {total_duration:.2f}ms")
    lines.append("")
    lines.append("0ms" + " " * (timeline_width - 10) + f"{total_duration:.0f}ms")
    lines.append("|" + "-" * (timeline_width - 2) + "|")
    
    # Create timeline bars
    for event in critical_events:
        relative_start = (event['timestamp_ms'] - start_time) / total_duration
        relative_duration = event['duration_ms'] / total_duration
        
        start_pos = int(relative_start * (timeline_width - 2))
        duration_chars = max(1, int(relative_duration * (timeline_width - 2)))
        
        # Create bar
        bar = ['.'] * (timeline_width - 2)
        
        # Choose character based on operation type
        op_type = categorize_operation(event)
        if 'Matrix' in op_type:
            char = '#'
        elif 'Convolution' in op_type:
            char = '='
        elif 'Memory' in op_type:
            char = '-'
        else:
            char = '+'
        
        for j in range(start_pos, min(start_pos + duration_chars, timeline_width - 2)):
            bar[j] = char
        
        bar_str = ''.join(bar)
        
        # Event name and duration
        name = event['name']
        if len(name) > 40:
            name = name[:37] + "..."
        
        lines.append(f"|{bar_str}| {name} ({event['duration_ms']:.1f}ms)")
    
    lines.append("|" + "-" * (timeline_width - 2) + "|")
    lines.append("")
    lines.append("Legend: # Matrix Ops  = Convolution  - Memory  + Other")
    
    return lines

def print_summary(events):
    """Print summary to console"""
    print("\nSUMMARY")
    print("-" * 40)
    
    # Operation type counts
    type_counts = {}
    for event in events:
        event_type = categorize_operation(event)
        type_counts[event_type] = type_counts.get(event_type, 0) + 1
    
    for op_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{op_type}: {count} operations")
    
    # Top 5 longest operations
    long_ops = sorted(events, key=lambda x: x['duration_ms'], reverse=True)[:5]
    print(f"\nTop 5 Longest Operations:")
    for i, op in enumerate(long_ops, 1):
        name = op['name'][:50] + ("..." if len(op['name']) > 50 else "")
        print(f"{i}. {name} ({op['duration_ms']:.1f}ms)")

def generate_pipeline_graphic(events, timeline_data):
    """Generate visual pipeline graphic of execution flow"""
    lines = []
    
    # Header
    lines.append("MLX EXECUTION FLOW PIPELINE GRAPHIC")
    lines.append("=" * 100)
    lines.append(f"Total Events: {len(events)} | Duration: {timeline_data.get('total_duration_ms', 0):.2f}ms")
    lines.append("")
    
    # Sort events by timestamp
    sorted_events = sorted(events, key=lambda x: x['timestamp_ms'])
    
    # Create execution phases based on operation types
    phases = create_execution_phases(sorted_events)
    
    # Generate pipeline view
    lines.append("EXECUTION PIPELINE OVERVIEW")
    lines.append("-" * 100)
    lines.append("")
    
    # Show pipeline stages
    for i, phase in enumerate(phases):
        lines.append(f"PHASE {i+1}: {phase['name']} ({phase['duration']:.1f}ms)")
        lines.append("=" * 80)
        lines.append(f"Operations: {phase['count']} | Start: {phase['start_time']:.1f}ms | End: {phase['end_time']:.1f}ms")
        lines.append("")
        
        # Create visual pipeline for this phase
        pipeline_visual = create_phase_pipeline_visual(phase)
        lines.extend(pipeline_visual)
        lines.append("")
    
    # Overall pipeline flow
    lines.append("COMPLETE EXECUTION PIPELINE")
    lines.append("=" * 100)
    
    # Create overall pipeline visualization
    overall_pipeline = create_overall_pipeline_visual(phases)
    lines.extend(overall_pipeline)
    lines.append("")
    
    # Execution dependency graph
    lines.append("EXECUTION DEPENDENCY FLOW")
    lines.append("=" * 100)
    
    dependency_graph = create_dependency_flow_graphic(sorted_events)
    lines.extend(dependency_graph)
    lines.append("")
    
    # Performance pipeline
    lines.append("PERFORMANCE PIPELINE")
    lines.append("=" * 100)
    
    perf_pipeline = create_performance_pipeline(sorted_events)
    lines.extend(perf_pipeline)
    
    return lines

def create_execution_phases(events):
    """Create execution phases based on operation patterns"""
    phases = []
    current_phase = None
    phase_threshold = 50  # ms gap to create new phase
    
    for event in events:
        op_type = categorize_operation(event)
        
        # Start new phase if significant time gap or different operation type
        if (current_phase is None or 
            event['timestamp_ms'] - current_phase['end_time'] > phase_threshold or
            (current_phase['primary_type'] != op_type and current_phase['count'] > 10)):
            
            if current_phase:
                phases.append(current_phase)
            
            current_phase = {
                'name': f"{op_type} Phase",
                'primary_type': op_type,
                'start_time': event['timestamp_ms'],
                'end_time': event['timestamp_ms'] + event['duration_ms'],
                'duration': event['duration_ms'],
                'count': 1,
                'operations': [event],
                'types': {op_type: 1}
            }
        else:
            # Add to current phase
            current_phase['end_time'] = max(current_phase['end_time'], 
                                          event['timestamp_ms'] + event['duration_ms'])
            current_phase['duration'] += event['duration_ms']
            current_phase['count'] += 1
            current_phase['operations'].append(event)
            current_phase['types'][op_type] = current_phase['types'].get(op_type, 0) + 1
            
            # Update phase name if type distribution changes
            dominant_type = max(current_phase['types'].items(), key=lambda x: x[1])[0]
            if dominant_type != current_phase['primary_type']:
                current_phase['name'] = f"Mixed {dominant_type} Phase"
                current_phase['primary_type'] = dominant_type
    
    if current_phase:
        phases.append(current_phase)
    
    return phases

def create_phase_pipeline_visual(phase):
    """Create visual representation of a single phase"""
    lines = []
    
    # Phase header with operation breakdown
    lines.append(f"┌─ Phase Details " + "─" * 65 + "┐")
    lines.append(f"│ Operation Types:")
    
    for op_type, count in sorted(phase['types'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / phase['count']) * 100
        bar_length = int(percentage / 2)  # Scale down for display
        bar = "█" * bar_length + "░" * (50 - bar_length)
        lines.append(f"│   {op_type:<25} │{bar}│ {count:3d} ({percentage:4.1f}%)")
    
    lines.append(f"└" + "─" * 77 + "┘")
    lines.append("")
    
    # Timeline visualization for this phase
    if len(phase['operations']) > 1:
        lines.append("Timeline Flow:")
        timeline_visual = create_timeline_flow(phase['operations'][:20])  # Show first 20 ops
        lines.extend(timeline_visual)
        
        if len(phase['operations']) > 20:
            lines.append(f"... and {len(phase['operations']) - 20} more operations")
    
    return lines

def create_timeline_flow(operations):
    """Create timeline flow visualization"""
    lines = []
    
    if not operations:
        return lines
    
    # Calculate relative positions
    start_time = operations[0]['timestamp_ms']
    end_time = max(op['timestamp_ms'] + op['duration_ms'] for op in operations)
    total_duration = end_time - start_time
    
    if total_duration == 0:
        return ["All operations occur simultaneously"]
    
    timeline_width = 80
    
    # Create timeline header
    lines.append("Time: " + "0ms" + " " * (timeline_width - 15) + f"{total_duration:.1f}ms")
    lines.append("      " + "┌" + "─" * (timeline_width - 2) + "┐")
    
    # Show operations on timeline
    for i, op in enumerate(operations[:10]):  # Limit to prevent clutter
        relative_start = (op['timestamp_ms'] - start_time) / total_duration
        relative_duration = op['duration_ms'] / total_duration
        
        start_pos = int(relative_start * (timeline_width - 2))
        duration_chars = max(1, int(relative_duration * (timeline_width - 2)))
        
        # Create timeline bar
        bar = [' '] * (timeline_width - 2)
        
        # Fill operation duration
        op_char = get_operation_char(op)
        for j in range(start_pos, min(start_pos + duration_chars, timeline_width - 2)):
            bar[j] = op_char
        
        bar_str = ''.join(bar)
        op_name = op['name'][:30] + ("..." if len(op['name']) > 30 else "")
        
        lines.append(f"  {i+1:2d}. │{bar_str}│ {op_name} ({op['duration_ms']:.1f}ms)")
    
    lines.append("      " + "└" + "─" * (timeline_width - 2) + "┘")
    
    return lines

def create_overall_pipeline_visual(phases):
    """Create overall pipeline visualization across all phases"""
    lines = []
    
    if not phases:
        return ["No phases found"]
    
    # Calculate total timeline
    total_start = min(p['start_time'] for p in phases)
    total_end = max(p['end_time'] for p in phases)
    total_duration = total_end - total_start
    
    pipeline_width = 90
    
    # Header
    lines.append("Execution Flow Pipeline:")
    lines.append("Time: 0ms" + " " * (pipeline_width - 20) + f"{total_duration:.1f}ms")
    lines.append("┌" + "─" * (pipeline_width - 2) + "┐")
    
    # Show each phase
    for i, phase in enumerate(phases):
        if total_duration > 0:
            relative_start = (phase['start_time'] - total_start) / total_duration
            relative_duration = phase['duration'] / total_duration
            
            start_pos = int(relative_start * (pipeline_width - 2))
            duration_chars = max(1, int(relative_duration * (pipeline_width - 2)))
        else:
            start_pos = 0
            duration_chars = 1
        
        # Create phase bar
        bar = [' '] * (pipeline_width - 2)
        phase_char = get_phase_char(phase['primary_type'])
        
        for j in range(start_pos, min(start_pos + duration_chars, pipeline_width - 2)):
            bar[j] = phase_char
        
        bar_str = ''.join(bar)
        phase_name = phase['name'][:40] + ("..." if len(phase['name']) > 40 else "")
        
        lines.append(f"│{bar_str}│ Phase {i+1}: {phase_name}")
    
    lines.append("└" + "─" * (pipeline_width - 2) + "┘")
    lines.append("")
    
    # Legend
    lines.append("Legend:")
    lines.append("  █ Memory Operations    ▓ Matrix Multiplication    ▒ Convolution")
    lines.append("  ░ Reduction Operations ┼ Other GPU Operations    ─ Mixed Operations")
    
    return lines

def create_dependency_flow_graphic(events):
    """Create dependency flow graphic"""
    lines = []
    
    # Group events by type and show flow
    type_groups = defaultdict(list)
    for event in events:
        op_type = categorize_operation(event)
        type_groups[op_type].append(event)
    
    # Show operation flow
    lines.append("Operation Dependency Flow:")
    lines.append("")
    
    # Calculate flow between operation types
    flows = calculate_operation_flows(events)
    
    # Create flow diagram
    for flow_name, flow_data in flows.items():
        if flow_data['count'] >= 2:  # Only show significant flows
            arrow_thickness = min(flow_data['count'] // 10, 5)
            arrow = "=" * max(1, arrow_thickness) + ">"
            
            lines.append(f"{flow_name}")
            lines.append(f"  Frequency: {flow_data['count']} times")
            lines.append(f"  Average Duration: {flow_data['avg_duration']:.1f}ms")
            lines.append(f"  Flow: {flow_data['source']} {arrow} {flow_data['target']}")
            lines.append("")
    
    return lines

def create_performance_pipeline(events):
    """Create performance-focused pipeline view"""
    lines = []
    
    # Sort by performance impact
    long_ops = sorted(events, key=lambda x: x['duration_ms'], reverse=True)[:15]
    
    lines.append("Performance Impact Pipeline:")
    lines.append("")
    
    total_time = sum(e['duration_ms'] for e in events)
    max_duration = max(e['duration_ms'] for e in events) if events else 1
    
    for i, op in enumerate(long_ops):
        impact_percent = (op['duration_ms'] / total_time) * 100
        visual_length = int((op['duration_ms'] / max_duration) * 50)
        
        # Create performance bar
        perf_bar = "█" * visual_length + "░" * (50 - visual_length)
        
        op_name = op['name'][:35] + ("..." if len(op['name']) > 35 else "")
        op_type = categorize_operation(op)
        
        lines.append(f"{i+1:2d}. {op_name:<40}")
        lines.append(f"    │{perf_bar}│ {op['duration_ms']:6.1f}ms ({impact_percent:4.1f}%)")
        lines.append(f"    Type: {op_type}")
        lines.append("")
    
    return lines

def get_operation_char(operation):
    """Get character representation for operation type"""
    op_type = categorize_operation(operation)
    chars = {
        'Memory Allocation': '█',
        'Memory Transfer': '▓',
        'Matrix Multiplication': '▒',
        'Convolution': '░',
        'Reduction': '┼',
        'Other GPU Operation': '─'
    }
    return chars.get(op_type, '·')

def get_phase_char(op_type):
    """Get character representation for phase type"""
    chars = {
        'Memory Allocation': '█',
        'Memory Transfer': '▓', 
        'Matrix Multiplication': '▒',
        'Convolution': '░',
        'Reduction': '┼',
        'Other GPU Operation': '─'
    }
    return chars.get(op_type, '·')

def calculate_operation_flows(events):
    """Calculate flows between operation types"""
    flows = defaultdict(lambda: {'count': 0, 'total_duration': 0, 'source': '', 'target': ''})
    
    for i in range(len(events) - 1):
        current_type = categorize_operation(events[i])
        next_type = categorize_operation(events[i + 1])
        
        if current_type != next_type:  # Only count transitions
            flow_key = f"{current_type} -> {next_type}"
            flows[flow_key]['count'] += 1
            flows[flow_key]['total_duration'] += events[i]['duration_ms'] + events[i + 1]['duration_ms']
            flows[flow_key]['source'] = current_type
            flows[flow_key]['target'] = next_type
    
    # Calculate averages
    for flow_data in flows.values():
        if flow_data['count'] > 0:
            flow_data['avg_duration'] = flow_data['total_duration'] / flow_data['count']
    
    return dict(sorted(flows.items(), key=lambda x: x[1]['count'], reverse=True))

if __name__ == "__main__":
    analyze_execution_flow() 