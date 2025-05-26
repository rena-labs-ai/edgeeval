"""MacOS LLM bench metrics collection"""

import psutil
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import threading

@dataclass
class SystemMetrics:
    """System metrics collected during benchmark."""
    # CPU metrics
    cpu_usage_avg: float = 0.0
    cpu_usage_peak: float = 0.0
    cpu_usage_stddev: float = 0.0
    
    # Memory metrics (RSS)
    rss_avg_mb: float = 0.0
    rss_peak_mb: float = 0.0
    rss_stddev_mb: float = 0.0
    
    # Memory metrics (Virtual)
    vms_avg_mb: float = 0.0
    vms_peak_mb: float = 0.0
    vms_stddev_mb: float = 0.0
    
    # GPU metrics (if available)
    gpu_usage_avg: Optional[float] = None
    gpu_usage_peak: Optional[float] = None
    gpu_usage_stddev: Optional[float] = None
    gpu_freq_peak_mhz: Optional[float] = None
    gpu_freq_avg_mhz: Optional[float] = None
    gpu_freq_stddev_mhz: Optional[float] = None

@dataclass
class ServerMetrics:
    """Server-side metrics for a request."""
    input_tokens: int = 0
    output_tokens: int = 0
    end_to_end_latency_s: float = 0.0
    time_to_first_token_s: float = 0.0
    system_metrics: Optional[SystemMetrics] = None

@dataclass
class Metrics:
    """Metrics for a request."""
    success: bool = True
    start_time: float = 0.0
    finish_time: float = 0.0
    end_to_end_latency_s: float = 0.0
    input_tokens: int = 0
    time_to_first_token_s: Optional[float] = None
    server_metrics: Optional[ServerMetrics] = None
    exec_feature: Optional[Dict[str, Any]] = None

class SystemMetricsCollector:
    """Collects system metrics in a background thread."""
    
    def __init__(self, pid: Optional[int] = None):
        self.pid = pid
        self.process = psutil.Process(pid) if pid else None
        self.stop_event = threading.Event()
        self.collection_thread = None
        self.metrics: Dict[str, List[float]] = {
            "cpu_percent": [],
            "memory_rss": [],
            "memory_vms": [],
        }
        
    def start(self):
        """Start collecting metrics in background."""
        self.collection_thread = threading.Thread(target=self._collect_metrics)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
    def stop(self):
        """Stop collecting metrics."""
        self.stop_event.set()
        if self.collection_thread:
            self.collection_thread.join()
            
    def _collect_metrics(self):
        """Collect metrics in a loop."""
        while not self.stop_event.is_set():
            try:
                if self.process:
                    # Process-specific metrics
                    self.metrics["cpu_percent"].append(self.process.cpu_percent())
                    mem_info = self.process.memory_info()
                    self.metrics["memory_rss"].append(mem_info.rss / (1024 * 1024))  # Convert to MB
                    self.metrics["memory_vms"].append(mem_info.vms / (1024 * 1024))  # Convert to MB
                else:
                    # System-wide metrics
                    self.metrics["cpu_percent"].append(psutil.cpu_percent())
                    mem_info = psutil.virtual_memory()
                    self.metrics["memory_rss"].append(mem_info.used / (1024 * 1024))
                    self.metrics["memory_vms"].append(mem_info.total / (1024 * 1024))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(0.1)  # Collect every 100ms
            
    def get_metrics(self) -> SystemMetrics:
        """Get collected metrics summary."""
        if not any(self.metrics.values()):
            return SystemMetrics()
            
        def compute_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"avg": 0.0, "peak": 0.0, "stddev": 0.0}
            return {
                "avg": sum(values) / len(values),
                "peak": max(values),
                "stddev": (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
            }
            
        cpu_stats = compute_stats(self.metrics["cpu_percent"])
        rss_stats = compute_stats(self.metrics["memory_rss"])
        vms_stats = compute_stats(self.metrics["memory_vms"])
        
        return SystemMetrics(
            cpu_usage_avg=cpu_stats["avg"],
            cpu_usage_peak=cpu_stats["peak"],
            cpu_usage_stddev=cpu_stats["stddev"],
            rss_avg_mb=rss_stats["avg"],
            rss_peak_mb=rss_stats["peak"],
            rss_stddev_mb=rss_stats["stddev"],
            vms_avg_mb=vms_stats["avg"],
            vms_peak_mb=vms_stats["peak"],
            vms_stddev_mb=vms_stats["stddev"]
        )

class MetricCollector:
    """Collects and analyzes metrics from request records."""
    
    def __init__(self):
        self.metrics = []
        self.system_collector = SystemMetricsCollector()
        
    def start_collection(self):
        """Start collecting system metrics."""
        self.system_collector.start()
        
    def stop_collection(self):
        """Stop collecting system metrics."""
        self.system_collector.stop()
        
    def add_metrics(self, metrics: Metrics):
        """Add metrics to the collection."""
        self.metrics.append(metrics)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the collected metrics."""
        if not self.metrics:
            return {}
            
        successful_requests = [m for m in self.metrics if m.success]
        if not successful_requests:
            return {
                "total_requests": len(self.metrics),
                "successful_requests": 0,
                "success_rate": 0.0
            }
            
        latencies = [m.end_to_end_latency_s for m in successful_requests]
        ttft = [m.time_to_first_token_s for m in successful_requests if m.time_to_first_token_s is not None]
        
        summary = {
            "total_requests": len(self.metrics),
            "successful_requests": len(successful_requests),
            "success_rate": len(successful_requests) / len(self.metrics),
            "latency": {
                "mean": sum(latencies) / len(latencies),
                "p50": sorted(latencies)[len(latencies) // 2],
                "p90": sorted(latencies)[int(len(latencies) * 0.9)],
                "p99": sorted(latencies)[int(len(latencies) * 0.99)]
            }
        }
        
        if ttft:
            summary["time_to_first_token"] = {
                "mean": sum(ttft) / len(ttft),
                "p50": sorted(ttft)[len(ttft) // 2],
                "p90": sorted(ttft)[int(len(ttft) * 0.9)],
                "p99": sorted(ttft)[int(len(ttft) * 0.99)]
            }
            
        # Add server metrics if available
        server_metrics = [m.server_metrics for m in successful_requests if m.server_metrics]
        if server_metrics:
            summary["server_metrics"] = {
                "input_tokens": sum(m.input_tokens for m in server_metrics) / len(server_metrics),
                "output_tokens": sum(m.output_tokens for m in server_metrics) / len(server_metrics),
                "tokens_per_second": sum(
                    (m.input_tokens + m.output_tokens) / m.end_to_end_latency_s 
                    for m in server_metrics
                ) / len(server_metrics)
            }
            
            # Add system metrics if available
            system_metrics = self.system_collector.get_metrics()
            summary["system_metrics"] = {
                "cpu": {
                    "usage_avg": system_metrics.cpu_usage_avg,
                    "usage_peak": system_metrics.cpu_usage_peak,
                    "usage_stddev": system_metrics.cpu_usage_stddev
                },
                "memory": {
                    "rss_avg_mb": system_metrics.rss_avg_mb,
                    "rss_peak_mb": system_metrics.rss_peak_mb,
                    "rss_stddev_mb": system_metrics.rss_stddev_mb,
                    "vms_avg_mb": system_metrics.vms_avg_mb,
                    "vms_peak_mb": system_metrics.vms_peak_mb,
                    "vms_stddev_mb": system_metrics.vms_stddev_mb
                }
            }
            
        return summary 