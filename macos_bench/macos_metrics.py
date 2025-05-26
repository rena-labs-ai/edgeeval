"""macOS-specific performance monitoring utilities."""

import subprocess
import time
import json
from typing import Dict, Any
import psutil
import numpy as np

class MacOSMetricsCollector:
    """Collector for macOS-specific metrics."""
    
    def __init__(self, server_pid):
        """Initialize the collector.
        
        Args:
            server_pid: PID of the main server process to monitor
        """
        self.server_pid = server_pid
        self.server_process = None
        
        # Lists to track historical values for deviation calculation
        self.cpu_readings = []
        self.gpu_readings = []
        self.gpu_freq_readings = []
        self.rss_readings = []
        self.vms_readings = []
        
        # Peak values
        self.cpu_peak = 0.0
        self.gpu_peak = 0.0
        self.gpu_freq_peak = 0.0
        self.rss_peak = 0.0
        self.vms_peak = 0.0
        
        if server_pid:
            try:
                self.server_process = psutil.Process(server_pid)
                # Initialize CPU usage with a small delay to get accurate first reading
                self.server_process.cpu_percent()
                time.sleep(0.1)  # Short delay for initial CPU reading
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"Warning: Could not initialize metrics for process {server_pid}: {e}")
    
    def collect_metrics(self):
        """Collect all metrics."""
        if not self.server_process:
            return self._get_empty_metrics()
        
        try:
            # Get CPU metrics for the specific process
            cpu_metrics = self._get_cpu_metrics()
            self.cpu_readings.append(cpu_metrics['process_cpu'])
            self.cpu_peak = max(self.cpu_peak, cpu_metrics['process_cpu'])
            
            # Get memory metrics (unified memory for CPU/GPU)
            rss_mb = cpu_metrics['rss_mb']
            vms_mb = cpu_metrics['vms_mb']
            self.rss_readings.append(rss_mb)
            self.vms_readings.append(vms_mb)
            self.rss_peak = max(self.rss_peak, rss_mb)
            self.vms_peak = max(self.vms_peak, vms_mb)
            
            # Get GPU metrics
            gpu_metrics = self._get_gpu_metrics()
            if gpu_metrics['active_pct'] > 0:  # Only append non-zero readings
                self.gpu_readings.append(gpu_metrics['active_pct'])
                self.gpu_peak = max(self.gpu_peak, gpu_metrics['active_pct'])
            
            # Track GPU frequency
            if gpu_metrics['frequency'] > 0:
                self.gpu_freq_readings.append(gpu_metrics['frequency'])
                self.gpu_freq_peak = max(self.gpu_freq_peak, gpu_metrics['frequency'])
            
            # Calculate standard deviations and averages safely
            gpu_stddev = 0.0
            gpu_avg = 0.0
            if len(self.gpu_readings) > 0:
                gpu_avg = float(np.mean(self.gpu_readings))
                if len(self.gpu_readings) > 1:
                    gpu_stddev = float(np.std(self.gpu_readings))
            
            cpu_stddev = 0.0
            cpu_avg = 0.0
            if len(self.cpu_readings) > 0:
                cpu_avg = float(np.mean(self.cpu_readings))
                if len(self.cpu_readings) > 1:
                    cpu_stddev = float(np.std(self.cpu_readings))
            
            gpu_freq_stddev = 0.0
            gpu_freq_avg = 0.0
            if len(self.gpu_freq_readings) > 0:
                gpu_freq_avg = float(np.mean(self.gpu_freq_readings))
                if len(self.gpu_freq_readings) > 1:
                    gpu_freq_stddev = float(np.std(self.gpu_freq_readings))
            
            # Calculate memory statistics
            rss_stddev = 0.0
            rss_avg = 0.0
            if len(self.rss_readings) > 0:
                rss_avg = float(np.mean(self.rss_readings))
                if len(self.rss_readings) > 1:
                    rss_stddev = float(np.std(self.rss_readings))
            
            vms_stddev = 0.0
            vms_avg = 0.0
            if len(self.vms_readings) > 0:
                vms_avg = float(np.mean(self.vms_readings))
                if len(self.vms_readings) > 1:
                    vms_stddev = float(np.std(self.vms_readings))
            
            metrics = {
                # CPU metrics
                "cpu_usage_avg": cpu_avg,
                "cpu_usage_peak": self.cpu_peak,
                "cpu_usage_stddev": cpu_stddev,
                
                # Memory metrics (RSS - Resident Set Size)
                "rss_avg_mb": rss_avg,
                "rss_peak_mb": self.rss_peak,
                "rss_stddev_mb": rss_stddev,
                
                # Memory metrics (VMS - Virtual Memory Size)
                "vms_avg_mb": vms_avg,
                "vms_peak_mb": self.vms_peak,
                "vms_stddev_mb": vms_stddev,
                
                # GPU metrics
                "gpu_usage_avg": gpu_avg,
                "gpu_usage_peak": self.gpu_peak,
                "gpu_usage_stddev": gpu_stddev,
                
                # GPU frequency metrics
                "gpu_freq_peak_mhz": self.gpu_freq_peak,
                "gpu_freq_avg_mhz": gpu_freq_avg,
                "gpu_freq_stddev_mhz": gpu_freq_stddev
            }
            
            # print("[Debug] Current metrics:", json.dumps(metrics, indent=2))
            return metrics
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            # print(f"Warning: Could not collect metrics for process {self.server_pid}: {e}")
            return self._get_empty_metrics()
    
    def _get_empty_metrics(self):
        """Return empty metrics structure with zeros."""
        return {
            "cpu_usage_avg": 0.0,
            "cpu_usage_peak": 0.0,
            "cpu_usage_stddev": 0.0,
            "rss_avg_mb": 0.0,
            "rss_peak_mb": 0.0,
            "rss_stddev_mb": 0.0,
            "vms_avg_mb": 0.0,
            "vms_peak_mb": 0.0,
            "vms_stddev_mb": 0.0,
            "gpu_usage_avg": 0.0,
            "gpu_usage_peak": 0.0,
            "gpu_usage_stddev": 0.0,
            "gpu_freq_peak_mhz": 0.0,
            "gpu_freq_avg_mhz": 0.0,
            "gpu_freq_stddev_mhz": 0.0
        }
    
    def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU metrics for the specific process."""
        try:
            # Get process-specific CPU usage and memory
            process_cpu = self.server_process.cpu_percent()
            memory_info = self.server_process.memory_info()
            rss_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            vms_mb = memory_info.vms / (1024 * 1024)  # Convert to MB
            
            # print(f"[Debug] Process CPU usage: {process_cpu}%")
            # print(f"[Debug] Process RSS memory: {rss_mb:.2f} MB")
            # print(f"[Debug] Process Virtual memory: {vms_mb:.2f} MB")
            
            return {
                'process_cpu': process_cpu,
                'rss_mb': rss_mb,
                'vms_mb': vms_mb
            }
        except Exception as e:
            print(f"Error getting CPU metrics: {e}")
            return {
                'process_cpu': 0.0,
                'rss_mb': 0.0,
                'vms_mb': 0.0
            }
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics."""
        try:
            cmd = ["sudo", "powermetrics", "-s", "gpu_power", "-i", "100", "-n", "1"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            # print("[Debug] GPU powermetrics result:", result.stdout)
            
            metrics = {
                'active_pct': 0.0,
                'frequency': 0.0
            }
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'GPU idle residency' in line:
                        try:
                            idle_pct = float(line.split(':')[1].strip().replace('%', ''))
                            metrics['active_pct'] = 100.0 - idle_pct
                        except (ValueError, IndexError):
                            pass
                    elif 'GPU HW active frequency' in line:
                        try:
                            metrics['frequency'] = float(line.split(':')[1].strip().split()[0])
                        except (ValueError, IndexError):
                            pass
            
            # print(f"[Debug] Parsed GPU metrics:", json.dumps(metrics, indent=2))
            return metrics
            
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")
            return {
                'active_pct': 0.0,
                'frequency': 0.0
            }
    
    def clear_history(self):
        """Clear the metrics history."""
        if self.server_process:
            try:
                # Reset CPU usage tracking
                self.server_process.cpu_percent()
                time.sleep(0.1)  # Short delay for accurate reset
                
                # Clear historical data
                self.cpu_readings = []
                self.gpu_readings = []
                self.gpu_freq_readings = []
                self.rss_readings = []
                self.vms_readings = []
                
                # Reset peaks
                self.cpu_peak = 0.0
                self.gpu_peak = 0.0
                self.gpu_freq_peak = 0.0
                self.rss_peak = 0.0
                self.vms_peak = 0.0
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass