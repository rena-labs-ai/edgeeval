from __future__ import annotations

import json
import platform
import subprocess
import threading
import time
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel

import psutil

def get_process_group_processes(pgid: int) -> List[psutil.Process]:
    """Get all processes belonging to the process group."""
    processes = []
    try:
        for proc in psutil.process_iter():
            try:
                # Use os.getpgid() to get the process group ID
                if os.getpgid(proc.pid) == pgid:
                    processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                # Process might have disappeared or we don't have permission
                continue
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"Exception: {type(e).__name__}")
                continue
    except Exception:
        pass
    return processes


class Metrics(BaseModel):
    system_metrics: Optional[SystemMetrics] = None
    # TODO: Add other platform-specific metrics such as power, GPU, etc.

class SystemMetrics(BaseModel):
    cpu_usage_avg: float = 0.0
    cpu_usage_peak: float = 0.0
    cpu_usage_stddev: float = 0.0
    pid: Optional[int] = None
    process_count_avg: float = 0.0
    process_count_peak: int = 0
    # TODO: Add other general metrics such as memory

class BaseMetricsCollector(ABC):
    """Abstract base class implemented by every concrete metrics collector."""
    def __init__(self, pgid: int, interval_s: float = 0.1):
        self._pgid = pgid
        self._interval_s = interval_s

    @abstractmethod
    def collect(self) -> Any:
        """Collect metrics."""

    @abstractmethod
    def snapshot(self) -> Any:
        """Return a summary of all samples collected so far."""

    @abstractmethod
    def store_metrics(self, results_dir: str):
        """Store metrics in files."""

class CPUCollector(BaseMetricsCollector):
    def __init__(self, pgid: int, interval_s: float = 0.1):
        super().__init__(pgid, interval_s)
        self._cpu_samples: List[float] = []
        self._process_counts: List[int] = []
        self._individual_process_metrics: Dict[int, List[float]] = {}

    def snapshot(self) -> SystemMetrics:
        if not self._cpu_samples:  # No samples collected – return zeroes.
            return SystemMetrics(pid=self._pgid)

        cpu_avg = mean(self._cpu_samples)
        cpu_peak = max(self._cpu_samples)
        cpu_std = pstdev(self._cpu_samples) if len(self._cpu_samples) > 1 else 0.0
        
        proc_count_avg = mean(self._process_counts) if self._process_counts else 0.0
        proc_count_peak = max(self._process_counts) if self._process_counts else 0

        return SystemMetrics(
            cpu_usage_avg=cpu_avg,
            cpu_usage_peak=cpu_peak,
            cpu_usage_stddev=cpu_std,
            pid=self._pgid,
            process_count_avg=proc_count_avg,
            process_count_peak=proc_count_peak,
        )

    def store_metrics(self, results_dir: str):
        """Store metrics in files."""
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Store aggregated CPU metrics
        with open(os.path.join(results_dir, f"pgid_{self._pgid}_cpu_total.json"), "w") as f:
            json.dump(self._cpu_samples, f)
        
        # Store process counts over time
        with open(os.path.join(results_dir, f"pgid_{self._pgid}_process_counts.json"), "w") as f:
            json.dump(self._process_counts, f)
        
        # Store individual process metrics
        with open(os.path.join(results_dir, f"pgid_{self._pgid}_individual_processes.json"), "w") as f:
            json.dump(self._individual_process_metrics, f)

    def collect(self) -> Any:
        """Main sampling loop."""
        processes = get_process_group_processes(self._pgid)
        total_cpu = 0.0
        current_pids = set()
        
        for proc in processes:
            try:
                cpu_percent = proc.cpu_percent()
                total_cpu += cpu_percent
                pid = proc.pid
                current_pids.add(pid)
                
                # Track individual process metrics
                if pid not in self._individual_process_metrics:
                    self._individual_process_metrics[pid] = []
                self._individual_process_metrics[pid].append(cpu_percent)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logging.error(f"Exception: {type(e).__name__}")
                continue
        
        self._cpu_samples.append(total_cpu)
        self._process_counts.append(len(processes))                


class MetricsTask:
    """Coordinates multiple collectors and aggregates a single JSON summary."""

    def __init__(self, pid: Optional[int] = None, pgid: Optional[int] = None, results_dir: Optional[str] = None):
        self._collectors: List[BaseMetricsCollector] = self._select_collectors(pid, pgid)
        self._results_dir = results_dir
        self._stop_evt = threading.Event()
        self._collection_threads: List[threading.Thread] = []
    
    def start(self):
        for c in self._collectors:
            t = threading.Thread(target=self._run, args=(c,), daemon=True)
            self._collection_threads.append(t)
            t.start()

    def stop(self) -> None:
        self._stop_evt.set()
        for t in self._collection_threads:            
            t.join()

    def _run(self, collector: BaseMetricsCollector):
        while not self._stop_evt.is_set():
            collector.collect()
            time.sleep(collector._interval_s)

    def get_summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict with all aggregated numbers."""
        summary: Dict[str, Any] = {}

        for col in self._collectors:
            snap = col.snapshot()
            key = col.__class__.__name__
            summary[key] = snap.dict()  # Convert Pydantic model to dict

        for col in self._collectors:
            col.store_metrics(self._results_dir)

        return summary

    @staticmethod
    def _select_collectors(pid: Optional[int], pgid: Optional[int]) -> List[BaseMetricsCollector]:
        collectors: List[BaseMetricsCollector] = []
        
        if pgid is not None:
            collectors.append(CPUCollector(pgid=pgid, interval_s=0.1))
            
        return collectors

# if __name__ == "__main__":
#     pgid = 538786
#     results_dir = "results"
#     task = MetricsTask(pgid=pgid, results_dir=results_dir)
#     task.start()
#     time.sleep(10)
#     task.stop()
#     print(task.get_summary())
#     task.store_metrics()