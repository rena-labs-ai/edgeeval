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

class ServerMetrics(BaseModel):
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

    @abstractmethod
    def start(self) -> None:
        """Begin background sampling (non-blocking)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop sampling and join any internal worker threads."""

    @abstractmethod
    def snapshot(self) -> Any:
        """Return a summary of all samples collected so far."""


class SystemMetricsCollector(BaseMetricsCollector):
    def __init__(self, pgid: int, interval_s: float = 0.1):
        self._pgid = pgid
        self._interval_s = interval_s
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._cpu_samples: List[float] = []
        self._process_counts: List[int] = []
        self._individual_process_metrics: Dict[int, List[float]] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return  # already running
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join()

    def _get_process_group_processes(self) -> List[psutil.Process]:
        """Get all processes belonging to the process group."""
        processes = []
        try:
            for proc in psutil.process_iter():
                try:
                    # Use os.getpgid() to get the process group ID
                    if os.getpgid(proc.pid) == self._pgid:
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

    def _run(self):
        """Main sampling loop."""
        while not self._stop_evt.is_set():
            try:
                processes = self._get_process_group_processes()
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
                        print(f"Exception: {type(e).__name__}")
                        continue
                
                self._cpu_samples.append(total_cpu)
                self._process_counts.append(len(processes))
                
            except Exception as e:
                # Continue monitoring even if there are temporary errors
                pass

            time.sleep(self._interval_s)


class MetricsTask:
    """Coordinates multiple collectors and aggregates a single JSON summary."""

    def __init__(self, pid: Optional[int] = None, pgid: Optional[int] = None, results_dir: Optional[str] = None):
        self._collectors: List[BaseMetricsCollector] = self._select_collectors(pid, pgid)
        self._results_dir = results_dir
    
    def start(self):
        for c in self._collectors:
            c.start()

    def stop(self):
        for c in self._collectors:
            c.stop()

    def get_summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict with all aggregated numbers."""
        summary: Dict[str, Any] = {}

        for col in self._collectors:
            snap = col.snapshot()
            key = col.__class__.__name__
            summary[key] = snap.dict()  # Convert Pydantic model to dict

        return summary

    def store_metrics(self):
        if self._results_dir:
            for col in self._collectors:
                col.store_metrics(self._results_dir)

    @staticmethod
    def _select_collectors(pid: Optional[int], pgid: Optional[int]) -> List[BaseMetricsCollector]:
        collectors: List[BaseMetricsCollector] = []
        
        if pgid is not None:
            collectors.append(SystemMetricsCollector(pgid=pgid, interval_s=0.1))
            
        return collectors
