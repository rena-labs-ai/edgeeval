from __future__ import annotations

import json
import platform
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


import psutil

class ServerMetrics(BaseModel):
    input_tokens: int
    prefill_tokens: int
    output_tokens: int
    end_to_end_latency_s: float
    prefill_tokens_per_s: float
    inter_token_latency_s: float
    time_per_output_token_s: float
    time_to_first_token_s: Optional[float] = None
    system_metrics: Optional[SystemMetrics] = None


class SystemMetrics(BaseModel):
    cpu_usage_avg: float = 0.0
    cpu_usage_peak: float = 0.0
    cpu_usage_stddev: float = 0.0

    # Memory (resident)
    rss_avg_mb: float = 0.0
    rss_peak_mb: float = 0.0
    rss_stddev_mb: float = 0.0

    # Memory (virtual)
    vms_avg_mb: float = 0.0
    vms_peak_mb: float = 0.0
    vms_stddev_mb: float = 0.0

    # GPU (optional)
    gpu_usage_avg: Optional[float] = None
    gpu_usage_peak: Optional[float] = None
    gpu_usage_stddev: Optional[float] = None

    gpu_freq_avg_mhz: Optional[float] = None
    gpu_freq_peak_mhz: Optional[float] = None
    gpu_freq_stddev_mhz: Optional[float] = None
    
class Metrics(BaseModel):
    success: bool
    start_time: float
    finish_time: float
    end_to_end_latency_s: float

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    inter_token_latency_s: Optional[float] = None
    time_per_output_token_s: Optional[float] = None
    time_to_first_token_s: Optional[float] = None
    system_metrics: Optional[SystemMetrics] = None
    exec_feature: Optional[Dict[str, Any]] = None


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


class ProcessMetricsCollector(BaseMetricsCollector):
    def __init__(self, pid: Optional[int] = None, interval_s: float = 0.1):
        self._process = psutil.Process(pid) if pid else None
        self._interval_s = interval_s
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._cpu: List[float] = []
        self._rss: List[float] = []
        self._vms: List[float] = []

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return  # already running
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join()

    def snapshot(self) -> SystemMetrics:
        if not self._cpu:  # No samples collected – return zeroes.
            return SystemMetrics()

        def _mk_stats(buf: List[float]):
            return mean(buf), max(buf), pstdev(buf)

        cpu_avg, cpu_peak, cpu_std = _mk_stats(self._cpu)
        rss_avg, rss_peak, rss_std = _mk_stats(self._rss)
        vms_avg, vms_peak, vms_std = _mk_stats(self._vms)

        return SystemMetrics(
            cpu_usage_avg=cpu_avg,
            cpu_usage_peak=cpu_peak,
            cpu_usage_stddev=cpu_std,
            rss_avg_mb=rss_avg,
            rss_peak_mb=rss_peak,
            rss_stddev_mb=rss_std,
            vms_avg_mb=vms_avg,
            vms_peak_mb=vms_peak,
            vms_stddev_mb=vms_std,
        )

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                if self._process:
                    self._cpu.append(self._process.cpu_percent())
                    mem = self._process.memory_info()
                    self._rss.append(mem.rss / (1024 ** 2))
                    self._vms.append(mem.vms / (1024 ** 2))
                else:
                    self._cpu.append(psutil.cpu_percent())
                    mem = psutil.virtual_memory()
                    self._rss.append(mem.used / (1024 ** 2))
                    self._vms.append(mem.total / (1024 ** 2))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break  # Target process vanished or permission denied.

            time.sleep(self._interval_s)


class MacOSMetricsCollector(BaseMetricsCollector):

    def __init__(self, interval_s: float = 0.5):
        self._interval_s = interval_s
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._gpu_util: List[float] = []
        self._gpu_freq: List[float] = []

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join()

    def snapshot(self) -> SystemMetrics:
        if not self._gpu_util:  # No samples → return empty metrics.
            return SystemMetrics()

        util_avg, util_peak, util_std = mean(self._gpu_util), max(self._gpu_util), pstdev(self._gpu_util)
        freq_avg, freq_peak, freq_std = mean(self._gpu_freq), max(self._gpu_freq), pstdev(self._gpu_freq)

        return SystemMetrics(
            gpu_usage_avg=util_avg,
            gpu_usage_peak=util_peak,
            gpu_usage_stddev=util_std,
            gpu_freq_avg_mhz=freq_avg,
            gpu_freq_peak_mhz=freq_peak,
            gpu_freq_stddev_mhz=freq_std,
        )

    def _run(self):
        while not self._stop_evt.is_set():
            util, freq = self._query_powermetrics()
            if util is not None:
                self._gpu_util.append(util)
            if freq is not None:
                self._gpu_freq.append(freq)
            time.sleep(self._interval_s)

    @staticmethod
    def _query_powermetrics() -> tuple[Optional[float], Optional[float]]:
        """Run `powermetrics` once and parse GPU utilisation + frequency."""
        cmd = ["sudo", "powermetrics", "-s", "gpu_power", "-i", "100", "-n", "1"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                return None, None
            
            for line in proc.stdout.split('\n'):
                if 'GPU idle residency' in line:
                    try:
                        idle_pct = float(line.split(':')[1].strip().replace('%', ''))
                        util = 100.0 - idle_pct
                    except (ValueError, IndexError):
                        pass
                elif 'GPU HW active frequency' in line:
                    try:
                        freq = float(line.split(':')[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass

            return util, freq
        except Exception:
            return None, None
        

class MetricsTask:
    """Coordinates multiple collectors and aggregates a *single* JSON summary."""

    def __init__(self, pid: Optional[int] = None):
        self._collectors: List[BaseMetricsCollector] = self._select_collectors(pid)
        self._request_metrics: List[Metrics] = []

    def start(self):
        for c in self._collectors:
            c.start()

    def stop(self):
        for c in self._collectors:
            c.stop()

    def add_request_metrics(self, m: Metrics):
        self._request_metrics.append(m)

    def get_summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict with all aggregated numbers."""

        summary: Dict[str, Any] = {}

        if self._request_metrics:
            ok = [m for m in self._request_metrics if m.success]
            summary.update(
                total_requests=len(self._request_metrics),
                successful_requests=len(ok),
                success_rate=len(ok) / len(self._request_metrics),
            )
            if ok:
                lats = [m.end_to_end_latency_s for m in ok]
                ttfts = [m.time_to_first_token_s for m in ok if m.time_to_first_token_s]
                summary["latency"] = _make_percentiles(lats)
                if ttfts:
                    summary["time_to_first_token"] = _make_percentiles(ttfts)

        for col in self._collectors:
            snap = col.snapshot()
            key = col.__class__.__name__
            summary[key] = snap  # dataclasses are JSON‑serialisable via asdict()

        return summary

    @staticmethod
    def _select_collectors(pid: Optional[int]) -> List[BaseMetricsCollector]:
        plat = platform.system().lower()
        arch = platform.machine().lower()

        collectors: List[BaseMetricsCollector] = [ProcessMetricsCollector(pid=pid)]

        if plat == "darwin" and arch.startswith("arm"):
            collectors.append(MacOSMetricsCollector())

        return collectors


def _make_percentiles(values: List[float]) -> Dict[str, float]:
    """Return mean, p50, p90, p99 for a numeric sample set."""
    if not values:
        return {}
    values_sorted = sorted(values)
    n = len(values_sorted)
    return {
        "mean": mean(values_sorted),
        "p50": values_sorted[int(n * 0.50)],
        "p90": values_sorted[int(n * 0.90)],
        "p99": values_sorted[max(0, int(n * 0.99) - 1)],
    }
