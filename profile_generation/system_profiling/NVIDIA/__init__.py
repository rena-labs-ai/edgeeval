from dataclasses import dataclass
from typing import Tuple, Dict
import psutil
import subprocess
import time
import signal
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

CWD = os.path.dirname(os.path.abspath(__file__))


def record_ollam_cpu_utilization(ollama_gid: int, results_dir: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["./cpu/gid.sh", str(ollama_gid), "0.2", results_dir, "ollama"],
        cwd=CWD
    )
    return proc


def record_renacore_cpu_utilization(renacore_gid: int, results_dir: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["./cpu/gid.sh", str(renacore_gid), "0.2", results_dir, "renacore"],
        cwd=CWD
    )
    return proc


def record_cpu_bandwidth(results_dir: str) -> subprocess.Popen:
    proc = subprocess.Popen(["./record_cpu_mem_bw.sh", results_dir], cwd=CWD)
    return proc


def record_gpu_utilization(results_dir: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["./record_gpu_mem_compute.sh", results_dir], cwd=CWD)
    return proc


def record_power(results_dir) -> subprocess.Popen:
    proc = subprocess.Popen(["./record_power.sh", results_dir], cwd=CWD)
    return proc


def start_all(ollama_gid, renacore_gid, results_dir: str) -> Dict[str, subprocess.Popen]:
    """
    Launch every collector and return a dict of PIDs so you can
    stop them later exactly like in the README.
    """
    pids = {}
    # pids["cpu_usage"] = record_cpu_utilization(results_dir)
    pids["cpu_usage_ollama"] = record_ollam_cpu_utilization(
        ollama_gid, results_dir)
    pids["cpu_usage_renacore"] = record_renacore_cpu_utilization(
        renacore_gid, results_dir)
    pids["cpu_bw"] = record_cpu_bandwidth(results_dir)
    pids["gpu_util"] = record_gpu_utilization(results_dir)
    pids["power"] = record_power(results_dir)
    return pids


def stop_all(pids: Dict[str, subprocess.Popen]) -> None:
    """
    Gracefully terminate every collector process:
    1. Send SIGTERM to allow clean shutdown.
    2. Wait briefly.
    3. Send SIGKILL if still running.
    """
    def _terminate(pid: int, sudo: bool = True):
        # Step 1: Try to terminate gracefully
        if sudo:
            subprocess.run(["sudo", "kill", str(pid)],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        else:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                return  # Already gone

        # Step 2: Wait up to 2 seconds for the process to exit
        for _ in range(10):
            try:
                os.kill(pid, 0)  # Check if still alive
                time.sleep(0.2)
            except ProcessLookupError:
                return  # Process exited

        # Step 3: Force kill if still alive
        if sudo:
            subprocess.run(["sudo", "kill", "-9", str(pid)],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Already exited

    for key in pids:
        _terminate(pids[key].pid)
