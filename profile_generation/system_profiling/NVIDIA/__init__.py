from typing import Tuple, Dict
import psutil
import subprocess
import time
import signal
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

CWD = os.path.dirname(os.path.abspath(__file__))


def _find_pid(substr: str, timeout: float = 5) -> int:
    """
    Return the first PID whose cmdline contains *substr*.
    Raise RuntimeError if nothing turns up within *timeout* seconds.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        for p in psutil.process_iter(["pid", "cmdline"]):
            # p.info["cmdline"] can be None on zombie processes
            if p.info["cmdline"] and substr in " ".join(p.info["cmdline"]):
                return p.info["pid"]
        time.sleep(0.2)

    raise RuntimeError(f"Unable to locate process containing '{substr}'")


def record_cpu_utilization(results_dir: str) -> Tuple[int, subprocess.Popen]:
    proc = subprocess.Popen(["./record_cpu_compute.sh", results_dir], cwd=CWD)
    pid = _find_pid("record_cpu_compute")
    return pid, proc


def record_cpu_bandwidth(results_dir: str) -> Tuple[int, subprocess.Popen]:
    proc = subprocess.Popen(["./record_cpu_mem_bw.sh", results_dir], cwd=CWD)
    pid = _find_pid("pcm-memory")
    return pid, proc


def record_gpu_utilization(results_dir: str) -> Tuple[int, subprocess.Popen]:
    proc = subprocess.Popen(
        ["./record_gpu_mem_compute.sh", results_dir], cwd=CWD)
    pid = _find_pid("dcgmi")
    return pid, proc


def record_power(results_dir) -> Tuple[int, subprocess.Popen]:
    proc = subprocess.Popen(["./record_power.sh", results_dir], cwd=CWD)
    pid = _find_pid("record_power")
    return pid, proc


def start_all(results_dir: str) -> Dict[str, int]:
    """
    Launch every collector and return a dict of PIDs so you can
    stop them later exactly like in the README.
    """
    pids = {}
    pids["cpu_usage"], _ = record_cpu_utilization(results_dir)
    pids["cpu_bw"],   _ = record_cpu_bandwidth(results_dir)
    pids["gpu_util"], _ = record_gpu_utilization(results_dir)
    pids["power"],    _ = record_power(results_dir)
    return pids


def stop_all(pids: Dict[str, int]) -> None:
    """
    Kill every collector with the right privilege level
    (mirrors the README kill commands).
    """
    def _kill(pid: int, sudo: bool = False):
        if sudo:
            subprocess.run(["sudo", "kill", "-9", str(pid)],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    _kill(pids.get("cpu_usage", 0))                   # kill -9 $cpu_usage_pid
    # sudo kill -9 $cpu_mem_bw_pid
    _kill(pids.get("cpu_bw", 0), sudo=True)
    # kill -9 $gpu_utilization_pid
    _kill(pids.get("gpu_util", 0))
    _kill(pids.get("power", 0),    sudo=True)         # sudo kill -9 $power_pid
