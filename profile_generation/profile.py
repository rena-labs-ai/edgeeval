import os
import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path
import logging
import system_profiling.metrics as metrics

STOP_PROCESS_TIMEOUT = 5


def start_ollama(ollama_dir: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["ollama", "serve"],
        cwd=ollama_dir
    )
    logging.info(f"Ollama started (PID {proc.pid})")
    return proc


def stop_process(proc: subprocess.Popen):
    if proc is None or proc.poll() is not None:
        return

    logging.info(f"Stopping PID {proc.pid}")
    try:
        proc.terminate()
        proc.wait(timeout=STOP_PROCESS_TIMEOUT)
        logging.info("done.")
    except subprocess.TimeoutExpired:
        logging.info("timeout, forcing kill.")
        subprocess.run(["kill", "-9", str(proc.pid)],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=False)


def send_browserd_request(browserd_path: str, prompt: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            "cargo", "run", "--bin", "browserd-cli",
            "--release", "--", "query", prompt
        ],
        cwd=browserd_path,
        start_new_session=True,
    )
    logging.info(f"Browserd request sent  (PID {proc.pid})")
    return proc


def _find_gid(proc: subprocess.Popen) -> int:
    return os.getpgid(proc.pid)

def profile_browserd(args: argparse.Namespace, results_path: Path):
    ollama_proc = start_ollama(args.ollama_dir)
    # TODO: how to wait for Ollama server to be ready?
    time.sleep(10)

    browserd_proc = send_browserd_request(
        args.browserd_path, "Analysis of iris dataset")

    ollama_gid = _find_gid(ollama_proc)
    browserd_gid = _find_gid(browserd_proc)

    ollama_metric_task = metrics.MetricsTask(pgid=ollama_gid, results_dir=results_path.absolute())
    browserd_metric_task = metrics.MetricsTask(pgid=browserd_gid, results_dir=results_path.absolute())
    ollama_metric_task.start()
    browserd_metric_task.start()

    logging.info("Profiling in progress.")
    try:
        # signal.pause()
        browserd_proc.wait()
    except KeyboardInterrupt:
        logging.info("Ctrl-C received, beginning shutdown …")
    finally:
        ollama_metric_task.stop()
        browserd_metric_task.stop()
        print(ollama_metric_task.get_summary())
        print(browserd_metric_task.get_summary())
        stop_process(browserd_proc)
        stop_process(ollama_proc)
        logging.info("All background processes have exited. Bye!")
