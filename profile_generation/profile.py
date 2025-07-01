import os
import argparse
import signal
import subprocess
import sys
import time
import system_profiling.NVIDIA as NVIDIA
import system_profiling.metrics as metrics


def get_args():
    p = argparse.ArgumentParser(description="Profile-generation harness")
    p.add_argument("--results_dir",  type=str, required=True)
    p.add_argument("--backend_dir",  type=str, required=True)
    p.add_argument("--renacore_path", type=str, required=True)
    return p.parse_args()


def start_ollama(ollama_dir: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["ollama", "serve"],
        cwd=ollama_dir
    )
    print(f"Ollama started (PID {proc.pid})")
    return proc


def stop_process(proc: subprocess.Popen, sudo: bool = False):
    if proc is None or proc.poll() is not None:
        return

    print(f"Stopping PID {proc.pid}", end=" ", flush=True)
    try:
        proc.terminate()
        proc.wait(timeout=5)
        print("done.")
    except subprocess.TimeoutExpired:
        print("timeout, forcing kill.")
        cmd = ["sudo", "kill", "-9",
               str(proc.pid)] if sudo else ["kill", "-9", str(proc.pid)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)


def send_renacore_request(renacore_dir: str, prompt: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            "cargo", "run", "--bin", "browserd-cli",
            "--release", "--", "query", prompt
        ],
        cwd=f"{renacore_dir}/rena-browserd",
        start_new_session=True,
    )
    print(f"Renacore request sent  (PID {proc.pid})")
    return proc


def _find_gid(proc: subprocess.Popen) -> int:
    return os.getpgid(proc.pid)


def main():
    args = get_args()

    ollama_proc = start_ollama(args.backend_dir)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    abs_results_dir = os.path.abspath(args.results_dir)
    renacore_proc = None

    # TODO: how to wait for Ollama server to be ready?
    time.sleep(10)
    renacore_proc = send_renacore_request(
        args.renacore_path, "Analysis of iris dataset")

    ollama_gid = _find_gid(ollama_proc)
    renacore_gid = _find_gid(renacore_proc)

    metrics_task_ollama = metrics.MetricsTask(pgid=ollama_gid, results_dir=abs_results_dir)
    metrics_task_renacore = metrics.MetricsTask(pgid=renacore_gid, results_dir=abs_results_dir)
    metrics_task_ollama.start()
    metrics_task_renacore.start()

    print("Profiling in progress — press Ctrl-C to stop.")
    try:
        signal.pause()
    except KeyboardInterrupt:
        print("Ctrl-C received, beginning shutdown …")
    finally:
        metrics_task_ollama.stop()
        metrics_task_renacore.stop()
        print(metrics_task_ollama.get_summary())
        print(metrics_task_renacore.get_summary())
        metrics_task_ollama.store_metrics()
        metrics_task_renacore.store_metrics()
        stop_process(renacore_proc)
        stop_process(ollama_proc)
        print("All background processes have exited. Bye!")


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGINT,  _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    try:
        main()
    except Exception as e:
        print(f"Unhandled error: {e}", file=sys.stderr)
        sys.exit(1)
