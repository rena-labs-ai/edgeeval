import os
import argparse
import signal
import subprocess
import sys
import time
import system_profiling.NVIDIA as NVIDIA


def get_args():
    p = argparse.ArgumentParser(description="Profile-generation harness")
    p.add_argument("--results_dir",  type=str, required=True)
    p.add_argument("--backend_dir",  type=str, required=True)
    p.add_argument("--renacore_path", type=str, required=True)
    return p.parse_args()


def start_ollama(ollama_dir: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["go", "run", "main.go", "serve"],
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
    cli = f"{renacore_dir}/rena-browserd/target/debug/browserd-cli"
    proc = subprocess.Popen(
        [
            "conda", "run", "-n", "rena",
            str(cli), "query", prompt
        ],
        cwd=f"{renacore_dir}/rena-browserd",
        start_new_session=True,
    )
    print(f"Renacore request sent  (PID {proc.pid})")
    # proc.wait()
    return proc


def main():
    args = get_args()

    ollama_proc = start_ollama(args.backend_dir)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    abs_results_dir = os.path.abspath(args.results_dir)
    nvidia_pids = NVIDIA.start_all(abs_results_dir)
    renacore_proc = None

    # TODO: how to wait for Ollama server to be ready?
    time.sleep(10)
    renacore_proc = send_renacore_request(
        args.renacore_path, "Analysis of iris dataset")

    print("Profiling in progress — press Ctrl-C to stop.")
    try:
        signal.pause()
    except KeyboardInterrupt:
        print("Ctrl-C received, beginning shutdown …")
    finally:
        NVIDIA.stop_all(nvidia_pids)
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
