#!/usr/bin/env python3

import signal
import logging
import sys
import argparse
from pathlib import Path
from profile import (
    profile_browserd
)

def get_args():
    p = argparse.ArgumentParser(description="Profile-generation harness")
    # argument
    p.add_argument("browserd_path",
                    metavar="BROWSERD_PATH",
                    type=str,
                    help="Path to the rena-browserd directory"
                )
    p.add_argument("ollama_dir",
                   metavar="OLLAMA_DIR",
                   type=str,
                   help="Path to the Ollama directory. Other inference engines are not supported yet."
                )

    # optional arguments
    p.add_argument("--results-dir",
                   metavar="RESULTS_DIR",
                   type=str,
                   default=".results",
                   help="Path to the results directory")
    return p.parse_args()

def main():
    signal.signal(signal.SIGINT,  _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    logging.basicConfig(level=logging.INFO)

    args = get_args()
    print(f"args: {args}")
    results_path = Path(args.results_dir)
    if not results_path.exists():
        results_path.mkdir(parents=True)

    print(f"Profiling browserd with Ollama at {args.ollama_dir} and results in {results_path}")
    profile_browserd(args, results_path)


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled error: {e}", file=sys.stderr)
        sys.exit(1)
