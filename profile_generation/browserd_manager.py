import os
import subprocess
from pathlib import Path
from typing import Optional

class RenaBrowserdManager:
    def __init__(self, browserd_path: str):
        self.browserd_path = browserd_path
        self.proc: Optional[subprocess.Popen] = None

    def eval(self, config_path: str, eval_path: str, trace_path: str):
        """Run rena evaluation under subprocess.
        This function waits until the evaluation is complete and returns 
        the json traces.
        
        An evaluation has multiple results, each result has several runs
        and each run has a number of steps with create_execution_plan and
        execute_execution_plan interleaved"""
        
        cli = f"{self.browserd_path}/target/release/browserd-cli"
        cmd = [
            "conda",
            "run",
            "-n",
            "rena",
            cli,
            "--config",
            config_path,
            "eval",
            eval_path,
        ]
        trace_file = Path(trace_path)
        trace_file.parent.mkdir(parents=True, exist_ok=True)  # ensure dir exists

        with trace_file.open("w", encoding="utf-8") as fout:
            self.proc = subprocess.Popen(
                cmd,
                stdout=fout,
                stderr=fout,
                cwd=self.browserd_path,
                text=True,
            )

    def get_pgid(self):
        assert self.proc is not None
        return os.getpgid(self.proc.pid)

    def wait(self):
        assert self.proc is not None
        self.proc.wait()