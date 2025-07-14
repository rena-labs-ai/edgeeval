import os
import uuid
import time
import subprocess
import csv
from typing import List, Optional

from utils import Ollama_pcap_iter, ollama_get_tokens_breakdown
from transformers import AutoTokenizer

class OllamaTCPMonitor:
    def __init__(self, port: int=11434, tokenizer: Optional[AutoTokenizer]=None):
        """Capture the response from ollama server
        
        The response has following metrics:
         - list of num_tool_call_tokens for the whole process
         - list of num_total_tokens for the whole process (for every results, runs, steps)
        """
        self.port = port
        self.tokenizer = tokenizer
        self.num_tool_call_tokens: List[int] = []
        self.num_total_tokens: List[int] = []

        self._tcpdump_proc: Optional[subprocess.Popen] = None
        self._pcap_file: Optional[str] = None

        self.ollama_proc: Optional[subprocess.Popen] = None

    def __enter__(self):
        self._ollama_serve()
        self._tcpdump_start()
        
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._tcpdump_proc:
            self._tcpdump_proc.kill()
        if self._pcap_file:
            os.remove(self._pcap_file)
        if self.ollama_proc:
            self.ollama_proc.terminate()
            self.ollama_proc.wait()

    def get_pgid(self):
        assert self.ollama_proc is not None
        return os.getpgid(self.ollama_proc.pid)

    def _get_tmp_pcap_path(self):
        unique_id = str(uuid.uuid4())[:8]
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        pcap_path = f"./tmp/ollama_{unique_id}.pcap"
        with open(pcap_path, 'w') as f:
            pass
        os.chmod(pcap_path, 0o666)
        return pcap_path
    
    def _ollama_serve(self):
        self.ollama_proc = subprocess.Popen(
            ["ollama", "serve"],
        )
        time.sleep(10)
    
    def _tcpdump_start(self):
        self._pcap_file = self._get_tmp_pcap_path()
        cmd = [
            "sudo",
            "-n",
            "tcpdump",
            "-i",
            "lo",
            "tcp",
            "port",
            str(self.port),
            "-w",
            self._pcap_file,
        ]
        
        print(f"Starting tcpdump with command: {' '.join(cmd)}")
        
        self._tcpdump_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        
        time.sleep(1)  # Give tcpdump a moment to start
        
        if self._tcpdump_proc.poll() is not None:
            stderr_output = self._tcpdump_proc.stderr.read().decode() if self._tcpdump_proc.stderr else "Unknown error"
            print(f"ERROR: tcpdump failed to start. Exit code: {self._tcpdump_proc.returncode}")
            print(f"Error message: {stderr_output}")
            raise RuntimeError(f"tcpdump failed to start: {stderr_output}")
        else:
            print(f"SUCCESS: tcpdump process started with PID: {self._tcpdump_proc.pid}")
        
    def _parse_pcap(self):
        pcap_iter = Ollama_pcap_iter(self._pcap_file)
        for response in pcap_iter:
            tokens = ollama_get_tokens_breakdown(response, self.tokenizer)
            if tokens['tool_call_tokens'] >0:
                self.num_tool_call_tokens.append(tokens['tool_call_tokens'])
                self.num_total_tokens.append(tokens['total_tokens'])

    def dump_csv(self, results_dir: str):
        self._parse_pcap()
        print(f"num_tool_call_tokens: {self.num_tool_call_tokens}")
        print(f"num_total_tokens: {self.num_total_tokens}")
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Write token metrics to CSV
        csv_path = os.path.join(results_dir, "token_metrics.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Index', 'Tool Call Tokens', 'Total Tokens']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i, (tool_tokens, total_tokens) in enumerate(zip(self.num_tool_call_tokens, self.num_total_tokens), 1):
                writer.writerow({
                    'Index': i,
                    'Tool Call Tokens': tool_tokens,
                    'Total Tokens': total_tokens
                })
        
        print(f"Token metrics dumped to {csv_path}")

        