import argparse

from transformers import AutoTokenizer

from system_monitor import SystemMonitor
from tcp_monitor import OllamaTCPMonitor
from trace_processor import EvalTraceProcessor
from browserd_manager import RenaBrowserdManager


def profile_browserd(args):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    with OllamaTCPMonitor(tokenizer=tokenizer) as ollama_monitor:
        system_monitor = SystemMonitor(args.results_dir)
        
        rena_manager = RenaBrowserdManager(args.browserd_path)
        rena_manager.eval(args.config_path, args.eval_path, args.trace_path)

        system_monitor.add_ollama_pgid(ollama_monitor.get_pgid())
        system_monitor.add_browserd_pgid(rena_manager.get_pgid())
        
        system_monitor.start()
        rena_manager.wait()
        system_monitor.stop()

        print(f"system monitor dumping metrics to {args.results_dir}")
        system_monitor.dump_csv(args.results_dir)
        print(f"ollama monitor dumping metrics to {args.results_dir}")
        ollama_monitor.dump_csv(args.results_dir)
    
    trace_processor = EvalTraceProcessor(args.trace_path)
    print(f"trace processor dumping csv to {args.results_dir}")
    trace_processor.dump_csv(args.results_dir)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browserd_path", type=str, required=True)
    parser.add_argument("--trace_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    profile_browserd(args)