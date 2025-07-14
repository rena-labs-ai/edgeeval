import os
import csv
from typing import List
from dataclasses import dataclass
from utils import extract_json_from_trace, parse_step_metrics

@dataclass
class StepMetrics:
    time_model_load: float = 0  # ms
    time_llm_call: float = 0  # ms
    time_tool_call: float = 0  # ms
    prefill_time: float = 0  # ms
    decode_time: float = 0  # ms
    prefill_tokens: int = 0
    decode_tokens: int = 0

@dataclass
class RunMetrics:
    total_time: float  # ms
    eval_score: float
    total_time_llm_calls: float  # ms
    total_time_tool_calls: float  # ms
    steps: List[StepMetrics]

class EvalTraceProcessor:
    def __init__(self, trace_path: str):
        """Each trace has multiple results, each result
        has several runs and each run has a number of steps with
        create_execution_plan and execute_execution_plan interleaved
        
        The trace has following metrics for each (result_id, run_id):
         - total_time (ms)
         - eval_score
         - total_time_llm_calls (ms)
         - total_time_tool_calls (ms)
         - list of time_llm_calls (ms) for each step
         - list of time_tool_calls (ms) for each step

         - list of prefill_time (ms) for each step
         - list of decode_time (ms) for each step
         - list of prefill_tokens (int) for each step
         - list of decode_tokens (int) for each step
        """

        self.trace_path = trace_path
        self.results: List[List[RunMetrics]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _load_trace(self):
        trace_data = extract_json_from_trace(self.trace_path)
        for res in trace_data['results']:
            runs: List[RunMetrics] = []
            for run in res['runs']:
                steps: List[StepMetrics] = []
                for step in run['trace']['inner_traces']:
                    if step['label'] == 'create_execution_plan':
                        # if len(steps) > 0 and steps[-1].time_tool_call == 0:
                        #     steps.pop()
                        time_model_load, \
                        time_llm_call, \
                        prefill_time, \
                        decode_time, \
                        prefill_tokens, \
                        decode_tokens = parse_step_metrics(step['inner_traces'][0]['metadata'])
                        step_metrics = StepMetrics(
                            time_model_load=time_model_load,
                            time_llm_call=time_llm_call,
                            prefill_time=prefill_time,
                            decode_time=decode_time,
                            prefill_tokens=prefill_tokens,
                            decode_tokens=decode_tokens)
                        steps.append(step_metrics)
                    elif step['label'] == 'execute_execution_plan':
                        steps[-1].time_tool_call = step['latency']
                    
                # if len(steps) > 0 and steps[-1].time_tool_call == 0:
                #     steps.pop()
                run_total_time = run['trace']['latency']
                eval_score = run['eval_score']
                total_time_llm_calls = sum(step.time_llm_call for step in steps)
                total_time_tool_calls = sum(step.time_tool_call for step in steps)
                run_metrics = RunMetrics(
                    total_time=run_total_time,
                    eval_score=eval_score,
                    total_time_llm_calls=total_time_llm_calls,
                    total_time_tool_calls=total_time_tool_calls,
                    steps=steps)
                runs.append(run_metrics)
            self.results.append(runs)

    def _dump_total_metrics(self, results_dir: str):
        run_metrics = self.results[0][0]
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Write total metrics to CSV
        csv_path = os.path.join(results_dir, "total_metrics.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Total Time (ms)', 'Eval Score', 'Total LLM Calls Time (ms)', 'Total Tool Calls Time (ms)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerow({
                'Total Time (ms)': run_metrics.total_time,
                'Eval Score': run_metrics.eval_score,
                'Total LLM Calls Time (ms)': run_metrics.total_time_llm_calls,
                'Total Tool Calls Time (ms)': run_metrics.total_time_tool_calls
            })
        
        print(f"Total metrics dumped to {csv_path}")
    
    def _dump_step_metrics(self, results_dir: str):
        run_metrics = self.results[0][0]
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Write step metrics to CSV
        csv_path = os.path.join(results_dir, "step_metrics.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Index', 'Model load (ms)', 'LLM call (ms)', 'Tool call (ms)', 'Prefill (ms)', 'Decode (ms)', 'Prefill tokens', 'Decode tokens']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i, step in enumerate(run_metrics.steps, 1):
                writer.writerow({
                    'Index': i,
                    'Model load (ms)': step.time_model_load,
                    'LLM call (ms)': step.time_llm_call,
                    'Tool call (ms)': step.time_tool_call,
                    'Prefill (ms)': step.prefill_time,
                    'Decode (ms)': step.decode_time,
                    'Prefill tokens': step.prefill_tokens,
                    'Decode tokens': step.decode_tokens
                })
        
        print(f"Step metrics dumped to {csv_path}")

    def dump_csv(self, results_dir: str):
        self._load_trace()
        print(f"results: {self.results}")
        assert len(self.results) == 1, "Currently only support one result"
        assert len(self.results[0]) == 1, "Currently only support one run"

        # dump total metrics
        self._dump_total_metrics(results_dir)

        # dump step metrics
        self._dump_step_metrics(results_dir)
        
