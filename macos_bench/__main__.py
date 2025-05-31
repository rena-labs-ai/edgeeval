"""MLC LLM benchmark main entrance"""

import functools
import json
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from transformers import AutoTokenizer  # pylint: disable=import-error
import psutil

from macos_bench.api_endpoint import SUPPORTED_BACKENDS, create_api_endpoint
from macos_bench.dataset import SUPPORTED_DATASET, Dataset, create_dataset
from macos_bench.request_processor import (
    MetricAnalyzer,
    RequestProcessor,
    create_pipelines,
)
from macos_bench.request_record import (
    RequestRecord,
    convert_reports_to_df,
    generate_metrics_summary,
    pretty_print_report,
)
from macos_bench.metrics import MetricsTask, Metrics

import macos_bench.support.argparse as argparse
import macos_bench.support.logging as logging

logging.enable_logging()
logger = logging.getLogger(__name__)


def _parse_num_concurrent_requests(num_str: Optional[str]) -> Optional[List[int]]:
    if num_str is None:
        return None
    numbers = num_str.split(",")
    if any(not number.isdigit() for number in numbers):
        raise ValueError(f"Unrecognized num_concurrent_requests list: {numbers}")
    return list(int(number) for number in numbers)


def _parse_request_rate(request_rate_str: Optional[str]) -> Optional[List[np.float32]]:
    if request_rate_str is None:
        return None
    request_rates = request_rate_str.split(",")
    results = []
    for rate_str in request_rates:
        request_rate = float(rate_str)
        if request_rate <= 0:
            raise ValueError(f"Invalid request rate {request_rate}")
        results.append(np.float32(request_rate))
    return results


def run_pipeline(
    pipeline: RequestProcessor,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    args: argparse.argparse.Namespace,
) -> Tuple[Dict[str, Any], List[RequestRecord]]:
    """Run the pipeline with the given dataset and args. Return the benchmark report dict."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Find the MLC server processes
    server_pid = None
    try:
        import subprocess
        result = subprocess.run(['lsof', '-i', f':{args.port}'], capture_output=True, text=True)
        if result.stdout:
            for line in result.stdout.splitlines()[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 2:
                    pid = int(parts[1])
                    server_pid = pid
                    logger.info(f"Found PID {pid} for inference server")
    except Exception as e:
        logger.warning(f"Error running lsof: {e}")
    
    if not server_pid:
        logger.warning(f"Could not find inference server process on port {args.port}")

    request_records = dataset.generate_request_records(
        args.input_len,
        args.output_len,
        args.input_len_std,
        args.output_len_std,
    )
    
    # Initialize metrics task with server PID
    metrics_task = MetricsTask(pid=server_pid)
    metrics_task.start()
    
    try:
        # Process all requests
        request_records = pipeline(request_records)
        
        num_total_requests = (
            args.num_requests if not args.per_gpu_workload else args.num_requests * args.num_gpus
        )
        assert len(request_records) == num_total_requests
        sorted_requests: List[RequestRecord] = [None] * num_total_requests
        for request_record in request_records:
            assert request_record.request_id is not None
            assert sorted_requests[request_record.request_id] is None
            sorted_requests[request_record.request_id] = request_record
            
            # Record metrics for each request
            metrics = Metrics(
                success=request_record.metrics.success,
                start_time=request_record.metrics.start_time,
                finish_time=request_record.metrics.finish_time,
                end_to_end_latency_s=request_record.metrics.end_to_end_latency_s,
                input_tokens=request_record.metrics.input_tokens,
                output_tokens=request_record.metrics.output_tokens,
                time_to_first_token_s=request_record.metrics.time_to_first_token_s,
                system_metrics=request_record.metrics.system_metrics,
            )
            metrics_task.add_request_metrics(metrics)
            
    finally:
        metrics_task.stop()

    # Get final metrics summary
    metrics_summary = metrics_task.get_summary()
    print(metrics_summary)
    
    request_records = MetricAnalyzer(tokenizer)(request_records)
    report = generate_metrics_summary(request_records, num_total_requests, args.num_gpus)
    
    # Add system metrics to the report
    if "server_metrics" not in report:
        report["server_metrics"] = {}
    report["server_metrics"].update(metrics_summary)
    
    return report, sorted_requests


def main(args: argparse.argparse.Namespace):
    """Main benchmark entrance."""
    if args.num_requests <= 0:
        raise ValueError("Number of requests to benchmark must be positive.")

    def _main():
        # Skip tokenizer loading for Ollama models
        tokenizer = None
        if args.api_endpoint != "ollama":
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        
        dataset = create_dataset(args, tokenizer)
        f_create_api_endpoint = functools.partial(create_api_endpoint, args)
        pipelines = create_pipelines(args, f_create_api_endpoint, dataset)
        
        reports = []
        alltime_records = {}
        for i, pipeline in enumerate(pipelines):
            report, request_records = run_pipeline(pipeline, dataset, tokenizer, args)
            exec_feature = (
                json.dumps(report["exec_feature"])
                if report["exec_feature"] is not None
                else f"pipeline{i}"
            )
            alltime_records[exec_feature] = [
                request_record.model_dump() for request_record in request_records
            ]
            reports.append(report)
            pretty_print_report(report)

        # Construct data frame
        df = convert_reports_to_df(reports)
        print(df)
        df.to_csv(args.output, index=False)
        logger.info("Benchmark results dumped to file %s", args.output)
        if args.debug_dump:
            debug_dump_filepath = (
                args.output[:-4] if args.output.endswith(".csv") else args.output
            ) + "_debug_dump.log"
            with open(debug_dump_filepath, "w", encoding="utf-8") as file:
                json.dump(alltime_records, file, indent=4)
            logger.info("Debug log dumped to file %s", debug_dump_filepath)

    _main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MLC LLM benchmark")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASET,
        help=f"The benchmark dataset kind. Supporting {SUPPORTED_DATASET}",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="The dataset file path.",
    )
    parser.add_argument(
        "--api-endpoint",
        type=str,
        choices=SUPPORTED_BACKENDS,
        default="openai",
        help="The API endpoint API for benchmarking.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="The path of the tokenizer directory. ",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="The number of GPUs used by the server. "
        "We need this to better analyze the throughput per GPU.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        required=True,
        help="The number of requests for benchmark.",
    )
    parser.add_argument(
        "--num-warmup-requests",
        type=int,
        help="The number of requests for warmup. "
        "It is optional when fixing the number of concurrent requests, and is required otherwise.",
    )
    parser.add_argument(
        "--per-gpu-workload",
        default=False,
        action="store_true",
        help='When set to True, the specified "num_concurrent_requests"/"request_rate" '
        "denote the workload **per GPU**, which means that the real values of "
        '"num_concurrent_requests"/"request_rate" used in benchmark'
        'will be multiplied by "num_gpus".',
    )
    parser.add_argument(
        "--num-concurrent-requests",
        type=_parse_num_concurrent_requests,
        help="The number(s) of concurrent requests to benchmark. "
        'It can be either one integer or a list of integer separated by commas(","). '
        "When specified, for each integer, the benchmark keeps these many consistent "
        "number of concurrently running requests.",
    )
    parser.add_argument(
        "--request-rate",
        type=_parse_request_rate,
        help="The request rate(s) denoting the number of new requests each second. "
        'It can be either one float number (or "inf") or a list of numbers separated '
        'by commas(","). '
        "When specified, the benchmark sends these many new requests each second. "
        'If it is "inf", all requests will be sent together at once.',
    )
    parser.add_argument(
        "--replay-timestamp-scale",
        type=float,
        help="The timestamp scale when replaying the timestamps in a dataset. "
        'The dataset replay mode is enabled when neither "--num-concurrent-requests" and '
        '"--request-rate" is specified. '
        "The scale is 1 by default in the replay mode.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        help="The benchmark request average input length. Default to None, "
        "which means the request input length depends on the dataset being used.",
    )
    parser.add_argument(
        "--input-len-std",
        type=float,
        default=0,
        help="The benchmark request input length standard deviation. Default to 0.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        help="The benchmark request average output length. Default to None, "
        "which means the request output length depends on the dataset being used.",
    )
    parser.add_argument(
        "--output-len-std",
        type=float,
        default=0,
        help="The benchmark request output length standard deviation. Default to 0.",
    )
    parser.add_argument(
        "--stream",
        type=bool,
        default=True,
        help="Whether to benchmark stream responses. "
        "When not enabled, metrics such as time-to-first-token (TTFT) will not be available. "
        "Default to True.",
    )
    parser.add_argument(
        # NOTE: The current implementation of server metrics still has some issues that need fixes,
        # which makes it not work to include server metrics.
        "--include-server-metrics",
        action="store_true",
        help="Whether to also benchmark the server side request metrics. "
        "This option is only available when benchmarking MLC server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="The host address of the backend API.",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="The port of the backend API.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3 * 60 * 60,
        help="The timeout limit of each request.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The random number seed. Default to 0.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature value for logit adjustment. Default to 1.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="The top-p value for sampling. Default to 1.",
    )
    parser.add_argument(
        "--ignore-eos",
        default=False,
        action="store_true",
        help='Whether to set the "ignore_eos" field.',
    )
    parser.add_argument(
        "--apply-chat-template",
        default=False,
        action="store_true",
        help="Whether to apply chat template to the request input text. "
        'It is not supported when "--input-len" is specified.',
    )
    parser.add_argument(
        "--num-process-workers",
        type=int,
        help="The number of parallel process workers to send the requests.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Whether to disable showing progress bar with tqdm during benchmarking.",
    )
    parser.add_argument(
        "--max-schedule-gap",
        type=float,
        default=0.5,
        help="The maximum allowed delay between the scheduled time in seconds.",
    )
    parser.add_argument(
        "--mlc-model-lib",
        type=str,
        help="The model lib path when benchmarking MLC serve. "
        "When specified, the server is automatic launched and no external server launch is needed.",
    )
    parser.add_argument(
        "--cuda-profile",
        default=False,
        action="store_true",
        help="Whether to enable cuda profile on server. "
        "The --mlc-model-lib path should be provided when enabling this option.",
    )
    parser.add_argument(
        "--debug-dump",
        default=False,
        action="store_true",
        help="Whether to dump all request record raw data to file.",
    )
    parser.add_argument(
        "--multi-round",
        default=False,
        action="store_true",
        help="Whether to chat like multi round conversion with history log each request. "
        "Only enabled when benchmarked with fixed concurrent request mode."
        "The --num-concurrent-requests should be provided when enabling this option.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="benchmark.csv",
        help="The path of the output file where to dump the benchmark results.",
    )

    main(parser.parse_args())
