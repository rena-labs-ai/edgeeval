"""MLC LLM Bench – request records, aggregation, and pretty‑printing.

Interfaces preserved exactly as before (`RequestRecord`, `ServerMetrics`,
`Metrics`, `generate_metrics_summary`, `convert_reports_to_df`,
`pretty_print_report`).  Internally, we now tolerate both *pydantic* models and
*dataclass* instances for the `metrics` field by converting either form to a
plain dict via `_to_dict`.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import pandas as pd  # pylint: disable=import-error
from pydantic import BaseModel, Field

from macos_bench.metrics import Metrics as ClientMetrics, SystemMetrics  # dataclass
from macos_bench.protocol.openai_api_protocol import ChatCompletionRequest


class ServerMetrics(BaseModel):
    input_tokens: int
    prefill_tokens: int
    output_tokens: int
    end_to_end_latency_s: float
    prefill_tokens_per_s: float
    inter_token_latency_s: float
    time_per_output_token_s: float
    time_to_first_token_s: Optional[float] = None
    system_metrics: Optional[SystemMetrics] = None


class Metrics(BaseModel):  # legacy pydantic wrapper when needed
    success: bool
    start_time: float
    finish_time: float
    end_to_end_latency_s: float

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    inter_token_latency_s: Optional[float] = None
    time_per_output_token_s: Optional[float] = None
    time_to_first_token_s: Optional[float] = None
    system_metrics: Optional[SystemMetrics] = None
    exec_feature: Optional[Dict[str, Any]] = None


class RequestRecord(BaseModel):
    request_id: Optional[int] = None
    chat_cmpl: ChatCompletionRequest
    output_str: Optional[str] = None
    first_chunk_output_str: str = ""
    timestamp: Optional[float] = None

    # Accept either the dataclass Metrics or the pydantic wrapper.
    metrics: Optional[ClientMetrics] | Optional[Metrics] = Field(default=None)
    server_metrics: Optional[ServerMetrics] = None
    error_msg: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error_msg is None

class GroupedRequestRecord(RequestRecord):
    """The data structure for request record groups.
    For datasets that have common prefix sharing, the request records
    that share a same common prefix will be wrapped in a GroupedRequestRecord
    at the beginning.
    """

    records: List[RequestRecord]
    
def generate_metrics_summary(
    request_records: List[RequestRecord],
    num_total_requests: int,
    num_gpus: int,
) -> Dict[str, Any]:
    completed = [r for r in request_records if r.metrics]
    if not completed:
        return {}

    duration = (
        max(_to_dict(r.metrics)["finish_time"] for r in completed)
        - min(_to_dict(r.metrics)["start_time"] for r in completed)
    ) or 1e-5

    metrics_df = pd.DataFrame([_to_dict(r.metrics) for r in completed])

    report: Dict[str, Any] = {
        "num_total_requests": num_total_requests,
        "num_completed_requests": len(completed),
        "num_gpus": num_gpus,
        "duration": duration,
        "request_throughput": len(completed) / duration,
    }

    _add_stats(report, metrics_df, "end_to_end_latency_s", factor=1e3, alias="latency_ms")
    _add_stats(report, metrics_df, "time_to_first_token_s", factor=1e3, alias="ttft_ms")
    _add_stats(report, metrics_df, "time_per_output_token_s", factor=1e3, alias="tpot_ms")
    _add_stats(report, metrics_df, "inter_token_latency_s", factor=1e3, alias="itl_ms")
    _add_stats(report, metrics_df, "input_tokens")
    _add_stats(report, metrics_df, "output_tokens")

    tot_in = metrics_df["input_tokens"].sum()
    tot_out = metrics_df["output_tokens"].sum()
    report.update(
        total_input_tokens=int(tot_in),
        total_output_tokens=int(tot_out),
        input_token_throughput=tot_in / duration,
        output_token_throughput=tot_out / duration,
        input_token_throughput_per_gpu=(tot_in / duration) / num_gpus,
        output_token_throughput_per_gpu=(tot_out / duration) / num_gpus,
    )

    server_metrics = [r.server_metrics for r in completed if r.server_metrics]
    if server_metrics:
        server_df = pd.DataFrame([m.model_dump() for m in server_metrics])
        server_report: Dict[str, Any] = {}
        _add_stats(server_report, server_df, "end_to_end_latency_s")
        _add_stats(server_report, server_df, "input_tokens")
        _add_stats(server_report, server_df, "output_tokens")
        for field in ("cpu_usage_avg", "gpu_usage_avg", "rss_peak_mb", "gpu_freq_peak_mhz"):
            if field in server_df.columns:
                server_report[field] = float(server_df[field].dropna().iloc[-1])
        report["server_metrics"] = server_report

    report["exec_feature"] = _to_dict(completed[0].metrics).get("exec_feature")
    return report

def convert_reports_to_df(reports: List[Dict[str, Any]]) -> pd.DataFrame:
    def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(_flatten(v, key))
            else:
                flat[key] = v
        return flat

    return pd.DataFrame([_flatten(r) for r in reports])


def pretty_print_report(report: Dict[str, Any], *, _is_sub: bool = False) -> None:  # noqa: C901
    """Pretty console output.  When called recursively for server_metrics,
    `_is_sub` is True so we switch to a simpler layout that doesn’t expect
    top‑level request counters.
    """

    def hdr(t: str):
        print(f" {t} ".center(80, "="))

    def sec(t: str):
        print(t.center(80, "-"))

    if _is_sub or "num_total_requests" not in report:
        # sub‑report: just dump scalar fields and latency if present
        hdr("Server Metrics")
        for k, v in report.items():
            if isinstance(v, dict):
                continue
            if isinstance(v, SystemMetrics):
                print(f"CPU usage avg: {v.cpu_usage_avg}, peak: {v.cpu_usage_peak}, stddev: {v.cpu_usage_stddev}")
                print(f"RSS avg: {v.rss_avg_mb}, peak: {v.rss_peak_mb}, stddev: {v.rss_stddev_mb}")
                print(f"VMS avg: {v.vms_avg_mb}, peak: {v.vms_peak_mb}, stddev: {v.vms_stddev_mb}")
            print(f"{k:<40}{v}")
        if "latency_ms" in report:
            sec("Latency (ms)")
            s = report["latency_ms"]
            print(f"mean={s['mean']:.2f} p50={s['quantiles']['p50']:.2f} p95={s['quantiles']['p95']:.2f} max={s['max']:.2f}")
        return

    # top‑level benchmark report ---------------------------------------- #
    hdr("Benchmark Result")
    print(f"{'Total requests:':<50}{report['num_total_requests']}")
    print(f"{'Completed requests:':<50}{report['num_completed_requests']}")
    print(f"{'Duration (s):':<50}{report['duration']:.2f}")
    print(f"{'Request throughput (req/s):':<50}{report['request_throughput']:.2f}")

    sec("Latency (ms)")
    for k in ("latency_ms", "ttft_ms", "tpot_ms", "itl_ms"):
        if k in report:
            s = report[k]
            print(f"{k:<25} mean={s['mean']:.2f} p50={s['quantiles']['p50']:.2f} p95={s['quantiles']['p95']:.2f} max={s['max']:.2f}")

    sec("Token stats")
    for k in ("input_tokens", "output_tokens"):
        if k in report:
            s = report[k]
            print(f"{k:<25} mean={s['mean']:.1f} p50={s['quantiles']['p50']:.1f} max={s['max']}")

    if "server_metrics" in report:
        pretty_print_report(report["server_metrics"], _is_sub=True)


def _add_stats(tgt: Dict[str, Any], df: pd.DataFrame, col: str, *, factor: float = 1.0, alias: Optional[str] = None):
    if col not in df.columns:
        return
    series = df[col].dropna() * factor
    if series.empty:
        return
    key = alias or col
    tgt[key] = {
        "mean": series.mean(),
        "stddev": series.std(ddof=0),
        "min": series.min(),
        "max": series.max(),
        "quantiles": {f"p{int(q*100)}": v for q, v in series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).items()},
    }


def _to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported metrics type: {type(obj)}")

