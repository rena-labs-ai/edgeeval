from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import pandas as pd  # pylint: disable=import-error
from pydantic import BaseModel, Field

from macos_bench.metrics import Metrics, ServerMetrics  # dataclass
from macos_bench.protocol.openai_api_protocol import ChatCompletionRequest




class RequestRecord(BaseModel):
    request_id: Optional[int] = None
    chat_cmpl: ChatCompletionRequest
    output_str: Optional[str] = None
    first_chunk_output_str: str = ""
    timestamp: Optional[float] = None

    # Accept either the dataclass Metrics or the pydantic wrapper.
    metrics: Optional[Metrics] = None
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
    `_is_sub` is True so we switch to a simpler layout that doesn't expect
    top-level request counters.
    """

    def hdr(t: str):
        print(f" {t} ".center(60, "="))

    def sec(t: str):
        print(t.center(60, "-"))

    if _is_sub or "num_total_requests" not in report:
        hdr("System Metrics")

        def _kv(label: str, value: Optional[float], unit: str = "") -> None:
            txt = f"{value:>8.2f}{unit}" if value is not None else "   –"
            print(f"  {label:<24}{txt}")

        for k, v in report.items():
            if isinstance(v, dict):
                continue
            if k == "ProcessMetricsCollector":
                sec("CPU (%)")
                _kv("avg",   v.cpu_usage_avg)
                _kv("peak",  v.cpu_usage_peak)
                _kv("stddev", v.cpu_usage_stddev)

                sec("Memory RSS (MB)")
                _kv("avg",   v.rss_avg_mb)
                _kv("peak",  v.rss_peak_mb)
                _kv("stddev", v.rss_stddev_mb)

                sec("Memory VMS (MB)")
                _kv("avg",   v.vms_avg_mb)
                _kv("peak",  v.vms_peak_mb)
                _kv("stddev", v.vms_stddev_mb)
                
            if k == "MacOSMetricsCollector":
                sec("GPU (%)")
                _kv("avg",   v.gpu_usage_avg)
                _kv("peak",  v.gpu_usage_peak)
                _kv("stddev", v.gpu_usage_stddev)

                sec("GPU freq (MHz)")
                _kv("avg",   v.gpu_freq_avg_mhz)
                _kv("peak",  v.gpu_freq_peak_mhz)
                _kv("stddev", v.gpu_freq_stddev_mhz)

        return

    hdr("Request Metrics")
    print(f"{'Total requests:':<50}{report['num_total_requests']}")
    print(f"{'Completed requests:':<50}{report['num_completed_requests']}")
    print(f"{'Duration (s):':<50}{report['duration']:.2f}")
    print(f"{'Request throughput (req/s):':<50}{report['request_throughput']:.2f}")

    if any(k in report for k in ("latency_ms", "ttft_ms", "tpot_ms", "itl_ms")):
        sec("Latency (ms)")
        print(f"{'metric':<10} {'mean':>8} {'p50':>8} {'p95':>8} {'max':>8}")
        for k, label in [
            ("latency_ms", "e2e"),
            ("ttft_ms",    "ttft"),
            ("tpot_ms",    "tpot"),
            ("itl_ms",     "itl"),
        ]:
            if k not in report:
                continue
            s = report[k]
            print(
                f"{label:<12}"
                f"{s['mean']:>9.2f}"
                f"{s['quantiles']['p50']:>9.2f}"
                f"{s['quantiles']['p95']:>9.2f}"
                f"{s['max']:>9.2f}"
            )

    if any(k in report for k in ("input_tokens", "output_tokens")):
        sec("Token stats")
        print(f"{'metric':<10} {'mean':>8} {'p50':>8} {'max':>8}")
        for k, label in [
            ("input_tokens",  "input"),
            ("output_tokens", "output"),
        ]:
            if k not in report:
                continue
            s = report[k]
            print(
                f"{label:<12}"
                f"{s['mean']:>9.1f}"
                f"{s['quantiles']['p50']:>9.1f}"
                f"{s['max']:>9}"
            )

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

