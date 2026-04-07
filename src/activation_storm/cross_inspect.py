from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


METRIC_GROUPS = (
    (
        "activation",
        "Activation Metrics",
        (
            ("layer_variance", "Layer Variance"),
            ("kurtosis", "Kurtosis"),
            ("top_energy_share", "Top Energy Share"),
            ("participation_ratio", "Participation Ratio"),
        ),
    ),
    (
        "attention",
        "Attention + Sink",
        (
            ("mean_entropy", "Mean Entropy"),
            ("sink_mass", "Sink Mass"),
            ("sink_head_ratio", "Sink Heads"),
        ),
    ),
    (
        "contribution",
        "Depth Utilization",
        (
            ("logit_shift_rms", "Logit Shift RMS"),
        ),
    ),
)

METRIC_SPECS = tuple(metric_spec for _group_id, _group_label, metric_specs in METRIC_GROUPS for metric_spec in metric_specs)
METRIC_KEYS = tuple(metric_key for metric_key, _metric_label in METRIC_SPECS)
DEFAULT_DEPTH_GRID = tuple(index / 100 for index in range(101))


@dataclass(frozen=True)
class CrossInspectRunSummary:
    run_id: str
    timestamp: str
    log_file: str
    model_id: str
    model_label: str
    layer_count: int
    prompt: str
    prompt_preview: str
    target_token: str
    metrics_layer_count: int

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "log_file": self.log_file,
            "model_id": self.model_id,
            "model_label": self.model_label,
            "layer_count": self.layer_count,
            "prompt": self.prompt,
            "prompt_preview": self.prompt_preview,
            "target_token": self.target_token,
            "metrics_layer_count": self.metrics_layer_count,
        }


@dataclass(frozen=True)
class CrossInspectLayerPoint:
    layer_index: int
    relative_depth: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class CrossInspectRunRecord:
    summary: CrossInspectRunSummary
    layer_points: tuple[CrossInspectLayerPoint, ...]
    sort_timestamp: datetime


def _prompt_preview(prompt: str, limit: int = 96) -> str:
    compact = " ".join(prompt.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 1].rstrip()}…"


def _safe_float(value) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _relative_depth(layer_index: int, layer_count: int) -> float:
    if layer_count <= 1:
        return 0.0
    return layer_index / (layer_count - 1)


def _mean(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return 0.0
    return sum(finite_values) / len(finite_values)


def _std(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if len(finite_values) < 2:
        return 0.0
    mean_value = _mean(finite_values)
    variance = sum((value - mean_value) ** 2 for value in finite_values) / len(finite_values)
    return math.sqrt(variance)


def _interpolate_series(points: tuple[CrossInspectLayerPoint, ...], metric_key: str, target_depth: float) -> float:
    if not points:
        return float("nan")

    if target_depth <= points[0].relative_depth:
        return points[0].metrics[metric_key]
    if target_depth >= points[-1].relative_depth:
        return points[-1].metrics[metric_key]

    for left_point, right_point in zip(points, points[1:]):
        if left_point.relative_depth <= target_depth <= right_point.relative_depth:
            left_depth = left_point.relative_depth
            right_depth = right_point.relative_depth
            left_value = left_point.metrics[metric_key]
            right_value = right_point.metrics[metric_key]
            if right_depth <= left_depth:
                return right_value
            ratio = (target_depth - left_depth) / (right_depth - left_depth)
            return left_value + (right_value - left_value) * ratio

    return points[-1].metrics[metric_key]


class CrossInspectStore:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)

    def list_runs(self) -> list[CrossInspectRunSummary]:
        return [record.summary for record in self.load_run_records()]

    def analyze(self, payload: dict) -> dict:
        mode = payload.get("mode", "")
        records = {record.summary.run_id: record for record in self.load_run_records()}

        if mode == "aggregate_selected":
            run_ids = self._dedupe_run_ids(payload.get("run_ids", []))
            selected_records = self._resolve_records(records, run_ids)
            self._validate_group(selected_records, require_non_empty=True)
            return {
                "mode": mode,
                "metric_catalog": self.metric_catalog(),
                "group_a": self._build_group_payload(selected_records),
                "warnings": [],
            }

        if mode == "compare_two_runs":
            run_ids = self._dedupe_run_ids(payload.get("run_ids", []))
            if len(run_ids) != 2:
                raise ValueError("Compare two runs requires exactly 2 selected runs.")
            selected_records = self._resolve_records(records, run_ids)
            self._validate_group(selected_records, require_non_empty=True)
            group_a_records = [selected_records[0]]
            group_b_records = [selected_records[1]]
            group_a = self._build_group_payload(group_a_records)
            group_b = self._build_group_payload(group_b_records)
            return {
                "mode": mode,
                "metric_catalog": self.metric_catalog(),
                "group_a": group_a,
                "group_b": group_b,
                "delta": self._build_delta_payload(group_a["metric_trends"], group_b["metric_trends"]),
                "warnings": [],
            }

        if mode == "compare_groups":
            group_a_ids = self._dedupe_run_ids(payload.get("group_a_run_ids", []))
            group_b_ids = self._dedupe_run_ids(payload.get("group_b_run_ids", []))
            group_a_records = self._resolve_records(records, group_a_ids)
            group_b_records = self._resolve_records(records, group_b_ids)
            self._validate_group(group_a_records, require_non_empty=True, group_label="Group A")
            self._validate_group(group_b_records, require_non_empty=True, group_label="Group B")
            group_a = self._build_group_payload(group_a_records)
            group_b = self._build_group_payload(group_b_records)
            return {
                "mode": mode,
                "metric_catalog": self.metric_catalog(),
                "group_a": group_a,
                "group_b": group_b,
                "delta": self._build_delta_payload(group_a["metric_trends"], group_b["metric_trends"]),
                "warnings": [],
            }

        raise ValueError(f"Unsupported cross-inspect mode: {mode}")

    def load_run_records(self) -> list[CrossInspectRunRecord]:
        records: list[CrossInspectRunRecord] = []
        if not self.log_dir.exists():
            return records

        for log_path in sorted(self.log_dir.glob("metrics_*.jsonl")):
            try:
                lines = log_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line_number, raw_line in enumerate(lines, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    payload = json.loads(raw_line)
                    record = self._parse_run_record(payload=payload, log_path=log_path, line_number=line_number)
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                records.append(record)

        records.sort(key=lambda record: (record.sort_timestamp, record.summary.log_file, record.summary.run_id), reverse=True)
        return records

    @staticmethod
    def metric_catalog() -> list[dict]:
        return [
            {
                "group_id": group_id,
                "group_label": group_label,
                "metrics": [{"key": metric_key, "label": metric_label} for metric_key, metric_label in metric_specs],
            }
            for group_id, group_label, metric_specs in METRIC_GROUPS
        ]

    def _build_group_payload(self, records: list[CrossInspectRunRecord]) -> dict:
        if not records:
            raise ValueError("Cross-Inspect group cannot be empty.")
        metric_trends = self._build_group_metric_trends(records)
        model = records[0].summary
        return {
            "model": {
                "id": model.model_id,
                "label": model.model_label,
                "layer_count": model.layer_count,
            },
            "run_count": len(records),
            "runs": [record.summary.to_dict() for record in records],
            "metric_trends": metric_trends,
            "metric_summaries": self._build_metric_summaries(metric_trends),
        }

    def _build_group_metric_trends(self, records: list[CrossInspectRunRecord]) -> dict[str, list[dict]]:
        metric_trends: dict[str, list[dict]] = {}
        for metric_key in METRIC_KEYS:
            series = []
            for relative_depth in DEFAULT_DEPTH_GRID:
                samples = [_interpolate_series(record.layer_points, metric_key, relative_depth) for record in records]
                series.append(
                    {
                        "relative_depth": round(relative_depth, 4),
                        "mean": round(_mean(samples), 6),
                        "std": round(_std(samples), 6),
                    }
                )
            metric_trends[metric_key] = series
        return metric_trends

    def _build_delta_payload(self, group_a_trends: dict[str, list[dict]], group_b_trends: dict[str, list[dict]]) -> dict:
        metric_trends: dict[str, list[dict]] = {}
        for metric_key in METRIC_KEYS:
            delta_series = []
            for group_a_point, group_b_point in zip(group_a_trends[metric_key], group_b_trends[metric_key]):
                delta_series.append(
                    {
                        "relative_depth": group_a_point["relative_depth"],
                        "mean_delta": round(group_b_point["mean"] - group_a_point["mean"], 6),
                    }
                )
            metric_trends[metric_key] = delta_series
        return {
            "metric_trends": metric_trends,
            "metric_summaries": self._build_delta_summaries(metric_trends),
        }

    def _build_metric_summaries(self, metric_trends: dict[str, list[dict]]) -> dict[str, dict]:
        summaries: dict[str, dict] = {}
        for metric_key, series in metric_trends.items():
            mean_values = [point["mean"] for point in series]
            peak_point = max(series, key=lambda point: point["mean"])
            summaries[metric_key] = {
                "series_mean": round(_mean(mean_values), 6),
                "peak_value": round(peak_point["mean"], 6),
                "peak_depth": round(peak_point["relative_depth"], 4),
                "final_value": round(series[-1]["mean"], 6),
            }
        return summaries

    def _build_delta_summaries(self, metric_trends: dict[str, list[dict]]) -> dict[str, dict]:
        summaries: dict[str, dict] = {}
        for metric_key, series in metric_trends.items():
            mean_values = [point["mean_delta"] for point in series]
            peak_point = max(series, key=lambda point: abs(point["mean_delta"]))
            summaries[metric_key] = {
                "series_mean": round(_mean(mean_values), 6),
                "series_std": round(_std(mean_values), 6),
                "peak_value": round(peak_point["mean_delta"], 6),
                "peak_depth": round(peak_point["relative_depth"], 4),
                "final_value": round(series[-1]["mean_delta"], 6),
            }
        return summaries

    def _parse_run_record(self, payload: dict, log_path: Path, line_number: int) -> CrossInspectRunRecord:
        timestamp = str(payload["timestamp"])
        model = payload["model"]
        prompt = str(payload["request"]["prompt"])
        target_token = str(payload.get("context", {}).get("target_token", ""))
        layer_analysis = payload["metrics"]["layer_analysis"]
        if not isinstance(layer_analysis, list) or not layer_analysis:
            raise ValueError("Missing layer_analysis entries.")

        layer_count = int(model.get("layer_count") or len(layer_analysis))
        points: list[CrossInspectLayerPoint] = []
        for layer_entry in layer_analysis:
            layer_index = int(layer_entry["layer_index"])
            metrics = {
                "layer_variance": _safe_float(layer_entry["activation_metrics"]["layer_variance"]),
                "kurtosis": _safe_float(layer_entry["activation_metrics"]["kurtosis"]),
                "top_energy_share": _safe_float(layer_entry["activation_metrics"]["top_energy_share"]),
                "participation_ratio": _safe_float(layer_entry["activation_metrics"]["participation_ratio"]),
                "mean_entropy": _safe_float(layer_entry["attention_metrics"]["mean_entropy"]),
                "sink_mass": _safe_float(layer_entry["attention_metrics"]["sink_mass"]),
                "sink_head_ratio": _safe_float(layer_entry["attention_metrics"]["sink_head_ratio"]),
                "logit_shift_rms": _safe_float(layer_entry["contribution_metrics"]["logit_shift_rms"]),
            }
            points.append(
                CrossInspectLayerPoint(
                    layer_index=layer_index,
                    relative_depth=_relative_depth(layer_index, len(layer_analysis)),
                    metrics=metrics,
                )
            )
        points.sort(key=lambda point: point.layer_index)

        summary = CrossInspectRunSummary(
            run_id=f"{log_path.name}:{line_number}",
            timestamp=timestamp,
            log_file=log_path.name,
            model_id=str(model["id"]),
            model_label=str(model["label"]),
            layer_count=layer_count,
            prompt=prompt,
            prompt_preview=_prompt_preview(prompt),
            target_token=target_token,
            metrics_layer_count=len(points),
        )
        return CrossInspectRunRecord(
            summary=summary,
            layer_points=tuple(points),
            sort_timestamp=datetime.fromisoformat(timestamp),
        )

    @staticmethod
    def _dedupe_run_ids(run_ids: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for run_id in run_ids:
            if run_id in seen:
                continue
            seen.add(run_id)
            deduped.append(run_id)
        return deduped

    @staticmethod
    def _resolve_records(records: dict[str, CrossInspectRunRecord], run_ids: list[str]) -> list[CrossInspectRunRecord]:
        missing = [run_id for run_id in run_ids if run_id not in records]
        if missing:
            raise ValueError(f"Unknown run_id values: {', '.join(missing)}")
        return [records[run_id] for run_id in run_ids]

    @staticmethod
    def _validate_group(
        records: list[CrossInspectRunRecord],
        *,
        require_non_empty: bool,
        group_label: str = "Selection",
    ) -> None:
        if require_non_empty and not records:
            raise ValueError(f"{group_label} must contain at least one run.")
        model_ids = {record.summary.model_id for record in records}
        if len(model_ids) > 1:
            raise ValueError(f"{group_label} must contain runs from exactly one model.")
