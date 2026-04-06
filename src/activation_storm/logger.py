from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .types import FlowAnalysisResult


@dataclass(frozen=True)
class LoggerConfig:
    log_dir: Path = Path("logs")
    enabled: bool = True
    enabled_log_types: frozenset[str] = field(default_factory=lambda: frozenset({"metrics"}))


class JsonlLogWriter:
    def __init__(self, log_type: str, log_dir: Path, session_stamp: str) -> None:
        self.log_type = log_type
        self.path = log_dir / f"{log_type}_{session_stamp}.jsonl"

    def write(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


class RunLogger:
    def __init__(self, config: LoggerConfig, session_stamp: str | None = None) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._session_stamp = session_stamp or datetime.now().strftime("%Y-%m-%d_%H-%M")
        self._writers: dict[str, JsonlLogWriter] = {}
        self._builders: dict[str, Callable[..., dict]] = {
            "metrics": self._build_metrics_record,
        }

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def log_metrics(
        self,
        *,
        prompt: str,
        include_special_tokens: bool,
        result: FlowAnalysisResult,
    ) -> None:
        self.log(
            "metrics",
            prompt=prompt,
            include_special_tokens=include_special_tokens,
            result=result,
        )

    def log(self, log_type: str, **payload) -> None:
        if not self._config.enabled or log_type not in self._config.enabled_log_types:
            return

        builder = self._builders.get(log_type)
        if builder is None:
            return

        record = builder(**payload)
        with self._lock:
            writer = self._writers.get(log_type)
            if writer is None:
                writer = JsonlLogWriter(log_type=log_type, log_dir=self._config.log_dir, session_stamp=self._session_stamp)
                self._writers[log_type] = writer
            writer.write(record)

    def _build_metrics_record(
        self,
        *,
        prompt: str,
        include_special_tokens: bool,
        result: FlowAnalysisResult,
    ) -> dict:
        model = result.model
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "log_type": "metrics",
            "model": {
                "id": model.id,
                "label": model.label,
                "prompt_mode": model.prompt_mode,
                "layer_count": model.layer_count,
                "layer_width": model.layer_width,
                "stage_sequence": model.stage_sequence,
            },
            "request": {
                "prompt": prompt,
                "include_special_tokens": include_special_tokens,
            },
            "context": {
                "tokens": result.tokens,
                "target_position": result.target_position,
                "target_token_id": result.target_token_id,
                "target_token": result.target_token,
            },
            "metrics": {
                "layer_analysis": [entry.to_dict() for entry in result.layer_analysis],
            },
        }
