from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    layer_count: int
    layer_width: int
    stage_sequence: list[str]
    prompt_mode: str = "base"
    default_prompt: str = "The capital of France is"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FlowStep:
    step_index: int
    layer_index: int
    stage_id: str
    stage_label: str
    rows: int
    cols: int
    scale: float
    encoded_field: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FlowAnalysisResult:
    model: ModelInfo
    tokens: list[str]
    hidden_width: int
    token_limit: int
    token_limit_applied: bool
    steps: list[FlowStep]

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "tokens": self.tokens,
            "hidden_width": self.hidden_width,
            "token_limit": self.token_limit,
            "token_limit_applied": self.token_limit_applied,
            "steps": [step.to_dict() for step in self.steps],
        }
