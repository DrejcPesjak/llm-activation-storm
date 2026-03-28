from __future__ import annotations

from dataclasses import asdict, dataclass, field


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
class LogitToken:
    token_id: int
    token: str
    logit: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LayerTopTokens:
    layer_index: int
    top_tokens: list[LogitToken]

    def to_dict(self) -> dict:
        return {
            "layer_index": self.layer_index,
            "top_tokens": [token.to_dict() for token in self.top_tokens],
        }


@dataclass(frozen=True)
class FlowAnalysisResult:
    model: ModelInfo
    tokens: list[str]
    hidden_width: int
    token_limit: int
    token_limit_applied: bool
    steps: list[FlowStep]
    target_position: int = -1
    target_token_id: int | None = None
    target_token: str = ""
    layer_top_tokens: list[LayerTopTokens] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "tokens": self.tokens,
            "hidden_width": self.hidden_width,
            "token_limit": self.token_limit,
            "token_limit_applied": self.token_limit_applied,
            "steps": [step.to_dict() for step in self.steps],
            "target_position": self.target_position,
            "target_token_id": self.target_token_id,
            "target_token": self.target_token,
            "layer_top_tokens": [entry.to_dict() for entry in self.layer_top_tokens],
        }
