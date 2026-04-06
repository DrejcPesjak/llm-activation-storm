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
class ActivationMetrics:
    layer_variance: float
    kurtosis: float
    top_energy_share: float
    participation_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AttentionMetrics:
    mean_entropy: float
    sink_mass: float
    sink_head_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ContributionMetrics:
    logit_shift_rms: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LayerAnalysis:
    layer_index: int
    top_tokens: list[LogitToken]
    activation_metrics: ActivationMetrics
    attention_metrics: AttentionMetrics
    contribution_metrics: ContributionMetrics

    def to_dict(self) -> dict:
        return {
            "layer_index": self.layer_index,
            "top_tokens": [token.to_dict() for token in self.top_tokens],
            "activation_metrics": self.activation_metrics.to_dict(),
            "attention_metrics": self.attention_metrics.to_dict(),
            "contribution_metrics": self.contribution_metrics.to_dict(),
        }


@dataclass(frozen=True)
class FlowAnalysisResult:
    model: ModelInfo
    tokens: list[str]
    hidden_width: int
    token_limit: int
    token_limit_applied: bool
    steps: list[FlowStep]
    visible_token_mask: list[bool] = field(default_factory=list)
    target_position: int = -1
    target_token_id: int | None = None
    target_token: str = ""
    layer_analysis: list[LayerAnalysis] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "tokens": self.tokens,
            "visible_token_mask": self.visible_token_mask,
            "hidden_width": self.hidden_width,
            "token_limit": self.token_limit,
            "token_limit_applied": self.token_limit_applied,
            "steps": [step.to_dict() for step in self.steps],
            "target_position": self.target_position,
            "target_token_id": self.target_token_id,
            "target_token": self.target_token,
            "layer_analysis": [entry.to_dict() for entry in self.layer_analysis],
        }
