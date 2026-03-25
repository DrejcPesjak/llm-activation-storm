from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    layer_count: int
    layer_width: int
    families: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ActivationFrame:
    token_index: int
    token_text: str
    values: dict[str, list[float]]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisResult:
    model: ModelInfo
    tokens: list[str]
    layers: list[str]
    families: list[str]
    frames: list[ActivationFrame]

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "tokens": self.tokens,
            "layers": self.layers,
            "families": self.families,
            "frames": [frame.to_dict() for frame in self.frames],
        }
