from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from .types import FlowAnalysisResult, ModelInfo


@dataclass(frozen=True)
class TLModelSpec:
    model_id: str
    label: str
    prompt_mode: str
    layer_count: int
    layer_width: int
    trust_remote_code: bool = False


TL_MODEL_SPECS = [
    TLModelSpec("gpt2-small", "GPT-2 Small", "base", 12, 768),
    TLModelSpec("gpt2-xl", "GPT-2 XL", "base", 48, 1600),
    # TODO: Pythia doesn't have a real hook_resid_mid stage
    # blocks without assuming a real hook_resid_mid stage.
    # TLModelSpec("pythia-160m", "Pythia 160M", "base", 12, 768),
    # TLModelSpec("pythia-1b", "Pythia 1B", "base", 16, 2048),
    # TLModelSpec("pythia-2.8b", "Pythia 2.8B", "base", 32, 2560),
    TLModelSpec("llama-7b", "LLaMA 7B", "base", 32, 4096),
    TLModelSpec("llama-2-7b", "Llama 2 7B", "base", 32, 4096),
    TLModelSpec("llama-2-7b-chat", "Llama 2 7B Chat", "chat", 32, 4096),
    TLModelSpec("meta-llama/Llama-3.2-1B", "Llama 3.2 1B", "base", 16, 2048),
    TLModelSpec("meta-llama/Llama-3.2-3B", "Llama 3.2 3B", "base", 28, 3072),
    TLModelSpec("meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B Instruct", "chat", 16, 2048),
    TLModelSpec("meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B Instruct", "chat", 28, 3072),
    TLModelSpec("mistral-7b", "Mistral 7B", "base", 32, 4096),
    TLModelSpec("qwen-1.8b", "Qwen 1.8B", "base", 24, 2048, trust_remote_code=True),
    TLModelSpec("qwen-1.8b-chat", "Qwen 1.8B Chat", "chat", 24, 2048, trust_remote_code=True),
    TLModelSpec("qwen3-1.7b", "Qwen3 1.7B", "base", 28, 2048, trust_remote_code=True),
    TLModelSpec("gemma-2-2b-it", "Gemma 2 2B IT", "chat", 26, 2304),
    TLModelSpec("gemma-3-1b-it", "Gemma 3 1B IT", "chat", 26, 1152),
]

DEFAULT_PROMPTS = {
    "base": "The capital of France is",
    "chat": "Who is the best basketball player of all time?",
}


class ModelAdapter:
    model_id: str

    def model_info(self) -> ModelInfo:
        raise NotImplementedError

    def architecture_text(self) -> str:
        raise NotImplementedError

    def analyze_prompt(self, prompt: str, include_special_tokens: bool = False) -> FlowAnalysisResult:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class ModelResidencyManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._release_callbacks: dict[str, Callable[[], None]] = {}
        self._active_model_id: str | None = None

    def register(self, model_id: str, release_callback: Callable[[], None]) -> None:
        self._release_callbacks[model_id] = release_callback

    def activate(self, model_id: str) -> None:
        previous_model_id: str | None = None
        with self._lock:
            if self._active_model_id == model_id:
                return
            previous_model_id = self._active_model_id
            self._active_model_id = model_id

        if previous_model_id is not None:
            release_callback = self._release_callbacks.get(previous_model_id)
            if release_callback is not None:
                release_callback()


from .gemma3_adapter import Gemma3Adapter
from .transformer_lens_adapter import TransformerLensAdapter


def build_registry() -> dict[str, ModelAdapter]:
    residency = ModelResidencyManager()
    gemma_adapter = Gemma3Adapter(residency=residency)
    registry: dict[str, ModelAdapter] = {gemma_adapter.model_id: gemma_adapter}
    for spec in TL_MODEL_SPECS:
        registry[spec.model_id] = TransformerLensAdapter(spec=spec, residency=residency)
    return registry
