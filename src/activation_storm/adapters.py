from __future__ import annotations

import gc
import threading
from abc import ABC, abstractmethod

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .capture import FAMILIES, build_layer_hooks, ease_matrix, normalize_family_values
from .types import ActivationFrame, AnalysisResult, ModelInfo


class ModelAdapter(ABC):
    @abstractmethod
    def model_info(self) -> ModelInfo:
        raise NotImplementedError

    @abstractmethod
    def analyze_prompt(self, prompt: str) -> AnalysisResult:
        raise NotImplementedError


class Gemma3Adapter(ModelAdapter):
    model_id = "google/gemma-3-4b-it"
    label = "Gemma 3 4B IT"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        config = AutoConfig.from_pretrained(self.model_id)
        text_config = config.text_config
        self._model_info = ModelInfo(
            id=self.model_id,
            label=self.label,
            layer_count=int(text_config.num_hidden_layers),
            layer_width=int(text_config.hidden_size),
            families=list(FAMILIES),
        )

    def model_info(self) -> ModelInfo:
        return self._model_info

    def analyze_prompt(self, prompt: str) -> AnalysisResult:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("Prompt must not be empty.")

        with self._lock:
            self._ensure_loaded()

            tokenized = self._tokenize_prompt(clean_prompt)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            content_positions = self._content_positions(clean_prompt, input_ids[0])
            if not content_positions:
                raise ValueError("Prompt did not produce any content tokens.")

            visible_ids = input_ids[0, content_positions].detach().cpu().tolist()
            tokens = [self._display_token(token_id) for token_id in visible_ids]
            sink = {family: {} for family in FAMILIES}

            handles = build_layer_hooks(self._layers(), sink)
            try:
                with torch.inference_mode():
                    self._model(input_ids=input_ids, attention_mask=attention_mask)
            finally:
                for handle in handles:
                    handle.remove()

            token_count = len(content_positions)
            positions = torch.tensor(content_positions, dtype=torch.long)
            filtered = {
                family: {
                    layer_index: family_values[0].index_select(0, positions)
                    for layer_index, family_values in by_layer.items()
                }
                for family, by_layer in sink.items()
            }
            normalized = normalize_family_values(
                filtered,
                token_count=token_count,
                layer_count=self._model_info.layer_count,
            )
            softened = {family: ease_matrix(matrix) for family, matrix in normalized.items()}

        frames = [
            ActivationFrame(
                token_index=token_index,
                token_text=tokens[token_index],
                values={family: softened[family][token_index] for family in FAMILIES},
            )
            for token_index in range(token_count)
        ]
        layers = [f"L{index:02d}" for index in range(self._model_info.layer_count)]
        return AnalysisResult(
            model=self._model_info,
            tokens=tokens,
            layers=layers,
            families=list(FAMILIES),
            frames=frames,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"
        self._strip_vision_modules()

    def _layers(self):
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            return self._model.model.layers
        if hasattr(self._model, "model") and hasattr(self._model.model, "language_model"):
            return self._model.model.language_model.layers
        raise RuntimeError("Could not locate model layers for activation hooks.")

    def _format_chat(self, prompt: str) -> str:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    def _tokenize_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        return self._tokenizer(
            [self._format_chat(prompt)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )

    def _content_positions(self, prompt: str, input_ids: torch.Tensor) -> list[int]:
        prefix_ids = self._tokenizer("<start_of_turn>user\n", add_special_tokens=False)["input_ids"]
        prompt_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        start = 1 + len(prefix_ids)
        end = min(start + len(prompt_ids), int(input_ids.shape[0]))
        return list(range(start, end))

    def _display_token(self, token_id: int) -> str:
        text = self._tokenizer.decode([token_id], skip_special_tokens=False)
        text = text.replace("\n", "\\n")
        return text if text else " "

    def _strip_vision_modules(self) -> None:
        model = self._model

        if hasattr(model, "vision_model"):
            del model.vision_model
        for attr in ("vision", "image_processor", "visual", "vision_tower"):
            if hasattr(model, attr):
                delattr(model, attr)
        if hasattr(model, "model"):
            for attr in ("vision_tower", "mm_projector", "image_newline"):
                if hasattr(model.model, attr):
                    delattr(model.model, attr)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_registry() -> dict[str, ModelAdapter]:
    adapter = Gemma3Adapter()
    return {adapter.model_id: adapter}
