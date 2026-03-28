from __future__ import annotations

import gc
import threading

import torch
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_official_model_name

from .adapters import DEFAULT_PROMPTS, ModelAdapter, ModelResidencyManager, TLModelSpec
from .capture import STAGE_SPECS, encode_signed_field, signed_scale
from .types import FlowAnalysisResult, FlowStep, ModelInfo


TL_STAGE_SEQUENCE = [stage_id for stage_id, _stage_label in STAGE_SPECS]
TL_STAGE_LABELS = {
    "embeddings": "EMB",
    "attn_out": "ATTN",
    "resid_after_attn": "RESID",
    "mlp_out": "MLP",
    "resid_after_mlp": "RESID",
}
PARALLEL_TL_STAGE_SEQUENCE = ["embeddings", "attn_out", "mlp_out", "resid_after_mlp"]
SEQUENTIAL_LAYER_STAGE_HOOKS = {
    "attn_out": "hook_attn_out",
    "resid_after_attn": "hook_resid_mid",
    "mlp_out": "hook_mlp_out",
    "resid_after_mlp": "hook_resid_post",
}
PARALLEL_LAYER_STAGE_HOOKS = {
    "attn_out": "hook_attn_out",
    "mlp_out": "hook_mlp_out",
    "resid_after_mlp": "hook_resid_post",
}


class TransformerLensAdapter(ModelAdapter):
    def __init__(self, spec: TLModelSpec, residency: ModelResidencyManager) -> None:
        self._lock = threading.Lock()
        self._residency = residency
        self._spec = spec
        self.model_id = spec.model_id
        self._official_model_name = get_official_model_name(spec.model_id)
        self._model: HookedTransformer | None = None
        self._model_info: ModelInfo | None = None
        residency.register(self.model_id, self.release)

    def model_info(self) -> ModelInfo:
        if self._model_info is None:
            self._prime_model_info()
        return self._model_info

    def architecture_text(self) -> str:
        self._residency.activate(self.model_id)
        with self._lock:
            self._ensure_loaded()
            return f"{self._spec.label}\n{self._official_model_name}\n\n{self._model}"

    def analyze_prompt(self, prompt: str, include_special_tokens: bool = False) -> FlowAnalysisResult:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("Prompt must not be empty.")

        self._residency.activate(self.model_id)
        with self._lock:
            self._ensure_loaded()
            rendered_prompt = self._render_prompt(clean_prompt)
            tokens = self._model.to_tokens(rendered_prompt, move_to_device=True)
            if tokens.shape[0] != 1:
                raise RuntimeError("Expected a single prompt batch.")

            token_ids = tokens[0].detach().cpu().tolist()
            positions = self._visible_positions(
                rendered_prompt=rendered_prompt,
                prompt=clean_prompt,
                token_ids=token_ids,
                include_special_tokens=include_special_tokens,
            )
            if not positions:
                raise ValueError("Prompt did not produce any visible tokens.")

            cache_names = self._cache_names()
            with torch.inference_mode():
                _, cache = self._model.run_with_cache(
                    tokens,
                    names_filter=cache_names,
                    return_cache_object=False,
                )

            visible_ids = [token_ids[index] for index in positions]
            visible_tokens = [self._display_token(token_id) for token_id in visible_ids]
            steps = self._build_steps_from_cache(cache=cache, positions=positions)

        model_info = self.model_info()
        return FlowAnalysisResult(
            model=model_info,
            tokens=visible_tokens,
            hidden_width=model_info.layer_width,
            token_limit=len(visible_tokens),
            token_limit_applied=False,
            steps=steps,
        )

    def release(self) -> None:
        with self._lock:
            if self._model is None:
                return
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prime_model_info(self) -> None:
        self._model_info = ModelInfo(
            id=self.model_id,
            label=self._spec.label,
            layer_count=self._spec.layer_count,
            layer_width=self._spec.layer_width,
            stage_sequence=self._stage_sequence(),
            prompt_mode=self._spec.prompt_mode,
            default_prompt=DEFAULT_PROMPTS[self._spec.prompt_mode],
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = HookedTransformer.from_pretrained_no_processing(
            self.model_id,
            device=device,
            dtype=dtype,
            trust_remote_code=self._spec.trust_remote_code,
        )

    def _render_prompt(self, prompt: str) -> str:
        if self._spec.prompt_mode == "base":
            return prompt

        tokenizer = self._model.tokenizer
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError(f"{self.model_id} requires a tokenizer chat template but none is available.")

        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _visible_positions(
        self,
        rendered_prompt: str,
        prompt: str,
        token_ids: list[int],
        include_special_tokens: bool,
    ) -> list[int]:
        if include_special_tokens:
            return list(range(len(token_ids)))

        if prompt not in rendered_prompt:
            raise RuntimeError(f"Could not locate prompt text inside rendered prompt for {self.model_id}.")

        prefix_text, _suffix_text = rendered_prompt.split(prompt, 1)
        start = len(self._token_ids_for_text(prefix_text))
        end = len(self._token_ids_for_text(prefix_text + prompt))
        if end <= start:
            return []
        if end > len(token_ids):
            raise RuntimeError(f"Computed prompt span exceeds rendered token sequence for {self.model_id}.")
        return list(range(start, end))

    def _cache_names(self) -> list[str]:
        model_info = self.model_info()
        names = ["hook_embed"]
        layer_stage_hooks = self._layer_stage_hooks()
        for layer_index in range(model_info.layer_count):
            names.extend(
                f"blocks.{layer_index}.{hook_name}" for hook_name in layer_stage_hooks.values()
            )
        return names

    def _build_steps_from_cache(self, cache: dict[str, torch.Tensor], positions: list[int]) -> list[FlowStep]:
        model_info = self.model_info()
        position_tensor = torch.tensor(positions, dtype=torch.long)
        steps: list[FlowStep] = []
        step_index = 0

        embedding = cache.get("hook_embed")
        if embedding is None:
            raise RuntimeError(f"Missing hook_embed in cache for {self.model_id}")
        steps.append(
            self._build_step(
                step_index=step_index,
                layer_index=-1,
                stage_id="embeddings",
                tensor=embedding,
                positions=position_tensor,
            )
        )
        step_index += 1

        layer_stage_hooks = self._layer_stage_hooks()
        for layer_index in range(model_info.layer_count):
            for stage_id in model_info.stage_sequence:
                if stage_id == "embeddings":
                    continue
                hook_name = f"blocks.{layer_index}.{layer_stage_hooks[stage_id]}"
                tensor = cache.get(hook_name)
                if tensor is None:
                    raise RuntimeError(f"Missing {hook_name} in cache for {self.model_id}")
                steps.append(
                    self._build_step(
                        step_index=step_index,
                        layer_index=layer_index,
                        stage_id=stage_id,
                        tensor=tensor,
                        positions=position_tensor,
                    )
                )
                step_index += 1

        return steps

    def _build_step(
        self,
        step_index: int,
        layer_index: int,
        stage_id: str,
        tensor: torch.Tensor,
        positions: torch.Tensor,
    ) -> FlowStep:
        if tensor.ndim != 3:
            raise RuntimeError(f"Expected [batch, seq, hidden] tensor for {stage_id}, got {tuple(tensor.shape)}")

        field = tensor[0].detach().float().cpu().index_select(0, positions)
        scale = signed_scale(field)
        return FlowStep(
            step_index=step_index,
            layer_index=layer_index,
            stage_id=stage_id,
            stage_label=TL_STAGE_LABELS[stage_id],
            rows=int(field.shape[0]),
            cols=int(field.shape[1]),
            scale=round(scale, 6),
            encoded_field=encode_signed_field(field, scale),
        )

    def _display_token(self, token_id: int) -> str:
        text = self._model.to_string([token_id])
        text = text.replace("\n", "\\n")
        return text if text else " "

    def _token_ids_for_text(self, text: str) -> list[int]:
        return self._model.to_tokens(text, move_to_device=False)[0].detach().cpu().tolist()

    def _stage_sequence(self) -> list[str]:
        if self._spec.parallel_attn_mlp:
            return PARALLEL_TL_STAGE_SEQUENCE.copy()
        return TL_STAGE_SEQUENCE.copy()

    def _layer_stage_hooks(self) -> dict[str, str]:
        if self._spec.parallel_attn_mlp:
            return PARALLEL_LAYER_STAGE_HOOKS
        return SEQUENTIAL_LAYER_STAGE_HOOKS


def find_subsequence(sequence: list[int], subsequence: list[int]) -> int | None:
    if not subsequence:
        return 0
    last_start = len(sequence) - len(subsequence)
    for start in range(max(last_start + 1, 0)):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    return None
