from __future__ import annotations

import gc
import threading

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .analysis_metrics import (
    compute_activation_kurtosis,
    compute_attention_entropy_metrics,
    compute_logit_shift_rms,
    compute_participation_ratio,
    compute_tensor_variance,
    compute_top_energy_share,
)
from .adapters import DEFAULT_PROMPTS, ModelAdapter, ModelResidencyManager
from .capture import (
    STAGE_SPECS,
    build_flow_steps,
    build_stage_hooks,
    top_logit_tokens,
)
from .types import (
    ActivationMetrics,
    AttentionMetrics,
    ContributionMetrics,
    FlowAnalysisResult,
    FlowStep,
    LayerAnalysis,
    LogitToken,
    ModelInfo,
)


TL_STAGE_SEQUENCE = [stage_id for stage_id, _stage_label in STAGE_SPECS]


class Gemma3Adapter(ModelAdapter):
    model_id = "google/gemma-3-4b-it"
    label = "Gemma 3 4B IT"

    def __init__(self, residency: ModelResidencyManager) -> None:
        self._lock = threading.Lock()
        self._residency = residency
        self._model = None
        self._tokenizer = None
        self._analysis_attention_enabled = False
        config = AutoConfig.from_pretrained(self.model_id)
        text_config = config.text_config
        self._max_length = getattr(text_config, "max_position_embeddings", None)
        self._model_info = ModelInfo(
            id=self.model_id,
            label=self.label,
            layer_count=int(text_config.num_hidden_layers),
            layer_width=int(text_config.hidden_size),
            stage_sequence=TL_STAGE_SEQUENCE,
            prompt_mode="chat",
            default_prompt=DEFAULT_PROMPTS["chat"],
        )
        residency.register(self.model_id, self.release)

    def model_info(self) -> ModelInfo:
        return self._model_info

    def architecture_text(self) -> str:
        self._residency.activate(self.model_id)
        with self._lock:
            self._ensure_loaded()
            return str(self._model)

    def analyze_prompt(
        self,
        prompt: str,
        include_special_tokens: bool = False,
        include_layer_analysis: bool = True,
    ) -> FlowAnalysisResult:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("Prompt must not be empty.")

        self._residency.activate(self.model_id)
        with self._lock:
            self._ensure_loaded()
            if include_layer_analysis:
                self._ensure_attention_outputs_enabled()

            tokenized = self._tokenize_prompt(clean_prompt)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            positions, token_limit_applied = self._visible_positions(
                prompt=clean_prompt,
                input_ids=input_ids[0],
                attention_mask=attention_mask[0],
                include_special_tokens=include_special_tokens,
            )
            if not positions:
                raise ValueError("Prompt did not produce any visible tokens.")

            visible_ids = input_ids[0, positions].detach().cpu().tolist()
            tokens = [self._display_token(token_id) for token_id in visible_ids]
            positions_tensor = torch.tensor(positions, dtype=torch.long)
            sink: dict[int, dict[str, torch.Tensor]] = {}

            handles = build_stage_hooks(self._embedding_module(), self._layers(), sink)
            try:
                with torch.inference_mode():
                    outputs = self._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=include_layer_analysis,
                        logits_to_keep=1,
                    )
            finally:
                for handle in handles:
                    handle.remove()

            steps = build_flow_steps(
                sink=sink,
                positions=positions_tensor,
                hidden_width=self._model_info.layer_width,
                step_factory=FlowStep,
            )
            target_position = int(attention_mask[0].sum().item()) - 1
            target_token_id = int(input_ids[0, target_position].item())
            layer_analysis = []
            if include_layer_analysis:
                layer_analysis = self._build_layer_analysis(
                    sink=sink,
                    attentions=outputs.attentions,
                    positions=positions,
                    target_position=target_position,
                )

        return FlowAnalysisResult(
            model=self._model_info,
            tokens=tokens,
            hidden_width=self._model_info.layer_width,
            token_limit=len(tokens),
            token_limit_applied=token_limit_applied,
            steps=steps,
            target_position=target_position,
            target_token_id=target_token_id,
            target_token=self._display_token(target_token_id),
            layer_analysis=layer_analysis,
        )

    def release(self) -> None:
        with self._lock:
            if self._model is None and self._tokenizer is None:
                return
            self._model = None
            self._tokenizer = None
            self._analysis_attention_enabled = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        self._analysis_attention_enabled = False
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"
        self._strip_vision_modules()

    def _ensure_attention_outputs_enabled(self) -> None:
        if self._analysis_attention_enabled:
            return
        if hasattr(self._model, "set_attn_implementation"):
            self._model.set_attn_implementation("eager")
        self._analysis_attention_enabled = True

    def _layers(self):
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            return self._model.model.layers
        if hasattr(self._model, "model") and hasattr(self._model.model, "language_model"):
            return self._model.model.language_model.layers
        raise RuntimeError("Could not locate model layers for activation hooks.")

    def _embedding_module(self):
        if hasattr(self._model, "model") and hasattr(self._model.model, "embed_tokens"):
            return self._model.model.embed_tokens
        if hasattr(self._model, "model") and hasattr(self._model.model, "language_model"):
            language_model = self._model.model.language_model
            if hasattr(language_model, "embed_tokens"):
                return language_model.embed_tokens
        raise RuntimeError("Could not locate token embedding module for activation hooks.")

    def _format_chat(self, prompt: str) -> str:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    def _tokenize_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        return self._tokenizer(
            [self._format_chat(prompt)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
            add_special_tokens=True,
        )

    def _visible_positions(
        self,
        prompt: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        include_special_tokens: bool,
    ) -> tuple[list[int], bool]:
        if include_special_tokens:
            visible_positions = list(range(int(attention_mask.sum().item())))
        else:
            prefix_ids = self._tokenizer("<start_of_turn>user\n", add_special_tokens=False)["input_ids"]
            prompt_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
            start = 1 + len(prefix_ids)
            available = max(int(input_ids.shape[0]) - start - 5, 0)
            end = start + min(len(prompt_ids), available)
            visible_positions = list(range(start, end))

        token_limit_applied = False
        return visible_positions, token_limit_applied

    def _display_token(self, token_id: int) -> str:
        text = self._tokenizer.decode([token_id], skip_special_tokens=False)
        text = text.replace("\n", "\\n")
        return text if text else " "

    def _build_layer_analysis(
        self,
        sink: dict[int, dict[str, torch.Tensor]],
        attentions,
        positions: list[int],
        target_position: int,
    ) -> list[LayerAnalysis]:
        positions_tensor = torch.tensor(positions, dtype=torch.long)
        layer_analysis: list[LayerAnalysis] = []
        embedding = sink.get(-1, {}).get("embeddings")
        if embedding is None:
            raise RuntimeError("Missing stage 'embeddings' for embedding layer")
        previous_logits = self._project_hidden_to_logits(embedding[0, target_position, :])

        for layer_index in sorted(index for index in sink if index >= 0):
            layer_data = sink[layer_index]
            resid = layer_data.get("resid_after_mlp")
            if resid is None:
                raise RuntimeError(f"Missing stage 'resid_after_mlp' for layer {layer_index}")
            resid_field = resid[0].detach().float().cpu().index_select(0, positions_tensor)
            target_hidden = resid[0, target_position, :]
            current_logits = self._project_hidden_to_logits(target_hidden)
            mean_entropy, sink_mass, sink_head_ratio = self._attention_metrics_for_layer(
                attentions=attentions,
                layer_index=layer_index,
            )

            layer_analysis.append(
                LayerAnalysis(
                    layer_index=layer_index,
                    top_tokens=self._top_tokens_from_logits(current_logits),
                    activation_metrics=ActivationMetrics(
                        layer_variance=round(compute_tensor_variance(resid_field), 6),
                        kurtosis=round(compute_activation_kurtosis(resid_field), 6),
                        top_energy_share=round(compute_top_energy_share(resid_field), 6),
                        participation_ratio=round(compute_participation_ratio(resid_field), 6),
                    ),
                    attention_metrics=AttentionMetrics(
                        mean_entropy=round(mean_entropy, 6),
                        sink_mass=round(sink_mass, 6),
                        sink_head_ratio=round(sink_head_ratio, 6),
                    ),
                    contribution_metrics=ContributionMetrics(
                        logit_shift_rms=round(compute_logit_shift_rms(current_logits, previous_logits), 6),
                    ),
                ),
            )
            previous_logits = current_logits

        return layer_analysis

    def _top_tokens_from_hidden(self, hidden: torch.Tensor) -> list[LogitToken]:
        logits = self._project_hidden_to_logits(hidden)
        return self._top_tokens_from_logits(logits)

    def _attention_metrics_for_layer(
        self,
        attentions,
        layer_index: int,
    ) -> tuple[float, float, float]:
        if attentions is None or len(attentions) <= layer_index:
            return float("nan"), float("nan"), float("nan")

        layer_attention = attentions[layer_index]
        if layer_attention is None:
            return float("nan"), float("nan"), float("nan")

        return compute_attention_entropy_metrics(layer_attention[0])

    def _project_hidden_to_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        lm_head = self._lm_head()
        residual = hidden.unsqueeze(0).unsqueeze(0).to(device=lm_head.weight.device, dtype=lm_head.weight.dtype)
        return lm_head(self._final_norm_module()(residual))[0, 0, :]

    def _top_tokens_from_logits(self, logits: torch.Tensor) -> list[LogitToken]:
        return top_logit_tokens(logits, self._display_token, LogitToken)

    def _final_norm_module(self):
        if hasattr(self._model, "model") and hasattr(self._model.model, "norm"):
            return self._model.model.norm
        if hasattr(self._model, "model") and hasattr(self._model.model, "language_model"):
            language_model = self._model.model.language_model
            if hasattr(language_model, "norm"):
                return language_model.norm
        raise RuntimeError("Could not locate final norm module for logits projection.")

    def _lm_head(self):
        if hasattr(self._model, "lm_head"):
            return self._model.lm_head
        raise RuntimeError("Could not locate lm_head for logits projection.")

    def _strip_vision_modules(self) -> None:
        model = self._model
        self._detach_component(model, "vision_model")
        for attr in ("vision", "image_processor", "visual", "vision_tower"):
            self._detach_component(model, attr)
        if hasattr(model, "model"):
            for attr in ("vision_tower", "multi_modal_projector", "mm_projector", "image_newline"):
                self._detach_component(model.model, attr)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _detach_component(self, parent, attr: str) -> None:
        if not hasattr(parent, attr):
            return

        component = getattr(parent, attr)
        if hasattr(component, "to"):
            try:
                component.to("cpu")
            except Exception:
                pass

        try:
            setattr(parent, attr, None)
        except Exception:
            pass
