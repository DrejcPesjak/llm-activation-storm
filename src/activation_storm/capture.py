from __future__ import annotations

import base64
from collections.abc import Callable

import torch


EMBEDDING_STAGE_SPEC = ("embeddings", "EMB")
LAYER_STAGE_SPECS = (
    ("attn_out", "ATTN"),
    ("resid_after_attn", "RESID"),
    ("mlp_out", "MLP"),
    ("resid_after_mlp", "RESID"),
)
STAGE_SPECS = (EMBEDDING_STAGE_SPEC, *LAYER_STAGE_SPECS)


def unwrap_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _detach_hidden(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu()


def build_stage_hooks(embed_tokens, layers, sink: dict[int, dict[str, torch.Tensor]]):
    handles = []

    def store_output(layer_index: int, stage_id: str):
        def hook(_module, _inputs, output):
            sink[layer_index][stage_id] = _detach_hidden(unwrap_tensor(output))
            return output

        return hook

    def store_input(layer_index: int, stage_id: str):
        def hook(_module, inputs):
            sink[layer_index][stage_id] = _detach_hidden(inputs[0])
            return None

        return hook

    sink[-1] = {}
    handles.append(embed_tokens.register_forward_hook(store_output(-1, "embeddings")))

    for layer_index, layer in enumerate(layers):
        sink[layer_index] = {}
        handles.append(layer.post_attention_layernorm.register_forward_hook(store_output(layer_index, "attn_out")))
        handles.append(
            layer.pre_feedforward_layernorm.register_forward_pre_hook(
                store_input(layer_index, "resid_after_attn")
            )
        )
        handles.append(layer.post_feedforward_layernorm.register_forward_hook(store_output(layer_index, "mlp_out")))
        handles.append(layer.register_forward_hook(store_output(layer_index, "resid_after_mlp")))

    return handles


def select_content_rows(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError(f"Expected [batch, seq, hidden] tensor, got shape {tuple(values.shape)}")
    return values[0].index_select(0, positions)


def signed_scale(values: torch.Tensor, quantile: float = 0.995) -> float:
    flat = values.abs().reshape(-1)
    if flat.numel() == 0:
        return 1.0
    scale = float(torch.quantile(flat, quantile).item())
    return scale if scale > 1e-6 else 1.0


def encode_signed_field(values: torch.Tensor, scale: float) -> str:
    normalized = torch.clamp(values / scale, -1.0, 1.0)
    quantized = torch.round((normalized + 1.0) * 127.5).to(torch.uint8)
    return base64.b64encode(quantized.numpy().tobytes()).decode("ascii")


def apply_logit_soft_cap(logits: torch.Tensor, soft_cap: float | None) -> torch.Tensor:
    if soft_cap is None or soft_cap <= 0.0:
        return logits
    capped = logits / soft_cap
    capped = torch.tanh(capped)
    return capped * soft_cap


def top_logit_tokens(
    logits: torch.Tensor,
    decode_token: Callable[[int], str],
    token_factory: Callable[..., object],
    limit: int = 10,
) -> list[object]:
    if logits.ndim != 1:
        raise ValueError(f"Expected [vocab] logits tensor, got shape {tuple(logits.shape)}")

    top_k = min(limit, int(logits.shape[0]))
    if top_k <= 0:
        return []

    values, indices = torch.topk(logits.float(), k=top_k)
    return [
        token_factory(
            token_id=int(token_id),
            token=decode_token(int(token_id)),
            logit=round(float(logit), 6),
        )
        for logit, token_id in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist())
    ]


def build_flow_steps(
    sink: dict[int, dict[str, torch.Tensor]],
    positions: torch.Tensor,
    hidden_width: int,
    step_factory: Callable[..., object],
) -> list[object]:
    steps = []
    step_index = 0

    if -1 in sink:
        embedding_field = sink[-1].get("embeddings")
        if embedding_field is None:
            raise RuntimeError("Missing stage 'embeddings' for embedding layer")
        field = select_content_rows(embedding_field, positions)
        scale = signed_scale(field)
        steps.append(
            step_factory(
                step_index=step_index,
                layer_index=-1,
                stage_id="embeddings",
                stage_label="EMB",
                rows=int(field.shape[0]),
                cols=hidden_width,
                scale=round(scale, 6),
                encoded_field=encode_signed_field(field, scale),
            )
        )
        step_index += 1

    for layer_index in sorted(index for index in sink if index >= 0):
        layer_data = sink[layer_index]
        for stage_id, stage_label in LAYER_STAGE_SPECS:
            if stage_id not in layer_data:
                raise RuntimeError(f"Missing stage '{stage_id}' for layer {layer_index}")
            field = select_content_rows(layer_data[stage_id], positions)
            scale = signed_scale(field)
            steps.append(
                step_factory(
                    step_index=step_index,
                    layer_index=layer_index,
                    stage_id=stage_id,
                    stage_label=stage_label,
                    rows=int(field.shape[0]),
                    cols=hidden_width,
                    scale=round(scale, 6),
                    encoded_field=encode_signed_field(field, scale),
                )
            )
            step_index += 1

    return steps
