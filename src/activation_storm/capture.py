from __future__ import annotations

from collections.abc import Callable

import torch


FAMILIES = ("resid", "attn", "mlp")


def tensor_rms(values: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(values.float() * values.float(), dim=-1))


def unwrap_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def build_layer_hooks(
    layers,
    sink: dict[str, dict[int, torch.Tensor]],
    reducer: Callable[[torch.Tensor], torch.Tensor] = tensor_rms,
):
    handles = []

    def record(family: str, layer_index: int):
        def hook(_module, _inputs, output):
            tensor = unwrap_tensor(output)
            sink[family][layer_index] = reducer(tensor).detach().cpu()
            return output

        return hook

    for layer_index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(record("resid", layer_index)))
        handles.append(layer.self_attn.register_forward_hook(record("attn", layer_index)))
        handles.append(layer.mlp.register_forward_hook(record("mlp", layer_index)))

    return handles


def normalize_family_values(
    family_by_layer: dict[str, dict[int, torch.Tensor]],
    token_count: int,
    layer_count: int,
) -> dict[str, list[list[float]]]:
    normalized: dict[str, list[list[float]]] = {}

    for family, by_layer in family_by_layer.items():
        matrix = torch.zeros(token_count, layer_count, dtype=torch.float32)
        for layer_index, values in by_layer.items():
            clipped = values[:token_count].float()
            matrix[: clipped.shape[0], layer_index] = clipped

        max_value = float(matrix.max().item()) if matrix.numel() else 0.0
        if max_value > 0:
            matrix = matrix / max_value

        normalized[family] = [
            [round(float(matrix[token_index, layer_index].item()), 4) for layer_index in range(layer_count)]
            for token_index in range(token_count)
        ]

    return normalized


def smoothstep(value: float) -> float:
    clamped = max(0.0, min(1.0, value))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def ease_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [[round(smoothstep(value), 4) for value in row] for row in matrix]
