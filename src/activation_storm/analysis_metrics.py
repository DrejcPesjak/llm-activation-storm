from __future__ import annotations

import math

import torch


def compute_target_rms(values: torch.Tensor) -> float:
    """Return the RMS size of one layer's target-token residual vector.

    This is a simple proxy for how large the current residual state is at the
    prediction position, inspired by depth/variance analyses:
    https://arxiv.org/abs/2502.05795
    """

    if values.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(values.float().pow(2))).item())


def compute_activation_kurtosis(values: torch.Tensor) -> float:
    """Measure how heavy-tailed the activations are.

    Higher kurtosis means a few values are much larger than the rest, following
    the outlier-feature measurement used in:
    https://arxiv.org/abs/2405.19279
    """

    flat = values.float().reshape(-1)
    if flat.numel() < 2:
        return 0.0
    centered = flat - flat.mean()
    variance = centered.pow(2).mean()
    if float(variance.item()) <= 1e-12:
        return 0.0
    fourth = centered.pow(4).mean()
    return float((fourth / variance.pow(2)).item())


def compute_top_energy_share(values: torch.Tensor, fraction: float = 0.01) -> float:
    """Measure how much energy is concentrated in the largest activations.

    This is a simple concentration proxy for outlier-channel dominance, inspired
    by activation-outlier studies such as:
    https://arxiv.org/abs/2404.03605
    """

    flat = values.float().reshape(-1).pow(2)
    if flat.numel() == 0:
        return 0.0
    top_k = max(1, math.ceil(flat.numel() * fraction))
    top_energy = torch.topk(flat, k=top_k).values.sum()
    total_energy = flat.sum()
    if float(total_energy.item()) <= 1e-12:
        return 0.0
    return float((top_energy / total_energy).item())


def compute_participation_ratio(values: torch.Tensor) -> float:
    """Return a simple effective-rank style score for the activation field.

    Larger values mean the representation uses more directions instead of
    collapsing into a very small subspace, inspired by over-squashing /
    representational-collapse analyses:
    https://arxiv.org/abs/2406.04267
    """

    matrix = values.float()
    if matrix.ndim == 1:
        matrix = matrix.unsqueeze(0)
    if matrix.numel() == 0:
        return 0.0
    singular_values = torch.linalg.svdvals(matrix)
    power = singular_values.pow(2)
    denominator = power.pow(2).sum()
    if float(denominator.item()) <= 1e-12:
        return 0.0
    numerator = power.sum().pow(2)
    return float((numerator / denominator).item())


def compute_attention_entropy_metrics(attention_row: torch.Tensor) -> tuple[float, float, float]:
    """Return entropy and first-token sink statistics for one attention row.

    The outputs are mean head entropy, mean mass on token 0, and the fraction
    of heads whose top target is token 0, following sink/entropy studies:
    https://proceedings.mlr.press/v202/zhai23a.html
    """

    if attention_row.ndim != 2:
        raise ValueError(f"Expected [heads, src] attention row, got shape {tuple(attention_row.shape)}")
    probs = attention_row.float().clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1)
    sink_mass = probs[:, 0]
    sink_heads = probs.argmax(dim=-1) == 0
    return (
        float(entropy.mean().item()),
        float(sink_mass.mean().item()),
        float(sink_heads.float().mean().item()),
    )


def compute_logit_shift_rms(current_logits: torch.Tensor, previous_logits: torch.Tensor) -> float:
    """Measure how much one layer changes the next-token logits.

    This is a simple layer-contribution proxy built from LogitLens-style logits,
    motivated by the depth-utilization discussion in:
    https://arxiv.org/abs/2502.05795
    """

    if current_logits.shape != previous_logits.shape:
        raise ValueError(
            f"Expected matching logits shapes, got {tuple(current_logits.shape)} and {tuple(previous_logits.shape)}"
        )
    delta = current_logits.float() - previous_logits.float()
    return compute_target_rms(delta)
