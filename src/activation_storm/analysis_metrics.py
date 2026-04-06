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


def compute_tensor_variance(values: torch.Tensor) -> float:
    """Return the variance of one layer's activation field.

    This is a simple depth-wise variance proxy for whether residual states are
    becoming more extreme as layers stack up:
    https://arxiv.org/abs/2502.05795
    """

    if values.numel() == 0:
        return 0.0
    return float(values.float().var(correction=0).item())


def compute_activation_kurtosis(values: torch.Tensor) -> float:
    """Measure whether a few channels dominate the activation magnitudes.

    This first averages over tokens, then measures kurtosis across channels so
    the score reflects channel-level outliers rather than isolated scalar spikes:
    https://arxiv.org/abs/2405.19279
    """

    matrix = values.float()
    if matrix.ndim == 1:
        channel_magnitudes = matrix.abs()
    else:
        channel_magnitudes = matrix.abs().mean(dim=tuple(range(matrix.ndim - 1)))
    if channel_magnitudes.numel() < 2:
        return 0.0
    centered = channel_magnitudes - channel_magnitudes.mean()
    variance = centered.pow(2).mean()
    if float(variance.item()) <= 1e-12:
        return 0.0
    fourth = centered.pow(4).mean()
    return float((fourth / (variance.pow(2) + 1e-12)).item())


def compute_top_energy_share(values: torch.Tensor, fraction: float = 0.01) -> float:
    """Measure how much channel energy sits in the strongest channels.

    This first averages squared activations into per-channel energy, then asks
    how much of the total is held by the top fraction of channels:
    https://arxiv.org/abs/2404.03605
    """

    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in (0, 1].")
    matrix = values.float()
    if matrix.ndim == 1:
        channel_energy = matrix.pow(2)
    else:
        channel_energy = matrix.pow(2).mean(dim=tuple(range(matrix.ndim - 1)))
    if channel_energy.numel() == 0:
        return 0.0
    top_k = max(1, math.ceil(channel_energy.numel() * fraction))
    top_energy = torch.topk(channel_energy, k=top_k).values.sum()
    total_energy = channel_energy.sum()
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
    if matrix.ndim > 2:
        matrix = matrix.reshape(-1, matrix.shape[-1])
    if matrix.ndim == 1:
        matrix = matrix.unsqueeze(0)
    if matrix.numel() == 0:
        return 0.0
    if matrix.shape[0] > 1:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(matrix)
    power = singular_values.pow(2)
    denominator = power.pow(2).sum()
    if float(denominator.item()) <= 1e-12:
        return 0.0
    numerator = power.sum().pow(2)
    return float((numerator / denominator).item())


def compute_attention_entropy_metrics(attention_probs: torch.Tensor) -> tuple[float, float, float]:
    """Return entropy and first-token sink statistics for one attention tensor.

    The outputs are mean entropy, mean mass on token 0, and the fraction of
    attention rows whose top target is token 0, averaged over heads and query
    positions, following sink/entropy studies:
    https://proceedings.mlr.press/v202/zhai23a.html
    """

    probs = attention_probs.float().clamp_min(1e-12)
    if probs.ndim == 2:
        probs = probs.unsqueeze(1)
    if probs.ndim != 3:
        raise ValueError(f"Expected [heads, query, src] attention tensor, got shape {tuple(probs.shape)}")
    entropy = -(probs * probs.log()).sum(dim=-1)
    sink_mass = probs[..., 0]
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
