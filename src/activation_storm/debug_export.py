from __future__ import annotations

import csv
from pathlib import Path

import torch

from .capture import STAGE_SPECS, select_content_rows, signed_scale

POSITIVE = (94, 234, 212)
NEGATIVE = (249, 168, 212)
HOTSPOT_MAGNITUDE_THRESHOLD = 0.84
HOTSPOT_OFFSET_MOD = 11
HOTSPOT_DRAW_LIMIT = 140
HOTSPOT_STRIDE = 29
HOTSPOT_STAGE_MULTIPLIER = 17


def _lerp_color(base: tuple[int, int, int], magnitude: float) -> tuple[int, int, int]:
    eased = magnitude ** 0.75
    floor = 8
    return tuple(round(floor + (channel - floor) * eased) for channel in base)


def _color_payload(value: float, scale: float) -> tuple[float, int, str, int]:
    norm = max(-1.0, min(1.0, value / scale))
    magnitude = abs(norm)
    base = POSITIVE if norm >= 0 else NEGATIVE
    rgb = _lerp_color(base, magnitude)
    alpha = max(18, round(255 * magnitude))
    color_hex = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    quantized = round((norm + 1.0) * 127.5)
    return norm, alpha, color_hex, quantized


def export_layer_zero_debug(
    *,
    output_dir: Path,
    sink: dict[int, dict[str, torch.Tensor]],
    positions: torch.Tensor,
    tokens: list[str],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_index = 0
    if layer_index not in sink:
        raise RuntimeError('Layer 0 activations were not captured.')

    layer_data = sink[layer_index]
    files = []
    plotted_rows = []

    for stage_offset, (stage_id, _stage_label) in enumerate(STAGE_SPECS):
        if stage_id not in layer_data:
            raise RuntimeError(f"Missing stage {stage_id!r} for layer 0")
        field = select_content_rows(layer_data[stage_id], positions)
        scale = signed_scale(field)
        edge_path = output_dir / f'layer00_{stage_id}_edge_columns.csv'
        _write_edge_columns(edge_path=edge_path, field=field, tokens=tokens)
        files.append(str(edge_path))
        plotted_rows.extend(
            _build_hotspot_rows(
                stage_id=stage_id,
                step_index=stage_offset,
                field=field,
                tokens=tokens,
                scale=scale,
            )
        )

    plotted_path = output_dir / 'layer00_plotted_points.csv'
    _write_plotted_rows(plotted_path=plotted_path, rows=plotted_rows)
    files.append(str(plotted_path))
    return {'output_dir': str(output_dir), 'files': files}


def _write_edge_columns(*, edge_path: Path, field: torch.Tensor, tokens: list[str]) -> None:
    last_dim = int(field.shape[1]) - 1
    with edge_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['token_index', 'token', 'dim_0', f'dim_{last_dim}'])
        writer.writeheader()
        for token_index, token in enumerate(tokens):
            writer.writerow(
                {
                    'token_index': token_index,
                    'token': token,
                    'dim_0': float(field[token_index, 0].item()),
                    f'dim_{last_dim}': float(field[token_index, last_dim].item()),
                }
            )


def _candidate_hotspots(*, field: torch.Tensor, scale: float) -> list[dict]:
    rows = []
    cols = int(field.shape[1])
    for token_index in range(int(field.shape[0])):
        for dim_index in range(cols):
            value = float(field[token_index, dim_index].item())
            norm, alpha, color_hex, quantized = _color_payload(value, scale)
            magnitude = abs(norm)
            offset = token_index * cols + dim_index
            if magnitude > HOTSPOT_MAGNITUDE_THRESHOLD and offset % HOTSPOT_OFFSET_MOD == 0:
                rows.append(
                    {
                        'token_index': token_index,
                        'dim_index': dim_index,
                        'value': value,
                        'normalized_value': round(norm, 6),
                        'magnitude': round(magnitude, 6),
                        'color_hex': color_hex,
                        'alpha': alpha,
                        'quantized_byte': quantized,
                        'radius': round(1.2 + magnitude * 3.8, 6),
                    }
                )
    return rows


def _build_hotspot_rows(*, stage_id: str, step_index: int, field: torch.Tensor, tokens: list[str], scale: float) -> list[dict]:
    candidates = _candidate_hotspots(field=field, scale=scale)
    if not candidates:
        return []

    selected = []
    count = min(HOTSPOT_DRAW_LIMIT, len(candidates))
    last_token = max(len(tokens) - 1, 1)
    last_dim = max(int(field.shape[1]) - 1, 1)

    for index in range(count):
        hotspot = candidates[(index * HOTSPOT_STRIDE + step_index * HOTSPOT_STAGE_MULTIPLIER) % len(candidates)]
        token_index = hotspot['token_index']
        dim_index = hotspot['dim_index']
        selected.append(
            {
                'stage_id': stage_id,
                'step_index': step_index,
                'draw_order': index,
                'token_index': token_index,
                'token': tokens[token_index],
                'token_pct': round(token_index / last_token, 6),
                'dim_index': dim_index,
                'dim_pct': round(dim_index / last_dim, 6),
                'value': hotspot['value'],
                'normalized_value': hotspot['normalized_value'],
                'magnitude': hotspot['magnitude'],
                'color_hex': hotspot['color_hex'],
                'alpha': hotspot['alpha'],
                'quantized_byte': hotspot['quantized_byte'],
                'radius': hotspot['radius'],
            }
        )
    return selected


def _write_plotted_rows(*, plotted_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        'stage_id',
        'step_index',
        'draw_order',
        'token_index',
        'token',
        'token_pct',
        'dim_index',
        'dim_pct',
        'value',
        'normalized_value',
        'magnitude',
        'color_hex',
        'alpha',
        'quantized_byte',
        'radius',
    ]
    with plotted_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
