from __future__ import annotations

import base64
import unittest

import torch

from activation_storm.analysis_metrics import (
    compute_activation_kurtosis,
    compute_attention_entropy_metrics,
    compute_logit_shift_rms,
    compute_participation_ratio,
    compute_target_rms,
    compute_top_energy_share,
)
from activation_storm.capture import (
    apply_logit_soft_cap,
    build_flow_steps,
    encode_signed_field,
    select_content_rows,
    signed_scale,
    top_logit_tokens,
)
from activation_storm.types import FlowStep, LogitToken


class CaptureTests(unittest.TestCase):
    def test_select_content_rows_uses_sequence_dimension(self):
        values = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
        positions = torch.tensor([0, 2], dtype=torch.long)
        selected = select_content_rows(values, positions)
        self.assertEqual(list(selected.shape), [2, 8])
        self.assertEqual(float(selected[1, 0]), 16.0)

    def test_signed_scale_returns_positive_floor(self):
        values = torch.zeros(2, 4)
        self.assertEqual(signed_scale(values), 1.0)

    def test_encode_signed_field_packs_bytes(self):
        values = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float32)
        encoded = encode_signed_field(values, scale=1.0)
        decoded = list(base64.b64decode(encoded))
        self.assertEqual(decoded, [0, 128, 255])

    def test_apply_logit_soft_cap_leaves_logits_unchanged_when_disabled(self):
        logits = torch.tensor([1.0, 2.0], dtype=torch.float32)
        capped = apply_logit_soft_cap(logits, None)
        self.assertTrue(torch.equal(capped, logits))

    def test_top_logit_tokens_returns_sorted_top_k(self):
        logits = torch.tensor([0.5, 3.2, 1.1, 2.4], dtype=torch.float32)
        top_tokens = top_logit_tokens(
            logits,
            decode_token=lambda token_id: f"tok-{token_id}",
            token_factory=LogitToken,
            limit=3,
        )
        self.assertEqual([entry.token_id for entry in top_tokens], [1, 3, 2])
        self.assertEqual(top_tokens[0].token, "tok-1")
        self.assertEqual(top_tokens[1].logit, 2.4)

    def test_tensor_rms_and_logit_shift_rms(self):
        values = torch.tensor([3.0, 4.0], dtype=torch.float32)
        self.assertAlmostEqual(compute_target_rms(values), 3.535533, places=5)
        self.assertAlmostEqual(
            compute_logit_shift_rms(torch.tensor([3.0, 5.0]), torch.tensor([1.0, 1.0])),
            3.162277,
            places=5,
        )

    def test_activation_distribution_metrics(self):
        values = torch.tensor([[1.0, 2.0], [3.0, 10.0]], dtype=torch.float32)
        self.assertGreater(compute_activation_kurtosis(values), 1.0)
        self.assertGreater(compute_top_energy_share(values), 0.5)
        self.assertGreaterEqual(compute_participation_ratio(values), 1.0)

    def test_attention_entropy_metrics_capture_sink_behavior(self):
        attention_row = torch.tensor(
            [
                [0.8, 0.2],
                [0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        mean_entropy, sink_mass, sink_head_ratio = compute_attention_entropy_metrics(attention_row)
        self.assertGreater(mean_entropy, 0.0)
        self.assertAlmostEqual(sink_mass, 0.65, places=5)
        self.assertAlmostEqual(sink_head_ratio, 1.0, places=5)

    def test_build_flow_steps_orders_embedding_then_layer_sequence(self):
        sink = {
            -1: {
                'embeddings': torch.ones(1, 2, 4) * 0.5,
            },
            0: {
                'attn_out': torch.ones(1, 2, 4),
                'resid_after_attn': torch.ones(1, 2, 4) * 2,
                'mlp_out': torch.ones(1, 2, 4) * 3,
                'resid_after_mlp': torch.ones(1, 2, 4) * 4,
            }
        }
        positions = torch.tensor([0, 1], dtype=torch.long)
        steps = build_flow_steps(sink, positions, hidden_width=4, step_factory=FlowStep)
        self.assertEqual([step.stage_id for step in steps], [
            'embeddings', 'attn_out', 'resid_after_attn', 'mlp_out', 'resid_after_mlp'
        ])
        self.assertEqual(steps[0].layer_index, -1)
        self.assertEqual(steps[1].layer_index, 0)
        self.assertEqual(steps[0].rows, 2)
        self.assertEqual(steps[0].cols, 4)


if __name__ == '__main__':
    unittest.main()
