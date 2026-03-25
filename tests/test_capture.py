from __future__ import annotations

import unittest

import torch

from activation_storm.capture import ease_matrix, normalize_family_values, tensor_rms


class CaptureTests(unittest.TestCase):
    def test_tensor_rms_reduces_hidden_dimension(self):
        values = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
        reduced = tensor_rms(values)
        self.assertEqual(list(reduced.shape), [1, 2])
        self.assertAlmostEqual(float(reduced[0, 0]), 3.5355, places=3)

    def test_normalize_family_values_scales_per_family(self):
        family_by_layer = {
            "resid": {
                0: torch.tensor([1.0, 2.0]),
                1: torch.tensor([2.0, 4.0]),
            }
        }
        normalized = normalize_family_values(family_by_layer, token_count=2, layer_count=2)
        self.assertEqual(normalized["resid"][0], [0.25, 0.5])
        self.assertEqual(normalized["resid"][1], [0.5, 1.0])

    def test_ease_matrix_preserves_zero_and_one(self):
        self.assertEqual(ease_matrix([[0.0, 0.5, 1.0]]), [[0.0, 0.5, 1.0]])


if __name__ == "__main__":
    unittest.main()
