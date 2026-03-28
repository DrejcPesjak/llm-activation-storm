from __future__ import annotations

import unittest

from transformer_lens.loading_from_pretrained import get_official_model_name

from activation_storm.adapters import TL_MODEL_SPECS
from activation_storm.transformer_lens_adapter import find_subsequence


class AdapterConfigTests(unittest.TestCase):
    def test_tl_model_specs_have_unique_ids_and_labels(self):
        model_ids = [spec.model_id for spec in TL_MODEL_SPECS]
        labels = [spec.label for spec in TL_MODEL_SPECS]
        self.assertEqual(len(model_ids), len(set(model_ids)))
        self.assertTrue(all(model_id for model_id in model_ids))
        self.assertTrue(all(label for label in labels))

    def test_tl_model_specs_resolve_via_transformer_lens_alias_map(self):
        resolved = {spec.model_id: get_official_model_name(spec.model_id) for spec in TL_MODEL_SPECS}
        self.assertEqual(resolved["gpt2-small"], "gpt2")
        self.assertEqual(resolved["llama-7b"], "llama-7b-hf")
        self.assertEqual(resolved["qwen3-1.7b"], "Qwen/Qwen3-1.7B")
        self.assertEqual(resolved["gemma-3-1b-it"], "google/gemma-3-1b-it")


class FindSubsequenceTests(unittest.TestCase):
    def test_find_subsequence_returns_first_match(self):
        self.assertEqual(find_subsequence([1, 2, 3, 2, 3], [2, 3]), 1)

    def test_find_subsequence_returns_none_when_not_found(self):
        self.assertIsNone(find_subsequence([1, 2, 3], [4, 5]))


if __name__ == "__main__":
    unittest.main()
