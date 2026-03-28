from __future__ import annotations

import unittest

from transformer_lens.loading_from_pretrained import get_official_model_name

from activation_storm.adapters import TL_MODEL_SPECS
from activation_storm.transformer_lens_adapter import (
    PARALLEL_TL_STAGE_SEQUENCE,
    TL_STAGE_SEQUENCE,
    TransformerLensAdapter,
    find_subsequence,
)


class AdapterConfigTests(unittest.TestCase):
    def test_tl_model_specs_have_unique_ids_and_labels(self):
        model_ids = [spec.model_id for spec in TL_MODEL_SPECS]
        labels = [spec.label for spec in TL_MODEL_SPECS]
        self.assertEqual(len(model_ids), len(set(model_ids)))
        self.assertTrue(all(model_id for model_id in model_ids))
        self.assertTrue(all(label for label in labels))
        self.assertIn("pythia-160m", model_ids)
        self.assertIn("pythia-1b", model_ids)
        self.assertIn("pythia-2.8b", model_ids)

    def test_tl_model_specs_resolve_via_transformer_lens_alias_map(self):
        resolved = {spec.model_id: get_official_model_name(spec.model_id) for spec in TL_MODEL_SPECS}
        self.assertEqual(resolved["gpt2-small"], "gpt2")
        self.assertEqual(resolved["llama-7b"], "llama-7b-hf")
        self.assertEqual(resolved["qwen3-1.7b"], "Qwen/Qwen3-1.7B")
        self.assertEqual(resolved["gemma-3-1b-it"], "google/gemma-3-1b-it")

    def test_qwen_specs_require_remote_code_explicitly(self):
        qwen_specs = {spec.model_id: spec for spec in TL_MODEL_SPECS if spec.model_id.startswith("qwen")}
        self.assertTrue(qwen_specs["qwen-1.8b"].trust_remote_code)
        self.assertTrue(qwen_specs["qwen-1.8b-chat"].trust_remote_code)
        self.assertTrue(qwen_specs["qwen3-1.7b"].trust_remote_code)

    def test_pythia_specs_mark_parallel_attention_mlp(self):
        pythia_specs = {spec.model_id: spec for spec in TL_MODEL_SPECS if spec.model_id.startswith("pythia")}
        self.assertTrue(pythia_specs["pythia-160m"].parallel_attn_mlp)
        self.assertTrue(pythia_specs["pythia-1b"].parallel_attn_mlp)
        self.assertTrue(pythia_specs["pythia-2.8b"].parallel_attn_mlp)


class TransformerLensStageTests(unittest.TestCase):
    def test_stage_sequence_is_reduced_for_parallel_models(self):
        parallel_spec = next(spec for spec in TL_MODEL_SPECS if spec.model_id == "pythia-160m")
        adapter = TransformerLensAdapter.__new__(TransformerLensAdapter)
        adapter._spec = parallel_spec
        self.assertEqual(adapter._stage_sequence(), PARALLEL_TL_STAGE_SEQUENCE)

    def test_stage_sequence_is_full_for_sequential_models(self):
        sequential_spec = next(spec for spec in TL_MODEL_SPECS if spec.model_id == "gpt2-small")
        adapter = TransformerLensAdapter.__new__(TransformerLensAdapter)
        adapter._spec = sequential_spec
        self.assertEqual(adapter._stage_sequence(), TL_STAGE_SEQUENCE)


class FindSubsequenceTests(unittest.TestCase):
    def test_find_subsequence_returns_first_match(self):
        self.assertEqual(find_subsequence([1, 2, 3, 2, 3], [2, 3]), 1)

    def test_find_subsequence_returns_none_when_not_found(self):
        self.assertIsNone(find_subsequence([1, 2, 3], [4, 5]))


if __name__ == "__main__":
    unittest.main()
