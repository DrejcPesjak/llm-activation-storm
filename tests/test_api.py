from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from activation_storm.api import ActivationStormApp
from activation_storm.types import FlowAnalysisResult, FlowStep, ModelInfo


class FakeAdapter:
    def architecture_text(self) -> str:
        return "FakeModel(\n  (layers): FakeStack()\n)"

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            id='fake',
            label='Fake',
            layer_count=2,
            layer_width=4,
            stage_sequence=['embeddings', 'attn_out', 'resid_after_attn', 'mlp_out', 'resid_after_mlp'],
        )

    def analyze_prompt(self, prompt: str, include_special_tokens: bool = False) -> FlowAnalysisResult:
        if not prompt.strip():
            raise ValueError('Prompt must not be empty.')
        return FlowAnalysisResult(
            model=self.model_info(),
            tokens=['hello'],
            hidden_width=4,
            token_limit=1,
            token_limit_applied=False,
            steps=[
                FlowStep(
                    step_index=0,
                    layer_index=-1,
                    stage_id='embeddings',
                    stage_label='EMB',
                    rows=1,
                    cols=4,
                    scale=1.0,
                    encoded_field='AAAA',
                ),
                FlowStep(
                    step_index=1,
                    layer_index=0,
                    stage_id='attn_out',
                    stage_label='ATTN',
                    rows=1,
                    cols=4,
                    scale=1.0,
                    encoded_field='AAAA',
                )
            ],
        )


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        static_dir = Path(self.tempdir.name)
        (static_dir / 'index.html').write_text('<h1>ok</h1>', encoding='utf-8')
        self.app = ActivationStormApp(static_dir=static_dir)
        self.app.registry = {'fake': FakeAdapter()}

    def tearDown(self):
        self.tempdir.cleanup()

    def test_models_payload(self):
        payload = self.app.models_payload()
        self.assertEqual(payload['default_model'], 'fake')
        self.assertEqual(payload['models'][0]['stage_sequence'][0], 'embeddings')

    def test_analyze_validates_model(self):
        with self.assertRaises(ValueError):
            self.app.analyze({'model_id': 'missing', 'prompt': 'hello'})

    def test_analyze_returns_serializable_payload(self):
        payload = self.app.analyze({'model_id': 'fake', 'prompt': 'hello'})
        self.assertEqual(payload['tokens'], ['hello'])
        self.assertEqual(payload['steps'][0]['stage_id'], 'embeddings')
        self.assertEqual(payload['steps'][1]['stage_id'], 'attn_out')
        self.assertEqual(payload['steps'][0]['rows'], 1)

    def test_architecture_payload_returns_model_printout(self):
        payload = self.app.architecture_payload('fake')
        self.assertEqual(payload['model']['id'], 'fake')
        self.assertIn('FakeModel', payload['architecture'])

    def test_architecture_payload_validates_model(self):
        with self.assertRaises(ValueError):
            self.app.architecture_payload('missing')


if __name__ == '__main__':
    unittest.main()
