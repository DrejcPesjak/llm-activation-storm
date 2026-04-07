from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from activation_storm.api import ActivationStormApp
from activation_storm.logger import LoggerConfig, RunLogger
from activation_storm.types import (
    ActivationMetrics,
    AttentionMetrics,
    ContributionMetrics,
    FlowAnalysisResult,
    FlowStep,
    LayerAnalysis,
    LogitToken,
    ModelInfo,
)


class FakeAdapter:
    def __init__(self):
        self.include_layer_analysis_calls: list[bool] = []

    def architecture_text(self) -> str:
        return "FakeModel(\n  (layers): FakeStack()\n)"

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            id='fake',
            label='Fake',
            layer_count=2,
            layer_width=4,
            stage_sequence=['embeddings', 'attn_out', 'resid_after_attn', 'mlp_out', 'resid_after_mlp'],
            prompt_mode='base',
            default_prompt='The capital of France is',
        )

    def analyze_prompt(
        self,
        prompt: str,
        include_special_tokens: bool = False,
        include_layer_analysis: bool = True,
    ) -> FlowAnalysisResult:
        if not prompt.strip():
            raise ValueError('Prompt must not be empty.')
        self.include_layer_analysis_calls.append(include_layer_analysis)
        layer_analysis = [
            LayerAnalysis(
                layer_index=0,
                top_tokens=[
                    LogitToken(token_id=1, token='world', logit=3.5),
                    LogitToken(token_id=2, token='there', logit=2.25),
                ],
                activation_metrics=ActivationMetrics(layer_variance=1.5, kurtosis=2.5, top_energy_share=0.33, participation_ratio=1.2),
                attention_metrics=AttentionMetrics(0.75, 0.2, 0.5),
                contribution_metrics=ContributionMetrics(1.1),
            ),
            LayerAnalysis(
                layer_index=1,
                top_tokens=[
                    LogitToken(token_id=3, token='again', logit=1.75),
                ],
                activation_metrics=ActivationMetrics(layer_variance=1.9, kurtosis=2.9, top_energy_share=0.4, participation_ratio=1.4),
                attention_metrics=AttentionMetrics(0.55, 0.15, 0.25),
                contribution_metrics=ContributionMetrics(0.8),
            ),
        ] if include_layer_analysis else []
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
            target_position=1,
            target_token_id=42,
            target_token='hello',
            layer_analysis=layer_analysis,
        )


class BrokenLogger:
    def log_metrics(self, **_kwargs):
        raise RuntimeError("log write failed")


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.static_dir = Path(self.tempdir.name) / 'static'
        self.static_dir.mkdir()
        static_dir = self.static_dir
        (static_dir / 'index.html').write_text('<h1>ok</h1>', encoding='utf-8')
        self.fake_adapter = FakeAdapter()
        self.log_dir = Path(self.tempdir.name) / 'logs'
        self.app = ActivationStormApp(static_dir=static_dir, registry={'fake': self.fake_adapter}, log_dir=self.log_dir)

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
        self.assertEqual(payload['target_position'], 1)
        self.assertEqual(payload['target_token'], 'hello')
        self.assertEqual(payload['layer_analysis'], [])
        self.assertEqual(self.fake_adapter.include_layer_analysis_calls[-1], False)
        self.assertEqual(list((Path(self.tempdir.name) / 'logs').glob('*')), [])

    def test_layer_analysis_payload_returns_analysis_only(self):
        payload = self.app.layer_analysis_payload({'model_id': 'fake', 'prompt': 'hello'})
        self.assertEqual(payload['target_token'], 'hello')
        self.assertEqual(payload['layer_analysis'][0]['top_tokens'][0]['token'], 'world')
        self.assertEqual(payload['layer_analysis'][1]['layer_index'], 1)
        self.assertEqual(payload['layer_analysis'][0]['activation_metrics']['kurtosis'], 2.5)
        self.assertEqual(self.fake_adapter.include_layer_analysis_calls[-1], True)

    def test_layer_analysis_payload_logs_metrics_record(self):
        log_dir = Path(self.tempdir.name) / 'logs'
        logger = RunLogger(LoggerConfig(log_dir=log_dir, enabled=True), session_stamp="2026-03-30_12-00")
        app = ActivationStormApp(static_dir=self.static_dir, registry={'fake': self.fake_adapter}, logger=logger)

        payload = app.layer_analysis_payload({'model_id': 'fake', 'prompt': 'hello', 'include_special_tokens': True})

        self.assertEqual(payload['target_token'], 'hello')
        files = list(log_dir.glob('metrics_2026-03-30_12-00.jsonl'))
        self.assertEqual(len(files), 1)
        lines = files[0].read_text(encoding='utf-8').strip().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record['log_type'], 'metrics')
        self.assertEqual(record['model']['id'], 'fake')
        self.assertEqual(record['request']['prompt'], 'hello')
        self.assertTrue(record['request']['include_special_tokens'])
        self.assertEqual(record['context']['tokens'], ['hello'])
        self.assertEqual(record['metrics']['layer_analysis'][0]['layer_index'], 0)

    def test_analyze_does_not_log_metrics_record(self):
        log_dir = Path(self.tempdir.name) / 'logs'
        logger = RunLogger(LoggerConfig(log_dir=log_dir, enabled=True), session_stamp="2026-03-30_12-00")
        app = ActivationStormApp(static_dir=self.static_dir, registry={'fake': self.fake_adapter}, logger=logger)

        app.analyze({'model_id': 'fake', 'prompt': 'hello'})

        self.assertEqual(list(log_dir.glob('*.jsonl')), [])

    def test_logging_failure_does_not_break_layer_analysis(self):
        app = ActivationStormApp(static_dir=self.static_dir, registry={'fake': self.fake_adapter}, logger=BrokenLogger())

        payload = app.layer_analysis_payload({'model_id': 'fake', 'prompt': 'hello'})

        self.assertEqual(payload['target_token'], 'hello')

    def test_architecture_payload_returns_model_printout(self):
        payload = self.app.architecture_payload('fake')
        self.assertEqual(payload['model']['id'], 'fake')
        self.assertIn('FakeModel', payload['architecture'])

    def test_architecture_payload_validates_model(self):
        with self.assertRaises(ValueError):
            self.app.architecture_payload('missing')

    def test_cross_inspect_runs_payload_reads_logged_metrics(self):
        logger = RunLogger(LoggerConfig(log_dir=self.log_dir, enabled=True), session_stamp="2026-03-30_12-00")
        app = ActivationStormApp(static_dir=self.static_dir, registry={'fake': self.fake_adapter}, logger=logger, log_dir=self.log_dir)
        app.layer_analysis_payload({'model_id': 'fake', 'prompt': 'hello'})

        payload = app.cross_inspect_runs_payload()

        self.assertEqual(len(payload['runs']), 1)
        self.assertEqual(payload['runs'][0]['model_id'], 'fake')
        self.assertEqual(payload['runs'][0]['prompt_preview'], 'hello')

    def test_cross_inspect_analyze_payload_aggregates_selection(self):
        logger = RunLogger(LoggerConfig(log_dir=self.log_dir, enabled=True), session_stamp="2026-03-30_12-00")
        app = ActivationStormApp(static_dir=self.static_dir, registry={'fake': self.fake_adapter}, logger=logger, log_dir=self.log_dir)
        app.layer_analysis_payload({'model_id': 'fake', 'prompt': 'hello'})
        app.layer_analysis_payload({'model_id': 'fake', 'prompt': 'world'})

        result = app.cross_inspect_analyze_payload({
            'mode': 'aggregate_selected',
            'run_ids': ['metrics_2026-03-30_12-00.jsonl:1', 'metrics_2026-03-30_12-00.jsonl:2'],
        })

        self.assertEqual(result['group_a']['run_count'], 2)
        self.assertIn('layer_variance', result['group_a']['metric_trends'])
        self.assertIn('metric_catalog', result)


if __name__ == '__main__':
    unittest.main()
