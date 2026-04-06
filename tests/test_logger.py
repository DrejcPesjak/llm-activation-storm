from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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


def fake_result() -> FlowAnalysisResult:
    return FlowAnalysisResult(
        model=ModelInfo(
            id='fake',
            label='Fake',
            layer_count=2,
            layer_width=4,
            stage_sequence=['embeddings', 'attn_out', 'resid_after_attn', 'mlp_out', 'resid_after_mlp'],
            prompt_mode='base',
            default_prompt='The capital of France is',
        ),
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
        ],
        target_position=1,
        target_token_id=42,
        target_token='hello',
        layer_analysis=[
            LayerAnalysis(
                layer_index=0,
                top_tokens=[LogitToken(token_id=1, token='world', logit=3.5)],
                activation_metrics=ActivationMetrics(layer_variance=1.5, kurtosis=2.5, top_energy_share=0.33, participation_ratio=1.2),
                attention_metrics=AttentionMetrics(0.75, 0.2, 0.5),
                contribution_metrics=ContributionMetrics(1.1),
            ),
        ],
    )


class LoggerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.tempdir.name) / "logs"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_log_metrics_creates_log_directory_and_file(self):
        logger = RunLogger(LoggerConfig(log_dir=self.log_dir, enabled=True), session_stamp="2026-03-30_12-30")

        logger.log_metrics(prompt="hello", include_special_tokens=False, result=fake_result())

        path = self.log_dir / "metrics_2026-03-30_12-30.jsonl"
        self.assertTrue(path.exists())
        record = json.loads(path.read_text(encoding="utf-8").strip())
        self.assertEqual(record["log_type"], "metrics")
        self.assertEqual(record["request"]["prompt"], "hello")

    def test_log_metrics_appends_to_same_session_file(self):
        logger = RunLogger(LoggerConfig(log_dir=self.log_dir, enabled=True), session_stamp="2026-03-30_12-30")

        logger.log_metrics(prompt="hello", include_special_tokens=False, result=fake_result())
        logger.log_metrics(prompt="world", include_special_tokens=True, result=fake_result())

        path = self.log_dir / "metrics_2026-03-30_12-30.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[1])["request"]["prompt"], "world")

    def test_disabled_logger_does_not_create_files(self):
        logger = RunLogger(LoggerConfig(log_dir=self.log_dir, enabled=False), session_stamp="2026-03-30_12-30")

        logger.log_metrics(prompt="hello", include_special_tokens=False, result=fake_result())

        self.assertFalse(self.log_dir.exists())


if __name__ == '__main__':
    unittest.main()
