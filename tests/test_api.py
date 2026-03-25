from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from activation_storm.api import ActivationStormApp
from activation_storm.types import ActivationFrame, AnalysisResult, ModelInfo


class FakeAdapter:
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            id="fake",
            label="Fake",
            layer_count=2,
            layer_width=4,
            families=["resid", "attn", "mlp"],
        )

    def analyze_prompt(self, prompt: str) -> AnalysisResult:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        return AnalysisResult(
            model=self.model_info(),
            tokens=["hello"],
            layers=["L00", "L01"],
            families=["resid", "attn", "mlp"],
            frames=[
                ActivationFrame(
                    token_index=0,
                    token_text="hello",
                    values={
                        "resid": [0.1, 0.2],
                        "attn": [0.2, 0.3],
                        "mlp": [0.3, 0.4],
                    },
                )
            ],
        )


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        static_dir = Path(self.tempdir.name)
        (static_dir / "index.html").write_text("<h1>ok</h1>", encoding="utf-8")
        self.app = ActivationStormApp(static_dir=static_dir)
        self.app.registry = {"fake": FakeAdapter()}

    def tearDown(self):
        self.tempdir.cleanup()

    def test_models_payload(self):
        payload = self.app.models_payload()
        self.assertEqual(payload["default_model"], "fake")
        self.assertEqual(payload["models"][0]["layer_count"], 2)

    def test_analyze_validates_model(self):
        with self.assertRaises(ValueError):
            self.app.analyze({"model_id": "missing", "prompt": "hello"})

    def test_analyze_returns_serializable_payload(self):
        payload = self.app.analyze({"model_id": "fake", "prompt": "hello"})
        self.assertEqual(payload["tokens"], ["hello"])
        self.assertEqual(payload["frames"][0]["values"]["mlp"], [0.3, 0.4])


if __name__ == "__main__":
    unittest.main()
