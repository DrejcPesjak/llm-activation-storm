from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from activation_storm.cross_inspect import CrossInspectStore


def make_record(
    *,
    timestamp: str,
    model_id: str,
    model_label: str,
    prompt: str,
    values: list[float],
) -> dict:
    return {
        "timestamp": timestamp,
        "log_type": "metrics",
        "model": {
            "id": model_id,
            "label": model_label,
            "prompt_mode": "base",
            "layer_count": len(values),
            "layer_width": 4,
            "stage_sequence": ["embeddings", "attn_out", "resid_after_attn", "mlp_out", "resid_after_mlp"],
        },
        "request": {
            "prompt": prompt,
            "include_special_tokens": False,
        },
        "context": {
            "tokens": ["hello"],
            "target_position": 0,
            "target_token_id": 1,
            "target_token": "hello",
        },
        "metrics": {
            "layer_analysis": [
                {
                    "layer_index": index,
                    "top_tokens": [],
                    "activation_metrics": {
                        "layer_variance": value,
                        "kurtosis": value + 1,
                        "top_energy_share": value + 2,
                        "participation_ratio": value + 3,
                    },
                    "attention_metrics": {
                        "mean_entropy": value + 4,
                        "sink_mass": value + 5,
                        "sink_head_ratio": value + 6,
                    },
                    "contribution_metrics": {
                        "logit_shift_rms": value + 7,
                    },
                }
                for index, value in enumerate(values)
            ]
        },
    }


class CrossInspectTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.tempdir.name) / "logs"
        self.log_dir.mkdir()
        self.store = CrossInspectStore(self.log_dir)

    def tearDown(self):
        self.tempdir.cleanup()

    def write_records(self, path_name: str, records: list[dict], *, malformed_line: str | None = None) -> None:
        path = self.log_dir / path_name
        lines = [json.dumps(record) for record in records]
        if malformed_line is not None:
            lines.append(malformed_line)
        path.write_text("\n".join(lines), encoding="utf-8")

    def test_list_runs_returns_newest_first_and_skips_invalid_lines(self):
        self.write_records(
            "metrics_2026-04-06_10-00.jsonl",
            [
                make_record(
                    timestamp="2026-04-06T10:00:00",
                    model_id="model-a",
                    model_label="Model A",
                    prompt="First prompt",
                    values=[1.0, 2.0],
                )
            ],
            malformed_line="{broken",
        )
        self.write_records(
            "metrics_2026-04-06_10-05.jsonl",
            [
                make_record(
                    timestamp="2026-04-06T10:05:00",
                    model_id="model-a",
                    model_label="Model A",
                    prompt="Second prompt",
                    values=[3.0, 4.0],
                )
            ],
        )

        runs = self.store.list_runs()

        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0].prompt, "Second prompt")
        self.assertEqual(runs[0].run_id, "metrics_2026-04-06_10-05.jsonl:1")
        self.assertEqual(runs[1].run_id, "metrics_2026-04-06_10-00.jsonl:1")

    def test_aggregate_selected_returns_mean_and_std(self):
        self.write_records(
            "metrics_2026-04-06_10-00.jsonl",
            [
                make_record(
                    timestamp="2026-04-06T10:00:00",
                    model_id="model-a",
                    model_label="Model A",
                    prompt="Prompt one",
                    values=[1.0, 3.0],
                ),
                make_record(
                    timestamp="2026-04-06T10:01:00",
                    model_id="model-a",
                    model_label="Model A",
                    prompt="Prompt two",
                    values=[3.0, 5.0],
                ),
            ],
        )

        result = self.store.analyze(
            {
                "mode": "aggregate_selected",
                "run_ids": [
                    "metrics_2026-04-06_10-00.jsonl:1",
                    "metrics_2026-04-06_10-00.jsonl:2",
                ],
            }
        )

        series = result["group_a"]["metric_trends"]["layer_variance"]
        self.assertEqual(result["group_a"]["run_count"], 2)
        self.assertEqual(series[0]["mean"], 2.0)
        self.assertEqual(series[-1]["mean"], 4.0)
        self.assertEqual(series[0]["std"], 1.0)
        self.assertEqual(result["group_a"]["metric_summaries"]["layer_variance"]["final_value"], 4.0)

    def test_compare_groups_aligns_relative_depth_and_computes_delta(self):
        self.write_records(
            "metrics_2026-04-06_10-00.jsonl",
            [
                make_record(
                    timestamp="2026-04-06T10:00:00",
                    model_id="model-a",
                    model_label="Model A",
                    prompt="Prompt A",
                    values=[0.0, 10.0],
                ),
                make_record(
                    timestamp="2026-04-06T10:01:00",
                    model_id="model-b",
                    model_label="Model B",
                    prompt="Prompt B",
                    values=[0.0, 5.0, 10.0],
                ),
            ],
        )

        result = self.store.analyze(
            {
                "mode": "compare_groups",
                "group_a_run_ids": ["metrics_2026-04-06_10-00.jsonl:1"],
                "group_b_run_ids": ["metrics_2026-04-06_10-00.jsonl:2"],
            }
        )

        delta_series = result["delta"]["metric_trends"]["layer_variance"]
        self.assertEqual(delta_series[0]["mean_delta"], 0.0)
        self.assertEqual(delta_series[50]["mean_delta"], 0.0)
        self.assertEqual(delta_series[-1]["mean_delta"], 0.0)

    def test_compare_two_runs_requires_same_model(self):
        self.write_records(
            "metrics_2026-04-06_10-00.jsonl",
            [
                make_record(
                    timestamp="2026-04-06T10:00:00",
                    model_id="model-a",
                    model_label="Model A",
                    prompt="Prompt A",
                    values=[1.0, 2.0],
                ),
                make_record(
                    timestamp="2026-04-06T10:01:00",
                    model_id="model-b",
                    model_label="Model B",
                    prompt="Prompt B",
                    values=[1.0, 2.0],
                ),
            ],
        )

        with self.assertRaisesRegex(ValueError, "exactly one model"):
            self.store.analyze(
                {
                    "mode": "compare_two_runs",
                    "run_ids": [
                        "metrics_2026-04-06_10-00.jsonl:1",
                        "metrics_2026-04-06_10-00.jsonl:2",
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
