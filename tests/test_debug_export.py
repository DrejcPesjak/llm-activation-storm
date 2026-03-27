from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from activation_storm.debug_export import export_layer_zero_debug


class DebugExportTests(unittest.TestCase):
    def test_export_layer_zero_debug_writes_expected_files(self):
        sink = {
            0: {
                'attn_out': torch.tensor([[[20.0, -1.0, 0.5], [2.0, -2.0, 1.5]]]),
                'resid_after_attn': torch.tensor([[[15.0, -0.3, 0.4], [0.5, -0.6, 0.7]]]),
                'mlp_out': torch.tensor([[[18.0, 1.2, 1.3], [1.4, 1.5, 1.6]]]),
                'resid_after_mlp': torch.tensor([[[22.0, 2.2, 2.3], [2.4, 2.5, 2.6]]]),
            }
        }
        positions = torch.tensor([0, 1], dtype=torch.long)
        tokens = ['foo', 'bar']

        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_layer_zero_debug(
                output_dir=Path(tmpdir),
                sink=sink,
                positions=positions,
                tokens=tokens,
            )

            self.assertEqual(len(result['files']), 5)
            edge_path = Path(tmpdir) / 'layer00_attn_out_edge_columns.csv'
            self.assertTrue(edge_path.exists())
            with edge_path.open(newline='', encoding='utf-8') as handle:
                edge_rows = list(csv.DictReader(handle))
            self.assertEqual(edge_rows[0]['token'], 'foo')
            self.assertEqual(edge_rows[0]['dim_0'], '20.0')
            self.assertEqual(edge_rows[0]['dim_2'], '0.5')

            plotted_path = Path(tmpdir) / 'layer00_plotted_points.csv'
            self.assertTrue(plotted_path.exists())
            with plotted_path.open(newline='', encoding='utf-8') as handle:
                plotted_rows = list(csv.DictReader(handle))
            self.assertEqual(len(plotted_rows), 4)
            self.assertEqual(plotted_rows[0]['stage_id'], 'attn_out')
            self.assertEqual(plotted_rows[0]['step_index'], '0')
            self.assertEqual(plotted_rows[0]['draw_order'], '0')
            self.assertEqual(plotted_rows[0]['token_index'], '0')
            self.assertEqual(plotted_rows[0]['dim_index'], '0')
            self.assertEqual(plotted_rows[0]['dim_pct'], '0.0')
            self.assertTrue(plotted_rows[0]['color_hex'].startswith('#'))
            self.assertGreater(float(plotted_rows[0]['radius']), 1.2)


if __name__ == '__main__':
    unittest.main()
