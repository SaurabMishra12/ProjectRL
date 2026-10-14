import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.resize_controller import freeze_subsets, import_responses
from controller_splits import load_partition, assert_disjoint
from protocol import file_hash


class ResizeControllerTests(unittest.TestCase):
    def test_authentic_subset_preserves_heldout_and_disjointness(self):
        parent = ROOT / 'artifacts/controller_splits_e219bbccd74a5888'
        settings = json.loads((parent / 'manifest.json').read_text())['spec']['config']
        settings = {**settings, 'train_questions': 100, 'validation_questions': 30}
        with tempfile.TemporaryDirectory() as tmp, patch('analysis.resize_controller.ROOT', Path(tmp)):
            result = freeze_subsets(parent, settings)
            parts = {n: load_partition(result, n, gold=True)
                     for n in ('train', 'validation', 'heldout', 'pilot')}
            self.assertEqual([len(parts[n]) for n in ('train', 'validation', 'heldout')], [100, 30, 300])
            self.assertEqual(file_hash(result / 'heldout_gold.json'), file_hash(parent / 'heldout_gold.json'))
            self.assertTrue(all(not any(v.values()) for v in assert_disjoint(parts).values()))
            before = file_hash(result / 'manifest.json')
            self.assertEqual(freeze_subsets(parent, settings), result)
            self.assertEqual(file_hash(result / 'manifest.json'), before)

    def test_import_authentic_response_checks_integrity_without_model_calls(self):
        source = ROOT / 'artifacts/controller_study_ef6d6250364374ee/inference'
        response = next((source / 'responses').glob('*.json'))
        with tempfile.TemporaryDirectory() as tmp:
            small = Path(tmp) / 'source'; (small / 'responses').mkdir(parents=True)
            copied = small / 'responses' / response.name
            copied.write_bytes(response.read_bytes())
            (small / 'calls.jsonl').write_bytes((source / 'calls.jsonl').read_bytes())
            destination = Path(tmp) / 'imported'
            self.assertEqual(import_responses(small, destination), 1)
            self.assertEqual(file_hash(copied), file_hash(destination / 'responses' / copied.name))
            self.assertFalse((destination / 'calls.jsonl').exists())
            data = json.loads(copied.read_text()); data['latency_seconds'] = -1
            copied.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                import_responses(small, Path(tmp) / 'invalid')
