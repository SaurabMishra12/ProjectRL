"""Software tests only: never experiment predictions or research findings."""
import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from protocol import ROOT, score, load_official, normalize_row, exact_slice, delivered_facts, parse_completion, digest, write_json

class CharacterTokenizer:
    # Test-only codec checks slicing contracts, never used to report payload tokens.
    def encode(self,text): return list(text)
    def decode(self,ids): return ''.join(ids)
    def count(self,text): return len(text)

class IntegrityTests(unittest.TestCase):
    def test_scoring_against_downloaded_official_code_and_real_legacy_answers(self):
        # Upstream imports ujson; stdlib substitution affects IO only, not scorer semantics.
        sys.modules.setdefault('ujson', json)
        spec=importlib.util.spec_from_file_location('upstream',ROOT/'references/hotpot_evaluate_v1.py')
        official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
        rows=[json.loads(l) for l in (ROOT.parent/'results/W1_web_cost_accuracy/raw_results.jsonl').read_text().splitlines()]
        for row in rows:
            for result in row.values():
                if not isinstance(result,dict) or 'answer' not in result: continue
                self.assertAlmostEqual(score(result['answer'],row['gold_answer'])['f1'],100*official.f1_score(result['answer'],row['gold_answer'])[0])
                self.assertEqual(score(result['answer'],row['gold_answer'])['em'],100*official.exact_match_score(result['answer'],row['gold_answer']))

    def test_real_dataset_matches_all_existing_eval_inputs(self):
        official={r['id']:r for r in load_official()}
        for row in json.loads((ROOT.parent/'web_eval_subset_100.json').read_text()):
            normalized=normalize_row(row)
            self.assertEqual(normalized,official[normalized['id']])

    def test_exact_slice_never_pads_short_source(self):
        for mode in ['prefix','crop']:
            selected=exact_slice('short',100,CharacterTokenizer(),mode,42)
            self.assertEqual(selected['text'],'short');self.assertEqual(selected['tokens'],5)
        with self.assertRaises(ValueError):exact_slice('',12,CharacterTokenizer())

    def test_exact_crop_is_deterministic_and_matches_prefix_budget(self):
        text=load_official()[0]['question']
        a=exact_slice(text,10,CharacterTokenizer(),'prefix')
        b=exact_slice(text,10,CharacterTokenizer(),'crop',11)
        self.assertEqual(a['tokens'],b['tokens'])
        self.assertEqual(b,exact_slice(text,10,CharacterTokenizer(),'crop',11))

    def test_partial_evidence_line_does_not_count_delivery(self):
        self.assertEqual(delivered_facts('Evidence: {"title":"cut off'),[])

    def test_citation_not_visible_to_agent_is_invalid(self):
        doc=load_official()[0]['documents'][0]
        text=json.dumps({'answer':'noanswer','citations':[[doc['title'],0]]})
        response=parse_completion(text,[{**doc,'visible_indices':[]}])
        self.assertEqual(response['citations'],[])
        self.assertEqual(len(response['invalid_citations']),1)

    def test_real_execution_requires_configuration_and_never_uses_mock(self):
        from backend import RealBackend
        cfg=json.loads((ROOT/'config.example.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError): RealBackend({**cfg,'provider':'mock'},None,temp)
            with self.assertRaises(ValueError): RealBackend(cfg,None,temp)

    def test_incomplete_results_cannot_be_reported(self):
        from reporting import load_completed
        with tempfile.TemporaryDirectory() as temp:
            write_json(pathlib.Path(temp)/'E1_status.json',{'status':'INCOMPLETE'})
            with self.assertRaises(ValueError):load_completed(temp,'E1')

    def test_duplicate_rows_cannot_pass_complete_sample_validation(self):
        from reporting import load_completed
        from protocol import file_hash
        with tempfile.TemporaryDirectory() as temp:
            p=pathlib.Path(temp)
            (p/'E1.jsonl').write_text('{"id":"a"}\n{"id":"a"}\n')
            write_json(p/'E1_status.json',{'status':'COMPLETE','sha256':file_hash(p/'E1.jsonl')})
            write_json(p/'frozen_manifest.json',{'plan':{'question_ids':['a','b']}})
            with self.assertRaises(ValueError):load_completed(p,'E1')

if __name__=='__main__':unittest.main()
