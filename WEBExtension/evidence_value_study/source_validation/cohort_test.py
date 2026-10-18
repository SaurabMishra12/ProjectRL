"""Guard tests for the offline candidate/annotation preparation workflow."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).with_name('cohort_prepare.py')

class CohortGuards(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE_PATH.exists(), 'offline cohort preparation is not implemented')
        spec = importlib.util.spec_from_file_location('cohort_prepare', MODULE_PATH)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_normalized_question_deduplicates_punctuation_unicode(self):
        self.assertEqual(self.m.normalize_question('Ｗho is X?'), self.m.normalize_question('who IS X !'))

    def test_exposure_excludes_question_alias_and_separates_rank_tail(self):
        records = [{'id':'2hop__1_2','question':'Who is X?'}, {'id':'2hop__3_4','question':'Who is Y?'}]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'used.json').write_text(json.dumps({'question':'WHO IS X!'}))
            (root/'rank.json').write_text(json.dumps({'pilot_ids':[], 'annotation_ids':[], 'confirmatory_order':['other_id']*1000+['2hop__3_4']}))
            result=self.m.audit_exposure(root, records, ['used.json','rank.json'], raw_paths=set(), ranking_paths={'rank.json'})
            self.assertIn('2hop__1_2',result['excluded_ids'])
            self.assertNotIn('2hop__3_4',result['excluded_ids'])
            self.assertIn('2hop__3_4',result['ranking_only_ids'])

    def test_utf16_historical_question_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'old.txt').write_bytes('{"question":"Who is X?"}'.encode('utf-16'))
            result=self.m.audit_exposure(root,[{'id':'2hop__1_2','question':'Who is X?'}],['old.txt'],raw_paths=set(),ranking_paths=set())
            self.assertIn('2hop__1_2',result['excluded_ids'])
            self.assertEqual(result['unreadable_inputs'],[])

    def test_hotpot_union_retains_whole_docs_and_requires_both_supports(self):
        row={'id':'h','question':'q','answer':'a','supporting_facts':{'title':['A','B']},'context':{'title':['A','B','C','D','E','F'], 'sentences':[['first.',' second.'],['b'],['c'],['d'],['e'],['f']]}}
        c=self.m.hotpot_candidate(row,0,{'source_sha256':'hash'})
        self.assertEqual(len(c['documents']),6)
        self.assertEqual(next(d['text'] for d in c['documents'] if d['title']=='A'),'first. second.')
        row['context']['title'][1]='G'
        self.assertIsNone(self.m.hotpot_candidate(row,0,{'source_sha256':'hash'}))

    def test_blinded_packages_hide_subset_and_gold_and_leave_labels_empty(self):
        c={'id':'q1','dataset':'fixture','question':'Which person?', 'gold_answer':'SECRET','support_ids':['d0','d1'], 'documents':[{'id':f'd{i}','title':f'T{i}','text':f'full {i}','provenance':{'raw_idx':i}} for i in range(6)]}
        packages, labels, keys=self.m.annotation_packages([c])
        self.assertEqual(set(packages),{'R1','R2'})
        self.assertEqual(len(packages['R1']),9)
        for slot, rows in packages.items():
            for r in rows:
                self.assertEqual(set(r),{'case_id','question','documents'})
                self.assertNotIn('SECRET',json.dumps(r))
                for d in r['documents']:self.assertEqual(set(d),{'id','title','text'})
            self.assertTrue(all(r['sufficient'] is None for r in labels[slot]))
        self.assertTrue(all(r.get('candidate_union_sha256') and r.get('planned_assignment_sha256') for r in keys))
        self.assertEqual({r['subset'] for r in keys}, {'A_only','B_only','distractors_only','A_plus_B','full_union','full_union_minus_A','full_union_minus_B','exact_receiver_missing_primary','exact_receiver_missing_flipped'})

    def test_idempotent_json_verification_preserves_existing_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            self.m.STUDY=Path(td)
            p=Path(td)/'record.json';original='{"b": 2, "a": 1}';p.write_text(original)
            try:self.m.write_json('record.json',{'a':1,'b':2})
            except FileExistsError:self.fail('Semantically identical JSON must verify without overwrite')
            self.assertEqual(p.read_text(),original)
            with self.assertRaises(FileExistsError):self.m.write_json('record.json',{'a':9,'b':2})

    def test_no_human_identity_never_qualifies(self):
        result=self.m.human_gate({'reviewers':[],'adjudicator':None}, [], [], [])
        self.assertEqual(result['qualified_count'],0)
        self.assertEqual(result['paid_gate'],'STOP')

if __name__=='__main__':unittest.main()
