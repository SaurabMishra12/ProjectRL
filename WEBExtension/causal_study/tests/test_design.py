"""Protocol invariants: these guard leakage and invalid experimental contrasts."""
import copy
import json
import unittest

from WEBExtension.causal_study import design


def fixture(i=0):
    return {
        'id': f'q{i}', 'question': f'Which linked place is sought in item {i}?',
        'type': 'bridge' if i % 2 else 'comparison', 'answer': 'PRIVATE_GOLD_SENTINEL',
        'supporting_facts': [['Alpha', 0], ['Beta', 0]],
        'documents': [
            {'title': 'Alpha', 'sentences': ['Alpha linked a person to a distant city.']},
            {'title': 'Beta', 'sentences': ['Beta describes that person and their birthplace.']},
            {'title': 'Cedar', 'sentences': ['Cedar was a wooden sailing boat built in Norway.']},
            {'title': 'Delta', 'sentences': ['Delta describes a mathematical shape with three corners.']},
            {'title': 'Elm', 'sentences': ['Elm is a flowering tree found across temperate forests.']},
            {'title': 'Fir', 'sentences': ['Fir records a small bird living near the sea.']},
            {'title': 'Gum', 'sentences': ['Gum describes a device for measuring changing temperatures.']},
        ],
    }


class DesignTests(unittest.TestCase):
    def test_equal_union_exact_counts_and_actual_support_placement(self):
        manifest = design.build_manifest([fixture(i) for i in range(12)], n=8)
        labels = manifest['private']['labels']
        for q in manifest['questions']:
            unions = []
            for name, placement in q['placements'].items():
                groups = placement['assignments']
                self.assertEqual([len(g) for g in groups], [2, 2, 2])
                titles = [d['title'] for g in groups for d in g]
                self.assertEqual(len(set(titles)), 6)
                unions.append(set(titles))
                support = set(labels[q['id']]['support_titles'])
                counts = sorted(len({d['title'] for d in g} & support) for g in groups)
                self.assertEqual(counts, [0, 0, 2] if name == 'colocated' else [0, 1, 1])
            self.assertEqual(unions[0], unions[1])
        self.assertEqual(manifest['protocol']['e6_successful_completions'], 144)

    def test_sampling_is_order_independent_and_excludes_exposure(self):
        rows = [fixture(i) for i in range(12)]
        a = design.build_manifest(rows, n=8, blocked_rows=[rows[0]])
        b = design.build_manifest(list(reversed(rows)), n=8, blocked_rows=[rows[0]])
        self.assertEqual(a, b)
        self.assertNotIn('q0', [q['id'] for q in a['questions']])
        self.assertEqual(a['selection']['excluded_exposed'], 1)

    def test_missing_support_does_not_get_replaced_or_invented(self):
        row = fixture()
        row['documents'] = [d for d in row['documents'] if d['title'] != 'Beta']
        with self.assertRaisesRegex(ValueError, 'eligible'):
            design.build_manifest([row], n=1)

    def test_prompt_whitelist_prevents_gold_metadata_leak_and_shared_evidence(self):
        q = design.build_manifest([fixture()], n=1)['questions'][0]
        q['answer'] = 'PRIVATE_GOLD_SENTINEL'
        q['supporting_facts'] = [['SECRET_TITLE', 999]]
        for placement in q['placements']:
            for node in range(3):
                messages = design.initial_messages(q, placement, node)
                visible = json.loads(messages[1]['content'])
                self.assertNotIn('PRIVATE_GOLD_SENTINEL', str(messages))
                self.assertNotIn('SECRET_TITLE', str(messages))
                expected = {d['title'] for d in q['placements'][placement]['assignments'][node]}
                self.assertEqual({f['title'] for f in visible['evidence']}, expected)
                self.assertEqual(visible['peer_messages'], [])

    def test_nocomm_and_broadcast_revisions_fork_same_initial_state(self):
        q = design.build_manifest([fixture()], n=1)['questions'][0]
        initial = []
        for group in q['placements']['distributed']['assignments']:
            initial.append({'answer': group[0]['title'], 'citations': [[group[0]['title'], 0]],
                            'gold': 'PRIVATE_GOLD_SENTINEL'})
        before = copy.deepcopy(initial)
        for node in range(3):
            no = json.loads(design.revision_messages(q, 'distributed', node, initial, 'nocomm')[1]['content'])
            co = json.loads(design.revision_messages(q, 'distributed', node, initial, 'broadcast')[1]['content'])
            self.assertEqual(no['previous_response'], co['previous_response'])
            self.assertEqual(no['evidence'], co['evidence'])
            self.assertEqual(no['peer_messages'], [])
            self.assertEqual(len(co['peer_messages']), 2)
            self.assertEqual({m['sender'] for m in co['peer_messages']}, set(range(3)) - {node})
            self.assertTrue(all(m['evidence'] for m in co['peer_messages']))
            self.assertNotIn('PRIVATE_GOLD_SENTINEL', str(co))
        self.assertEqual(initial, before)

    def test_peer_citation_must_have_been_visible_to_its_sender(self):
        q = design.build_manifest([fixture()], n=1)['questions'][0]
        groups = q['placements']['distributed']['assignments']
        initial = [{'answer': 'x', 'citations': [['NOT_SOURCE', 0]]} for _ in groups]
        msg = json.loads(design.revision_messages(q, 'distributed', 0, initial, 'broadcast')[1]['content'])
        self.assertTrue(all(not p['evidence'] for p in msg['peer_messages']))

    def test_e7_uses_whole_distinct_source_sentences_and_frozen_receiver(self):
        manifest = design.build_manifest([fixture()], n=1)
        q = manifest['questions'][0]
        initial = [{'answer': 'noanswer', 'citations': []} for _ in range(3)]
        plan = design.build_e7(q, manifest['private']['labels'][q['id']], initial)
        self.assertTrue(plan['eligible'], plan)
        receiver = plan['receiver']
        local = {d['title'] for d in q['placements']['distributed']['assignments'][receiver]}
        self.assertNotIn(plan['arms']['relevant']['facts'][0]['title'], local)
        self.assertIn(plan['arms']['redundant']['facts'][0]['title'], local)
        texts = []
        for arm in ['relevant', 'redundant', 'distractor']:
            facts = plan['arms'][arm]['facts']
            self.assertEqual(len(facts), 1)
            fact = facts[0]
            doc = next(d for d in q['documents'] if d['title'] == fact['title'])
            self.assertEqual(fact['text'], doc['sentences'][fact['sentence_id']])
            messages = design.e7_messages(q, plan, initial[receiver], arm)
            payload = json.loads(messages[1]['content'])
            self.assertEqual(payload['previous_response'], initial[receiver])
            self.assertEqual(len(payload['peer_messages']), 1)
            texts.append(fact['text'])
            self.assertNotIn('PRIVATE_GOLD_SENTINEL', str(messages))
            self.assertNotIn('relevant', messages[1]['content'])
        self.assertEqual(len(set(texts)), 3)

    def test_e7_rejects_duplicate_content_across_supporting_articles(self):
        row = fixture()
        row['documents'][1]['sentences'] = list(row['documents'][0]['sentences'])
        manifest = design.build_manifest([row], n=1)
        q = manifest['questions'][0]
        plan = design.build_e7(q, manifest['private']['labels'][q['id']],
                               [{'answer': 'x', 'citations': []} for _ in range(3)])
        self.assertFalse(plan['eligible'])

    def test_e7_missing_fact_must_not_be_duplicated_in_receiver_other_documents(self):
        manifest = design.build_manifest([fixture()], n=1)
        q = manifest['questions'][0]
        label = manifest['private']['labels'][q['id']]
        support_text = [d['sentences'][0] for d in q['documents'] if d['title'] in label['support_titles']]
        for doc in q['documents']:
            if doc['title'] not in label['support_titles']:
                doc['sentences'].extend(support_text)
        for group in q['placements']['distributed']['assignments']:
            for doc in group:
                if doc['title'] not in label['support_titles']:
                    doc['sentences'].extend(support_text)
        plan = design.build_e7(q, label, [{'answer': 'x', 'citations': []} for _ in range(3)])
        self.assertFalse(plan['eligible'])

    def test_preflight_rejects_corrupted_union_before_any_call(self):
        manifest = design.build_manifest([fixture()], n=1)
        result = design.validate_manifest(manifest)
        self.assertEqual(result['questions'], 1)
        self.assertEqual(result['required_completions'], 18)
        bad = copy.deepcopy(manifest)
        bad['questions'][0]['placements']['distributed']['assignments'][0][0]['sentences'][0] = 'Invented text.'
        with self.assertRaisesRegex(ValueError, 'source'):
            design.validate_manifest(bad)

    def test_primary_endpoint_and_aggregation_do_not_use_gold_or_llm(self):
        self.assertEqual(design.majority([{'answer': 'The Paris'}, {'answer': 'Paris'}, {'answer': 'Rome'}]), 'The Paris')
        self.assertEqual(design.majority([{'answer': 'A'}, {'answer': 'B'}, {'answer': 'C'}]), 'A')
        self.assertEqual(design.interaction({'distributed_comm': 75, 'distributed_nocomm': 25,
                                             'colocated_comm': 50, 'colocated_nocomm': 40}), 40)


if __name__ == '__main__':
    unittest.main()
