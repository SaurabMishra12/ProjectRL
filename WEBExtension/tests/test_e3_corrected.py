import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e3_corrected import content_pair, evidence_bank

class ByteTokenizer:
    def encode(self,text): return list(text.encode())
    def decode(self,ids): return bytes(ids).decode(errors='replace')
    def count(self,text): return len(self.encode(text))

class ContentPairTests(unittest.TestCase):
    def test_short_message_cannot_silently_send_identical_payloads(self):
        with self.assertRaises(ValueError): content_pair('x',256,ByteTokenizer(),23)
    def test_distinct_equal_length_real_substrings_and_deterministic(self):
        text='Evidence: The first fact.\nEvidence: A second distinct fact.\n'
        a,b=content_pair(text,12,ByteTokenizer(),23)
        self.assertEqual(a['tokens'],b['tokens'])
        self.assertEqual(a['tokens'],12)
        self.assertNotEqual(a['text'],b['text'])
        self.assertIn(a['text'],text);self.assertIn(b['text'],text)
        self.assertEqual((a,b),content_pair(text,12,ByteTokenizer(),23))
    def test_no_gold_input_and_no_citation_dependent_bank(self):
        state={'assignment':[[{'title':'Article','sentences':['First fact.','Second fact.']}]],'responses':[{'answer':'noanswer','citations':[]}]}
        text=evidence_bank(state)
        self.assertIn('First fact.',text);self.assertIn('Second fact.',text)
        self.assertNotIn('noanswer',text)
    def test_invalid_budget_rejected(self):
        with self.assertRaises(ValueError): content_pair('abcdef',0,ByteTokenizer(),23)

if __name__=='__main__': unittest.main()
