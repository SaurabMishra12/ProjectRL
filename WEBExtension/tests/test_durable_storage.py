"""Storage/control-flow fixtures only; no inference or experimental predictions."""
import sys
import json
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.durable_storage import durable_json, archive_file, blocking_attempts
from backend import unresolved_attempts

class DurableStorageTests(unittest.TestCase):
    def test_json_is_committed_and_invalid_value_preserves_old_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'record.json';durable_json(p,{'counter':1})
            self.assertEqual(json.loads(p.read_text()),{'counter':1})
            with self.assertRaises(TypeError):durable_json(p,{'bad':object()})
            self.assertEqual(json.loads(p.read_text()),{'counter':1})
            self.assertEqual(len(list(Path(tmp).glob('*.tmp'))),0)
    def test_archive_survives_inplace_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'checkpoint.pt';p.write_bytes(b'first state fixture')
            archived=archive_file(p,Path(tmp)/'archive')
            p.write_bytes(b'second state fixture')
            self.assertEqual(Path(archived['path']).read_bytes(),b'first state fixture')
            second=archive_file(p,Path(tmp)/'archive')
            self.assertNotEqual(archived['sha256'],second['sha256'])
    def records(self):
        return [{'event':'started','attempt_id':'old','key':'k','request_sha256':'sha','utc':'t'},
                {'event':'started','attempt_id':'new','key':'k','request_sha256':'sha','utc':'later'}]
    def test_authorized_retry_does_not_erase_original_unknown_liability(self):
        records=self.records();allowed={'old':{'key':'k','request_sha256':'sha','utc':'t'}}
        self.assertEqual([r['attempt_id'] for r in blocking_attempts(records,allowed)],['new'])
        self.assertEqual(len(unresolved_attempts(records)),2)
    def test_authorization_mismatch_still_blocks(self):
        self.assertEqual(len(blocking_attempts(self.records(),{'old':{'key':'WRONG','request_sha256':'sha','utc':'t'}})),2)

if __name__=='__main__':unittest.main()
