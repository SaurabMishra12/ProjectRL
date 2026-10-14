"""Download publisher-hosted, pinned real HotpotQA; never generate fallback examples."""
import datetime
import json
import urllib.request
from pathlib import Path
from protocol import ROOT, DATA_URL, PUBLISHER_PARQUET_SHA256, file_hash, write_json
REVISION = '1908d6afbbead072334abe2965f91bd2709910ab'
PARQUET_URL = f'https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{REVISION}/fullwiki/validation-00000-of-00001.parquet'

def prepare():
    import pyarrow.parquet as pq
    source = ROOT / 'data/hotpot_fullwiki_validation.parquet'
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        temp = source.with_suffix('.download')
        with urllib.request.urlopen(PARQUET_URL, timeout=60) as response, open(temp, 'wb') as f:
            while block := response.read(1 << 20): f.write(block)
        temp.replace(source)
    if file_hash(source) != PUBLISHER_PARQUET_SHA256: raise ValueError('Publisher parquet SHA-256 differs from verified pinned file')
    rows = pq.read_table(source).to_pylist()
    if len(rows) != 7405: raise ValueError('Incomplete official validation set')
    output = ROOT / 'data/hotpot_dev_fullwiki_v1.json'
    write_json(output, rows)
    meta = {'source_url': DATA_URL, 'actual_download_url': PARQUET_URL,
            'publisher': 'hotpotqa/hotpot_qa', 'revision': REVISION, 'split': 'validation',
            'configuration': 'fullwiki', 'source_sha256': file_hash(source), 'sha256': file_hash(output),
            'rows': len(rows), 'prepared_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'license': 'CC BY-SA 4.0', 'transformation': 'parquet to JSON; no text/label modification',
            'caveat': 'Original CMU host unavailable at preparation; publisher-hosted pinned mirror used.'}
    write_json(output.with_suffix('.provenance.json'), meta)
    return meta
if __name__ == '__main__': print(json.dumps(prepare(), indent=2))
