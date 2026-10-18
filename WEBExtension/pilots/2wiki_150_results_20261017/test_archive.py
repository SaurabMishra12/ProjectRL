"""Saving results must preserve exact file bytes and verify every member."""
import importlib
import hashlib
import json
import tarfile


def test_saved_archive_preserves_exact_results_and_checksums(tmp_path):
    module=importlib.import_module('WEBExtension.pilots.2wiki_150_results_20261017.archive')
    root=tmp_path/'source';root.mkdir();(root/'result.json').write_bytes(b'abc\n')
    out=tmp_path/'results.tar.gz'
    receipt=module.build_archive(root,[root/'result.json'],out)
    assert receipt['members_verified']==1
    with tarfile.open(out,'r:gz') as archive:
        assert archive.extractfile('result.json').read()==b'abc\n'
        inventory=json.load(archive.extractfile('member_inventory.json'))
        assert inventory['files']['result.json']==hashlib.sha256(b'abc\n').hexdigest()
    assert out.with_suffix(out.suffix+'.sha256').read_text().split()[0]==hashlib.sha256(out.read_bytes()).hexdigest()
