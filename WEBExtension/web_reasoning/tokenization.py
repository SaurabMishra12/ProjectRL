"""Pinned local tokenizer, with an explicitly non-scientific engineering alternative."""
import hashlib
from pathlib import Path


class UTF8FixtureTokenizer:
    identity={'kind':'mock','name':'utf8-bytes','revision':'mock-v1'}
    def count(self,text):return len(text.encode('utf-8'))
    def prefix(self,text,cap):return text.encode('utf-8')[:cap].decode('utf-8',errors='ignore')


class PinnedTokenizer:
    def __init__(self,path,revision,expected_hash):
        if not revision or revision=='main' or len(expected_hash)!=64:
            raise ValueError('pinned_tokenizer_identity_required')
        if not Path(path).is_dir():raise FileNotFoundError(path)
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),revision=revision,local_files_only=True,trust_remote_code=False)
        actual=hashlib.sha256(self.tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        if actual!=expected_hash:raise ValueError('tokenizer_hash_mismatch')
        self.identity={'kind':'pinned','name':str(path),'revision':revision,'sha256':actual}
    def count(self,text):return len(self.tokenizer.encode(text,add_special_tokens=False))
    def prefix(self,text,cap):
        ids=self.tokenizer.encode(text,add_special_tokens=False)[:cap]
        while ids:
            value=self.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            if '\ufffd' not in value and self.count(value)<=cap:return value
            ids=ids[:-1]
        return ''
