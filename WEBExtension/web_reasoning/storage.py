"""Small fsynced primitives adapted from the preserved project storage code."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(obj):return hashlib.sha256(canonical(obj).encode()).hexdigest()


def fsync_dir(directory):
    fd=os.open(directory,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
            temporary=Path(f.name);f.write((canonical(obj)+'\n').encode());f.flush();os.fsync(f.fileno())
        os.replace(temporary,path);fsync_dir(path.parent)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()


class Journal:
    def __init__(self,path):self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
    def append(self,event):
        with self.path.open('ab') as f:f.write((canonical(event)+'\n').encode());f.flush();os.fsync(f.fileno())
        fsync_dir(self.path.parent)
    def read(self):
        if not self.path.exists():return []
        data=self.path.read_bytes()
        if data and not data.endswith(b'\n'):raise ValueError('damaged_journal')
        return [json.loads(line) for line in data.splitlines()]
