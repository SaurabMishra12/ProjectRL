"""Crash-resistant local storage; no inference or experimental substitutions."""
import hashlib
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from backend import unresolved_attempts as original_unresolved_attempts

def fsync_directory(directory):
    fd=os.open(str(directory),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def durable_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as f:
            temporary=Path(f.name);f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path);fsync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
    return {'path':str(path.resolve()),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

def durable_json(path,value):
    data=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
    return durable_bytes(path,data)

def archive_file(source,directory):
    source=Path(source);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    def checksum(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            while chunk:=f.read(1<<20):h.update(chunk)
        return h.hexdigest()
    expected=checksum(source);destination=directory/(source.name+'.'+expected)
    if not destination.exists():
        temporary=destination.with_suffix('.tmp')
        try:
            shutil.copyfile(source,temporary)
            with temporary.open('rb') as f:os.fsync(f.fileno())
            if checksum(temporary)!=expected:raise ValueError('Source changed during snapshot')
            os.replace(temporary,destination);fsync_directory(directory)
        finally:
            if temporary.exists():temporary.unlink()
    if checksum(destination)!=expected:raise ValueError('Archive checksum mismatch')
    return {'source':str(source.resolve()),'path':str(destination.resolve()),
            'sha256':expected,'bytes':destination.stat().st_size}

def blocking_attempts(records,authorized):
    # Authorization permits a NEW attempt. It never declares the old one unbilled
    # or completed, and the original unresolved-liability audit remains unchanged.
    result=[]
    for r in original_unresolved_attempts(records):
        a=authorized.get(r['attempt_id'])
        if a is None or any(a.get(k)!=r.get(k) for k in ['key','request_sha256','utc']):result.append(r)
    return result

class Journal:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.lock=threading.Lock()
    def append(self,value):
        line=(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n').encode()
        with self.lock,self.path.open('ab') as f:
            f.write(line);f.flush();os.fsync(f.fileno())
