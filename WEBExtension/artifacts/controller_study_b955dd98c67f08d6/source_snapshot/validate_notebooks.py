"""Execute notebook no-inference paths; preserve outputs for review."""
import datetime
import importlib.metadata
import json
import re
import traceback
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from protocol import ROOT, write_json, file_hash

def validate():
    records=[]
    for p in sorted(ROOT.glob('*.ipynb')):
        nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
        # A validation run never turns on inference, even if a user previously edited a cell.
        for cell in nb.cells:
            if cell.cell_type=='code':
                cell.source=re.sub(r'\bEXECUTE\s*=\s*True\b', 'EXECUTE = False', cell.source)
        try:
            NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
            nbformat.write(nb,p)
            records.append({'notebook':p.name,'status':'PASS_NO_INFERENCE','sha256':file_hash(p)})
            print(p.name,'PASS_NO_INFERENCE',flush=True)
        except Exception as e:
            records.append({'notebook':p.name,'status':'FAIL','error':str(e)})
            write_json(ROOT/'artifacts/notebook_validation.json',records)
            raise
    write_json(ROOT/'artifacts/notebook_validation.json',records)
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','matplotlib','openai','transformers','pyarrow','nbformat','nbclient','ipykernel']}
    write_json(ROOT/'artifacts/environment_versions.json',versions)
    return records
if __name__=='__main__':validate()
