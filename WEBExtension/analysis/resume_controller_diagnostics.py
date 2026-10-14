"""Manual resume with sanitized provider error details; no automatic retries."""
import sys
import os
import json
import runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend import RealBackend
from protocol import write_json

original=RealBackend.complete
def diagnosed(self,*args,**kwargs):
    try:
        return original(self,*args,**kwargs)
    except Exception as error:
        cause=error
        while cause.__cause__ is not None:cause=cause.__cause__
        body=getattr(cause,'body',None)
        if body is not None:
            encoded=json.dumps(body)
            for name,value in os.environ.items():
                if any(token in name.upper() for token in ['KEY','TOKEN','SECRET']) and len(value)>8:
                    encoded=encoded.replace(value,'[REDACTED]')
            diagnostics={'error_type':type(cause).__name__,'http_status':getattr(cause,'status_code',None),
                         'provider_error_body':json.loads(encoded)}
            write_json(self.run_dir/'provider_error_diagnostic.json',diagnostics)
            print(json.dumps(diagnostics),flush=True)
        raise
RealBackend.complete=diagnosed
sys.argv=['run_controller_machine.py','all']
runpy.run_path(str(Path(__file__).resolve().parents[1]/'run_controller_machine.py'),run_name='__main__')
