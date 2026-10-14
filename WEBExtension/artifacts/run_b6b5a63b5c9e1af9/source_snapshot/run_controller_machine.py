"""Run only real controller phases on the configured machine; never print credentials."""
import argparse
import json
from pathlib import Path
from protocol import ROOT, write_json
from experiments import load_config
from controller_study import forecast, pilot, train, evaluate, study_directory

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['forecast','pilot','train','evaluate','all'])
    args=parser.parse_args();base=load_config();settings=json.loads((ROOT/'controller_config.json').read_text())
    if args.phase=='forecast':print(json.dumps(forecast(base,settings),indent=2));return
    directory,_=study_directory(base,settings)
    try:
        if args.phase in ['pilot','all']:print(pilot(base,settings,execute=True))
        if args.phase in ['train','all']:print(train(base,settings,execute=True))
        if args.phase in ['evaluate','all']:print(evaluate(base,settings,execute=True))
    except Exception as e:
        # Record sanitized provider status; raw requests/responses are handled by the real backend.
        cause=e
        while cause.__cause__ is not None:cause=cause.__cause__
        diagnostics={'status':'BLOCKED_OR_FAILED','phase':args.phase,'exception':type(e).__name__,
                     'root_error_type':type(cause).__name__,'http_status':getattr(cause,'status_code',None),
                     'note':'No credential or raw exception printed. Inspect recorded requests/provider response artifacts.'}
        write_json(directory/'machine_run_status.json',diagnostics)
        print(json.dumps(diagnostics,indent=2));raise SystemExit(1)
if __name__=='__main__':main()
