"""Serial authorized E1-E4 real runs followed by authenticated pooled/stratified reports."""
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import load_config, freeze, run
from protocol import write_json, file_hash
from reporting import report
from analysis.gold_strata import analyze

cfg=load_config();directory,_=freeze(cfg)
write_json(directory/'machine_pipeline.json',{'status':'RUNNING','runner_sha256':file_hash(Path(__file__))})
try:
    for experiment in ['E1','E2','E3','E4']:
        run(experiment,cfg,execute=True)
        report(directory,experiment)
        analyze(directory,experiment)
    write_json(directory/'machine_pipeline.json',{'status':'COMPLETE','runner_sha256':file_hash(Path(__file__))})
except Exception as error:
    write_json(directory/'machine_pipeline.json',{'status':'INCOMPLETE','error_type':type(error).__name__,
                                                 'runner_sha256':file_hash(Path(__file__))})
    print('Controlled run stopped:',type(error).__name__,flush=True)
    raise SystemExit(1)
