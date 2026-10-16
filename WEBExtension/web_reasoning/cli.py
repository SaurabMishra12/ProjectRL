"""Small command boundary; preparation and engineering execution require no credentials."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .preparation import prepare_run,verify_run,forecast
from .workflow import run_manifest
from .reporting import write_report
from .serialization import run_from_dict,targets_from_dict,public_from_dict
from .config import validate_config
from .storage import atomic_json


def main(argv=None):
    parser=argparse.ArgumentParser(description='Auditable SparseComm Web evidence experiments; mock by default.')
    commands=parser.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare');prepare.add_argument('--config',required=True);prepare.add_argument('--sources',required=True);prepare.add_argument('--output',required=True)
    planned=commands.add_parser('forecast');planned.add_argument('--manifest',required=True)
    run=commands.add_parser('run');run.add_argument('--manifest',required=True);run.add_argument('--mode',choices=('mock','live'),default='mock');run.add_argument('--cohort',choices=('pilot','development','evaluation','scaling'),default='pilot');run.add_argument('--permit')
    verify=commands.add_parser('verify');verify.add_argument('--run',required=True)
    report=commands.add_parser('report');report.add_argument('--run',required=True);choice=report.add_mutually_exclusive_group(required=True);choice.add_argument('--engineering',action='store_true');choice.add_argument('--scientific',action='store_true')
    variants=commands.add_parser('variants');variants.add_argument('--config',required=True);variants.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':print(prepare_run(args.config,args.sources,args.output))
    elif args.command=='forecast':
        path=Path(args.manifest);verify_run(path.parent);print(json.dumps(forecast(json.loads(path.read_text())),indent=2))
    elif args.command=='run':
        runs=run_manifest(args.manifest,args.mode,args.cohort,args.permit);print(json.dumps({'question_team_records':len(runs),'mode':args.mode,'paid_calls_authorized_by_this_command':args.mode=='live'}))
    elif args.command=='verify':print(json.dumps(verify_run(args.run),indent=2))
    elif args.command=='variants':
        from .ablations import ablation_variants
        config=validate_config(json.loads(Path(args.config).read_text()));directory=Path(args.output);directory.mkdir(parents=True,exist_ok=True)
        for i,c in enumerate(ablation_variants(config)):atomic_json(directory/f'variant_{i}.json',c.to_dict())
        print(directory)
    else:
        root=Path(args.run);verify_run(root);manifest=json.loads((root/'manifest.json').read_text());targets={};publics={}
        for cohort in ('pilot','development','evaluation'):
            targets.update({key:targets_from_dict(t) for key,t in json.loads((root/(cohort+'_targets.json')).read_text()).items()})
            publics.update({p['key']:public_from_dict(p) for p in json.loads((root/(cohort+'_inputs.json')).read_text())})
        runs=[run_from_dict(json.loads(p.read_text())['run']) for p in sorted((root/'results').glob('*.json'))]
        print(write_report(runs,targets,root/'reports',args.scientific,publics,
                           {'sources':manifest['sources'],'frozen_files':manifest['frozen_files'],'code_hashes':manifest['code_hashes'],'tokenizer':manifest['tokenizer']}))
    return 0
