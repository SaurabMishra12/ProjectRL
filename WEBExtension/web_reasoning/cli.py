"""Small command boundary; preparation and engineering execution require no credentials."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .preparation import prepare_run,verify_run,forecast
from .workflow import run_manifest
from .reporting import write_report
from .serialization import run_from_dict,targets_from_dict,public_from_dict
from .config import validate_config,uses_adaptive
from .storage import atomic_json,digest


def main(argv=None):
    parser=argparse.ArgumentParser(description='Auditable SparseComm Web evidence experiments; mock by default.')
    commands=parser.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare');prepare.add_argument('--config',required=True);prepare.add_argument('--sources',required=True);prepare.add_argument('--output',required=True)
    planned=commands.add_parser('forecast');planned.add_argument('--manifest',required=True)
    run=commands.add_parser('run');run.add_argument('--manifest',required=True);run.add_argument('--mode',choices=('mock','live'),default='mock');run.add_argument('--cohort',choices=('pilot','development','evaluation','scaling'),default='pilot');run.add_argument('--permit')
    verify=commands.add_parser('verify');verify.add_argument('--run',required=True)
    report=commands.add_parser('report');report.add_argument('--run',required=True);report.add_argument('--cohort',choices=('pilot','development','evaluation','scaling'),required=True);choice=report.add_mutually_exclusive_group(required=True);choice.add_argument('--engineering',action='store_true');choice.add_argument('--scientific',action='store_true')
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
        root=Path(args.run);verified=verify_run(root);manifest=json.loads((root/'manifest.json').read_text())
        config=validate_config(manifest['config'])
        if args.scientific and config.mode!='REAL':raise ValueError('mock_is_not_scientific_evidence')
        source_cohort='evaluation' if args.cohort=='scaling' else args.cohort
        targets={key:targets_from_dict(t) for key,t in json.loads((root/(source_cohort+'_targets.json')).read_text()).items()}
        publics={p['key']:public_from_dict(p) for p in json.loads((root/(source_cohort+'_inputs.json')).read_text())}
        keys=manifest['scaling_keys'] if args.cohort=='scaling' else sorted(publics)
        expected=[(key,n) for n in (config.team_sizes if args.cohort=='scaling' else (3,)) for key in keys]
        runs=[]
        for key,n in expected:
            path=root/'results'/f'{digest([source_cohort,key,n])}.json'
            if path.exists():runs.append(run_from_dict(json.loads(path.read_text())['run']))
        expected_arms=('nocomm','private','static','random','selected','instar','dense','full_union')
        if config.experiment=='W4' and uses_adaptive(config):expected_arms+=('selected_observable_stop','selected_always_run')
        print(write_report(runs,targets,root/'reports'/args.cohort,args.scientific,publics,
                           {'sources':manifest['sources'],'frozen_files':manifest['frozen_files'],'code_hashes':manifest['code_hashes'],'tokenizer':manifest['tokenizer'],
                            'physical_usage_all_cohorts':verified['physical_usage']},expected_cells=expected,cohort=args.cohort,dataset=manifest['sources']['dataset'],mode=config.mode,expected_arms=expected_arms))
    return 0
