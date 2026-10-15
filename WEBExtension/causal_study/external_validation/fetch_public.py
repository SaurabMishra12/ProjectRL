"""Explicit public, read-only GET audit; never imports or contacts inference providers."""
from pathlib import Path
import datetime, hashlib, json, urllib.request
ROOT=Path(__file__).resolve().parent

def get(url,name):
    path=ROOT/'sources'/name; path.parent.mkdir(parents=True,exist_ok=True)
    log={'url':url,'path':str(path.relative_to(ROOT)),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'ProjectRL-offline-research-audit'})
        with urllib.request.urlopen(req,timeout=45) as response:
            data=response.read(32*1024*1024+1)
            if len(data)>32*1024*1024: raise ValueError('32 MiB public-file limit exceeded')
            path.write_bytes(data);log.update(status=response.status,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    except Exception as exc: log.update(status='ACCESS_FAILED',error=str(exc))
    with (ROOT/'sources/fetch_log.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
    return path if log['status']==200 else None

if __name__=='__main__':
    for repo in ['Wang-ML-Lab/when-debate-helps','jonathansantilli/freemad','moozwen/sakana-debate']:
        info=get('https://api.github.com/repos/'+repo,repo.replace('/','__')+'_info.json')
        if info:
            branch=json.loads(info.read_text())['default_branch']
            get('https://api.github.com/repos/'+repo+'/git/trees/'+branch+'?recursive=1',repo.replace('/','__')+'_tree.json')
    for repo,paths in {'stonybrooknlp/musique':['README.md','LICENSE','evaluate_v1.0.py','download_data.sh','download_predictions.sh','metrics/answer.py','metrics/group.py'], 'deeplearning-wisc/debate-or-vote':['README.md','src/evaluator.py','src/main.py']}.items():
        tree=json.loads((ROOT/'sources'/(repo.replace('/','__')+'_tree.json')).read_text());commit=tree['sha']
        valid={x['path'] for x in tree['tree']}
        for path in paths:
            if path in valid:get('https://raw.githubusercontent.com/'+repo+'/'+commit+'/'+path,repo.replace('/','__')+'/'+path)
