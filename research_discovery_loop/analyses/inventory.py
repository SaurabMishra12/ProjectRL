"""Read-only repository inventory and immutable historical baseline; no model imports."""
from pathlib import Path
import os,json,hashlib,collections,datetime
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
SKIP={'.git','__pycache__','.pytest_cache','node_modules','.venv','.venv_linux','.superpowers','research_discovery_loop'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 inventory=[];protect={};types=collections.Counter()
 for directory,dirs,files in os.walk(REPO):
  dirs[:]=[d for d in dirs if d not in SKIP]
  for name in files:
   p=Path(directory)/name
   if p.is_symlink():continue
   try:s=p.stat()
   except OSError:continue
   rel=str(p.relative_to(REPO));types[p.suffix or '<none>']+=1
   entry={'path':rel,'bytes':s.st_size,'extension':p.suffix,'inspection':'INVENTORIED_NOT_CONTENT_REVIEWED'}
   if p.suffix in {'.py','.md','.tex','.sh','.yaml','.yml','.toml'} or (p.suffix in {'.json','.jsonl','.csv'} and s.st_size<2*1024*1024 and not any(k in p.parts for k in ['responses','requests','attempts','network_attempts'])):
    entry['sha256']=sha(p);protect[rel]=entry['sha256']
   inventory.append(entry)
 base=REPO/'WEBExtension/causal_study/external_validation'
 for rel,digest in json.loads((base/'historical_snapshot.json').read_text()).items():
  p=Path(rel);assert sha(p)==digest
  protect[str(p.relative_to(REPO))]=digest
 for item in json.loads((base/'artifact_manifest.json').read_text())['files']:
  p=base/item['path'];assert sha(p)==item['sha256'];protect[str(p.relative_to(REPO))]=item['sha256']
 protect[str((base/'artifact_manifest.json').relative_to(REPO))]=sha(base/'artifact_manifest.json')
 # Additional exact input hashes used by the final-findings bundle are preserved.
 manifest=REPO/'WEBExtension/analysis/final_findings_20261014/final_manifest.json'
 for rel,digest in json.loads(manifest.read_text())['inputs_sha256'].items():
  p=Path(rel);assert sha(p)==digest
  key=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)
  protect[key]=digest
 (ROOT/'results/repository_inventory.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':inventory,'extension_counts':dict(types),'scope_exclusions':sorted(SKIP),'not_a_claim_all_file_bodies_were_read':True},indent=2)+'\n')
 (ROOT/'results/historical_baseline.json').write_text(json.dumps(protect,indent=2)+'\n')
 print('Inventoried',len(inventory),'files; protected',len(protect),'source/research/manifest-backed artifacts; no historical writes.')
if __name__=='__main__':main()
