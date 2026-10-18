"""Save a verified local archive of all 150-question result artifacts."""
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import tarfile

from WEBExtension.web_reasoning.preparation import file_sha
from WEBExtension.web_reasoning.storage import atomic_json

OUT=Path(__file__).resolve().parent


def build_archive(root,paths,output):
    root=Path(root).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('preserve_existing_result_archive')
    inventory={str(p.resolve().relative_to(root)):file_sha(p) for p in sorted(set(map(Path,paths)))}
    payload=json.dumps({'files':inventory,'inventory_excludes_itself':True},sort_keys=True,indent=2).encode()+b'\n'
    output.parent.mkdir(parents=True,exist_ok=True);temp=output.with_suffix(output.suffix+'.partial')
    with tarfile.open(temp,'w:gz') as archive:
        for name in inventory:archive.add(root/name,arcname=name,recursive=False)
        info=tarfile.TarInfo('member_inventory.json');info.size=len(payload)
        archive.addfile(info,io.BytesIO(payload))
    with tarfile.open(temp,'r:gz') as archive:
        assert set(archive.getnames())==set(inventory)|{'member_inventory.json'}
        assert json.load(archive.extractfile('member_inventory.json'))['files']==inventory
        for name,expected in inventory.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==expected,name
    temp.replace(output);sha=file_sha(output)
    output.with_suffix(output.suffix+'.sha256').write_text(sha+'  '+output.name+'\n')
    return {'archive':str(output),'sha256':sha,'members_verified':len(inventory),'bytes':output.stat().st_size}


def main():
    root=Path.cwd().resolve()
    audit=json.loads((OUT/'audit.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    assert audit['status']=='VERIFIED_150_OFFLINE' and audit['sealed_questions']==150
    assert summary['status']=='COMPLETE_150' and summary['completed_questions']==150
    versions={}
    for name in ('openai','filelock','transformers','tokenizers','numpy','scipy','pytest','matplotlib'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    atomic_json(OUT/'environment.json',{'python':platform.python_version(),'packages':versions})
    directories=[root/'WEBExtension/web_reasoning/artifacts'/name for name in (
        '2wiki_revised_20261016','2wiki_150_20261017','2wiki_150_resume_20261017')]
    directories += [root/'WEBExtension/pilots'/name for name in (
        '2wiki_revised_20261016','2wiki_150_20261017','2wiki_150_resume_20261017','2wiki_150_results_20261017')]
    paths=[]
    for directory in directories:
        paths.extend(p for p in directory.rglob('*') if p.is_file() and not p.is_symlink()
                     and not set(p.parts)&{'__pycache__','.mplcache'}
                     and p.name!='runtime.lock' and not p.name.endswith('.partial'))
    package=root/'WEBExtension/web_reasoning'
    paths.extend(p for p in package.rglob('*.py') if '__pycache__' not in p.parts)
    paths.append(package/'references/sources.json')
    receipt=build_archive(root,paths,root/'WEBExtension/web_reasoning/artifacts/exports/2wiki_150_complete_20261017.tar.gz')
    atomic_json(OUT/'archive_receipt.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
