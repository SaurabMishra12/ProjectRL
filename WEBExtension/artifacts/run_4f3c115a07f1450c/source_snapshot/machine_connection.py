"""Use existing project Azure connection without copying or printing credentials."""
import ast
import os
import shlex
from pathlib import Path

def load_shell_azure_exports():
    allowed={'AZURE_OPENAI_API_KEY','AZURE_OPENAI_ENDPOINT','AZURE_OPENAI_BASE_URL','AZURE_OPENAI_API_VERSION'}
    path=Path.home()/'.bashrc'
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip().startswith('export AZURE_OPENAI_'): continue
            try: parts=shlex.split(line,comments=True)
            except ValueError: continue
            if len(parts)==2 and '=' in parts[1]:
                name,value=parts[1].split('=',1)
                if name in allowed and value and '$' not in value and '`' not in value: os.environ.setdefault(name,value)

def azure_connection():
    load_shell_azure_exports()
    if os.environ.get('AZURE_OPENAI_API_KEY') and os.environ.get('AZURE_OPENAI_ENDPOINT'):
        return {'api_key':os.environ['AZURE_OPENAI_API_KEY'],'azure_endpoint':os.environ['AZURE_OPENAI_ENDPOINT']}
    path=Path(__file__).resolve().parent.parent/'llm_backend.py'
    module=ast.parse(path.read_text())
    cls=next(x for x in module.body if isinstance(x,ast.ClassDef) and x.name=='AzureOpenAIBackend')
    calls=[x for x in ast.walk(cls) if isinstance(x,ast.Call) and ast.unparse(x.func)=='AzureOpenAI']
    if len(calls)!=1:raise RuntimeError('Ambiguous existing Azure client; configure environment explicitly')
    constants={k.arg:k.value.value for k in calls[0].keywords if isinstance(k.value,ast.Constant)}
    if not all(constants.get(k) for k in ['api_key','azure_endpoint']):raise RuntimeError('Configure Azure environment')
    return {k:constants[k] for k in ['api_key','azure_endpoint']}

def check_connection():
    from openai import AzureOpenAI
    connection=azure_connection()
    base_url=os.environ.get('AZURE_OPENAI_BASE_URL')
    if base_url:
        from openai import OpenAI
        client=OpenAI(api_key=connection['api_key'],base_url=base_url,max_retries=0,timeout=15)
    else:
        client=AzureOpenAI(**connection,api_version='2024-02-15-preview',max_retries=0,timeout=15)
    try:
        models=client.models.list()
        return {'status':'MODEL_LIST_ACCESSIBLE','gpt_oss_model_ids':[m.id for m in models.data if 'gpt-oss' in m.id.lower()],
                'note':'Read-only connectivity check; no inference or training performed'}
    except Exception as e:
        return {'status':'CONNECTION_NOT_VERIFIED','error_type':type(e).__name__,
                'http_status':getattr(e,'status_code',None),'note':'No credential or raw exception is printed'}

if __name__=='__main__':
    import json
    print(json.dumps(check_connection(),indent=2))
