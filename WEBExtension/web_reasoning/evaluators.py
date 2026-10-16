"""Hash-verified official evaluator functions; no semantic judge or answer rewriting."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from functools import lru_cache

REFERENCES=Path(__file__).parent/'references'


def load_reference(dataset,source_manifest=None):
    if dataset not in ('2wiki','finqa'):raise ValueError('unsupported_evaluator')
    manifest=Path(source_manifest) if source_manifest else REFERENCES/'sources.json'
    entries=json.loads(manifest.read_text());name=dataset+'_official.py'
    entry=next(x for x in entries if x['path']==name);path=manifest.parent/name
    source=path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=entry['sha256']:raise ValueError('official_source_hash_mismatch')
    # Only a JSON import compatibility shim changes. Upstream metric/program functions are unchanged.
    return compile_reference(dataset,str(path),source.decode())


@lru_cache(maxsize=4)
def compile_reference(dataset,path,source):
    text=source.replace('import ujson as json','import json')
    namespace={'__name__':'web_reasoning_official_'+dataset,'__file__':str(path)}
    exec(compile(text,str(path),'exec'),namespace)
    return SimpleNamespace(**namespace)


class OfficialEvaluator:
    def __init__(self,dataset,source_manifest=None,alias_file=None):
        self.dataset=dataset;self.module=load_reference(dataset,source_manifest)
        self.alias_file=alias_file
        if alias_file is not None:
            raw=Path(alias_file).read_bytes();self.alias_sha256=hashlib.sha256(raw).hexdigest()

    def score(self,public_key,outcome,targets,public=None,visible_atoms=None):
        if outcome.kind in ('invalid','unresolved'):
            return {'em':None,'f1':None,'status':'INCOMPLETE','reason':outcome.reason}
        prediction=outcome.proposal.answer if outcome.proposal and outcome.proposal.answer else 'noanswer'
        if self.dataset=='finqa':
            if not outcome.proposal or outcome.proposal.status!='supported' or not outcome.proposal.program:return {'execution_accuracy':0.,'program_accuracy':0.,'status':'COMPLETE'}
            from .finqa import score_official_program
            return score_official_program(outcome.proposal,targets,public,self.module,visible_atoms)
        golds=(targets.answer,)+targets.aliases
        answer_scores=[self.module.eval_answer(prediction,gold) for gold in golds]
        result={'em':100.*max(x[0] for x in answer_scores),'f1':100.*max(x[1] for x in answer_scores),
                'status':'COMPLETE','support_f1':None,'support_em':None}
        if public is not None:
            lookup={a.atom_id:(s.title,a.sentence_id) for s in public.sources for a in s.atoms}
            predicted=[lookup[f.atom_id] for f in outcome.proposal.facts if f.atom_id in lookup] if outcome.proposal else []
            metrics=dict(sp_em=0.,sp_f1=0.,sp_prec=0.,sp_recall=0.)
            self.module.update_sp(metrics,predicted,targets.supporting_locators)
            result.update(support_f1=100*metrics['sp_f1'],support_em=100*metrics['sp_em'])
        return result
