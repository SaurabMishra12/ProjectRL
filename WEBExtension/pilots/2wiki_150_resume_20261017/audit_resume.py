"""Run the independent auditor on the approved child and check timeout lineage."""
import importlib
import json
from WEBExtension.web_reasoning.storage import canonical,atomic_json
from WEBExtension.web_reasoning.preparation import file_sha


def main():
    resume=importlib.import_module('WEBExtension.pilots.2wiki_150_resume_20261017.resume')
    resume.activate()
    audit=importlib.import_module('WEBExtension.pilots.2wiki_150_results_20261017.audit')
    audit.RUN=resume.ROOT
    audit.main()
    m=resume.base.verify();old=m['quarantined_start'];aid=old['attempt_id']
    previous=[json.loads(l) for l in (resume.ROOT/'parent_calls_full.jsonl').read_text().splitlines()]
    prefix=''.join(canonical(e)+'\n' for e in previous if e['attempt_id']!=aid).encode()
    assert (resume.ROOT/'inference/calls.jsonl').read_bytes().startswith(prefix)
    assert not (resume.PARENT/'inference/attempts'/(aid+'.json')).exists()
    events=[json.loads(l) for l in (resume.ROOT/'inference/calls.jsonl').read_text().splitlines()]
    replacements=[e for e in events if e['event']=='started' and e['key']==old['key']]
    assert len(replacements)==1 and replacements[0]['attempt_id']!=aid
    saved=json.loads((audit.OUT/'audit.json').read_text())
    saved.update({'quarantined_historical_attempts':1,'full_parent_ledger_preserved':True,
        'approved_replacement_attempts':1,'total_attempts_including_unknown':saved['verified_attempts']+1,
        'maximum_possible_completions':saved['verified_physical_completions']+1,
        'conservative_global_input':saved['conservative_budget_input']+old['reserved_input'],
        'conservative_global_output':saved['conservative_budget_output']+old['reserved_output'],
        'replacement_audit_sha256':file_sha(__file__)})
    limit=m['combined_ceiling']
    assert saved['total_attempts_including_unknown']<=limit['max_attempts']
    assert saved['maximum_possible_completions']<=limit['max_completions']
    assert saved['conservative_global_input']<=limit['max_input_tokens']
    assert saved['conservative_global_output']<=limit['max_output_tokens']
    atomic_json(audit.OUT/'audit.json',saved)
    with (audit.OUT/'FINDINGS.md').open('a') as f:
        f.write('\nOne original Round-0 Azure request timed out at question 55. Its generation and billing status remain unknown in the unchanged parent ledger. The user explicitly authorized one replacement attempt. The two other saved Round-0 responses and all 54 completed questions were reused. The timeout\'s full reservation remains counted against the original combined ceiling; no unresolved attempt was relabeled as a provider rejection.\n')
    print(canonical(saved))


if __name__=='__main__':main()
