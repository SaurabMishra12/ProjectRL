"""Build notebook walkthroughs from shared, tested experiment code."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
BOOT='''from pathlib import Path
import sys, json
ROOT = Path.cwd() if (Path.cwd() / 'protocol.py').exists() else Path.cwd() / 'WEBExtension'
if not (ROOT / 'protocol.py').exists():
    raise FileNotFoundError('Launch Jupyter from the project root or WEBExtension folder')
ROOT = ROOT.resolve()
sys.path.insert(0, str(ROOT))
from protocol import load_official, write_json, file_hash
from experiments import load_config, preview, freeze, run
CFG = load_config(ROOT / 'config.json')
EXECUTE = False  # Set True only after configuring your real backend and maximum call budget.
print('Notebook root:', ROOT)
print('Inference enabled:', EXECUTE)
'''
def md(s):return nbf.v4.new_markdown_cell(s)
def code(s):return nbf.v4.new_code_cell(s)
def save(name,cells):
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                             'language_info':{'name':'python'}})
    nbf.write(nb,ROOT/name)

save('00_Setup_Data_Protocol.ipynb',[
md('''# Setup, authentic data, and frozen protocol

This suite implements E0–E4 from the supplied blueprint. Research strategy documents are proposals, not results. No experiment uses generated benchmark examples or mock inference. The dataset is publisher-hosted HotpotQA **fullwiki validation candidate contexts**; this is not retrieval over the entire Wikipedia corpus. Exact response logs authenticate new runs; legacy rescoring cannot authenticate historical inference.

Read `README.md`, then configure `config.json`. Install `requirements.txt` in your chosen notebook environment. API credentials belong in environment variables or the notebook platform's secret store. Do not print them. Use an immutable model/deployment revision and a matching tokenizer commit. There is no universally best model; keep the primary backbone fixed and compare another backbone separately.

The default sample is 300 questions, proportional bridge/comparison strata. BM25 ranks the same candidate pool once per question, retaining six documents. No gold answer or support label selects documents, roles, partitions, routing, crops, or stopping. Gold is used only after prediction for scoring and coverage diagnostics.'''),
code(BOOT),
md('''## Download and verify

Data are already included. `download_data.prepare()` can obtain the pinned publisher-hosted parquet when missing. It never invents fallback data. The original CMU download timed out during preparation; this is explicitly recorded. HotpotQA uses CC BY-SA 4.0. Keep the attribution and dataset provenance when distributing results.'''),
code('''from download_data import prepare
if not (ROOT / 'data/hotpot_dev_fullwiki_v1.provenance.json').exists():
    prepare()
rows = load_official()
print('Verified complete validation rows:', len(rows))
print(json.loads((ROOT / 'artifacts/dataset_verification.json').read_text()))'''),
md('''## Freeze before inference

`freeze()` binds config, source hashes, dataset hash and ordered question IDs. A changed protocol creates a different run directory. The primary contrast is `(CW−NoComm)_P − (CW−NoComm)_R` in F1 percentage points. Other contrasts are exploratory. E1 includes a same-call self-update control by default; CW versus NoComm alone combines information delivery with additional inference. Run a separate 10–20 question pilot to check invariants and costs, without selecting methods on favorable answers. Never merge pilot scores into the primary sample.'''),
code('''print(json.dumps(preview(CFG), indent=2))
RUN_DIR, sample = freeze(CFG)
print('Frozen sample:', len(sample), 'questions at', RUN_DIR)'''),
code('''from protocol import retrieve, assign_evidence
import pandas as pd
coverage=[]
for row in sample:
    pool=retrieve(row['question'],row['documents'],CFG['top_k'])
    gold=set(map(tuple,row['supporting_facts']))
    available={(d['title'],i) for d in pool for i in range(len(d['sentences']))}
    assigned=assign_evidence(pool,'P',row['id'],CFG['partition_seed'])
    per_node=[{(d['title'],i) for d in a for i in range(len(d['sentences']))} for a in assigned]
    coverage.append({'id':row['id'],'type':row['type'],'all_support_available':gold<=available,
                     'support_recall':len(gold&available)/len(gold),
                     'support_split':gold<=available and not any(gold<=o for o in per_node)})
coverage=pd.DataFrame(coverage)
coverage.to_csv(RUN_DIR/'retrieval_availability.csv',index=False)
display(coverage.groupby('type').agg(n=('id','size'), all_support_available=('all_support_available','mean'),
                                    mean_support_recall=('support_recall','mean'),split_support=('support_split','mean')))
print('These are dataset/retrieval diagnostics, not QA model results.')'''),
code('''import matplotlib.pyplot as plt
fig, axes=plt.subplots(1,3,figsize=(12,3.5))
for ax,title,labels in zip(axes,['Redundant evidence','Complementary evidence','Controlled recipient allocation'],
                          [['A+B+C']*3,['A','B','C'],['0 → 1','1 → 2','2 → 0']]):
    ax.set_title(title);ax.axis('off')
    for i,label in enumerate(labels):
        ax.text(.5,.8-i*.3,label,ha='center',va='center',bbox=dict(boxstyle='round',facecolor='#e8eff7'))
fig.tight_layout();fig.savefig(ROOT/'artifacts/protocol_diagram.svg');plt.show()'''),
md('''## Pilot rule

For a pilot, copy CFG with `n_questions=10`, `replicate_id="invariant_pilot"`, a confirmed immutable model/tokenizer revision and a sufficient call cap. Run E1, E3 and a short E4. Inspect raw prompts, token usage, invalid citations, exact payload matching and parsing failures before launching 300 questions. If the protocol changes, freeze a new manifest and disclose that change. Temperature zero does not guarantee provider determinism.''')])

save('01_E0_Legacy_Provenance_Cost_Audit.ipynb',[
md('''# E0: legacy provenance and cost audit

This notebook recomputes observations from real saved legacy logs; it does not rerun inference. Corrected W3/W4 are separate from original W3/W4. Empty saved answers remain empty and are scored by official rules, not repaired. Communication counts in these files remain **word proxies**; provider input/output usage is unknown. The current source checksum is not proof of the historical executing source revision.'''),code(BOOT),
code('''from audit import audit_legacy
summary, checks = audit_legacy()
display(summary)
print('Recreated tables:', ROOT/'artifacts/legacy_audit')'''),
code('''display([{'experiment':c['experiment'],'matching_proxy_violations':c.get('matching_proxy_violations'),
          'n_questions':c.get('n_questions'),'limitations':c.get('limitations')} for c in checks])
print('W4 matching excludes Full broadcast but includes the arm labelled Matched.')'''),
md('''## Interpretation

Matching must be checked per question, not only on the mean. The corrected sparse/static/random arms and corrected matched-broadcast arm should also be checked separately. Exact prompt and provider usage reconstruction is unavailable from `raw_results.jsonl`; do not turn these proxy counts into prices or total inference tokens. Audit manifests contain claimed source revisions, artifact paths and checksums, plus unresolved limitations.'''),
code('''import pandas as pd
q = pd.read_csv(ROOT/'artifacts/legacy_audit/question_level_rescore.csv')
w4=q[q.experiment=='W4_corrected'].pivot(index='id',columns='arm',values='word_proxy_comm')
core=['Static_Role','Learned_SparseComm','Random_Routing']
print('Core corrected routing proxy mismatches:',int((w4[core].nunique(axis=1)>1).sum()))
print('Matched broadcast versus static proxy mismatches:',int((w4.Complete_Broadcast_Matched!=w4.Static_Role).sum()))
print('Saved empty answers by experiment:',q.groupby('experiment').empty_answer.sum().to_dict())''')])

specs=[
('02_E1_Evidence_Availability.ipynb','E1','Evidence availability',
'''Compare shared/redundant R and randomly partitioned P evidence using the same candidate union per question. Freeze the three initial responses within each regime. NoComm is the initial normalized majority vote; SelfUpdate gives each agent one extra call with its own answer and no peers. CW and CCW each send every fixed source message once, update every recipient once, and preserve sender payload identity across directions.

Primary: `(CW−NoComm)_P − (CW−NoComm)_R`. CW−SelfUpdate tests the value of peer information at matched calls. Retrieval support availability, split support, newly delivered gold supporting sentences and invalid citations are post-hoc diagnostics. Citation overlap does not establish entailment. Message content consists of unverified peer answer plus verbatim source sentences selected by valid predicted citations. This is a new protocol; it is not numerically interchangeable with legacy W1–W4.

With self-updates enabled this costs up to 24 calls per question (7,200 for 300 questions), before exact cache reuse. A null or harmful result remains a valid finding.'''),
('03_E2_Balanced_Recipient_Routing.ipynb','E2','Balanced recipient allocation',
'''CW: 0→1, 1→2, 2→0. CCW: 0→2, 2→1, 1→0. Each node sends once, receives once and receives exactly one update. Every sender uses the identical preselected token payload in both arms. Semantic roles are randomized before inference and remain fixed within each question.

The contrast is a one-step recipient allocation effect, not learned routing and not multi-hop propagation of revised messages. Role prompts and evidence remain fixed; recipient input necessarily changes with its message source. All input/output token usage is retained even though aggregate payload is exactly matched. E2 uses R and reuses E1 R responses only when prompts and config identities match exactly. Standalone: up to 9 calls per question; receiver updates account for 6 of these.'''),
('04_E3_Exact_Budget_Content.ipynb','E3','Exact-budget content selection',
'''On the fixed edge 0→1, compare order-preserving prefix and seeded contiguous token crop of the same grounded initial source message. The endpoint is the recipient's answer F1; ensemble scores are saved separately in raw records. Both arms use the same question, assigned evidence, own initial answer and condition-neutral prompt. The crop method is not named in model-visible text.

The matched budget is min(configured quota, source message token length). Token slices must decode and re-encode to that exact count without invalid UTF-8. No padding or invented source text is allowed. A slicing failure stops the run. Record identical prefix/crop cases instead of excluding them. Crops can break evidence lines; only complete parseable verbatim sentence lines count as delivered facts. Standalone: 3 initial + 2 receiver calls per question; after E1, only 2 new receiver calls.'''),
('05_E4_Shared_Trajectory_Stopping.ipynb','E4','Shared-prefix stopping replay',
'''Generate the entire condition-neutral complementary-evidence CW trajectory to a fixed depth. Subsequent messages come from newly revised states; this is a dynamic policy trajectory, not E2's destination-only intervention. Replay the first non-abstaining unanimous normalized answer on the saved prefix. No gold label is consulted by the stop decision.

Report stopped minus continued F1 with paired uncertainty, wrong unanimity, continuations that correct wrong stopped answers, and hypothetical calls avoided. Actual generation already paid for the full trajectory: replay savings are not observed dollar savings or latency speedups. To claim quality preservation, set and justify a positive Web-task F1 margin before freezing. An insignificant test is not noninferiority. Full trajectory: 3 initial + 3×stop_depth calls per question.''')]
for name,exp,title,description in specs:
    save(name,[md('# '+exp+': '+title+'\n\n'+description),code(BOOT),code('''print(json.dumps(preview(CFG),indent=2))
RUN_DIR, sample=freeze(CFG)
print('Frozen manifest:',RUN_DIR/'frozen_manifest.json')'''),
    md('''## Execute and resume

The default below performs no inference. Configure your real backend, pinned tokenizer/model revision and maximum calls in `config.json`, then set EXECUTE=True in the setup cell and rerun. Interrupted completed questions are retained; exact cached responses avoid repeated calls. An unresolved in-flight call requires provider-log reconciliation before resume. Never change parameters mid-run to repair favorable results.'''),
    code(f"result = run('{exp}', CFG, execute=EXECUTE)\nprint(result)"),
    code(f"""from reporting import report
status_path=RUN_DIR/'{exp}_status.json'
if status_path.exists() and json.loads(status_path.read_text()).get('status')=='COMPLETE':
    table,stats=report(RUN_DIR,'{exp}')
    display(table)
    print(json.dumps(stats,indent=2))
else:
    print('{exp}: NOT COMPLETED. No model result table is available.')""")])

save('06_Statistics_Tables_Figures.ipynb',[
md('''# Paired statistics, grounding diagnostics, and figures

Only manifest-validated complete runs are reported. F1/EM use the official HotpotQA answer normalization in percentage points; equivalence is tested against the downloaded upstream scorer. Confidence intervals resample questions, preserving treatment pairing. A sign randomization test supplements the interval. E1 has one prespecified primary interaction; other tests are exploratory with within-experiment Holm correction. Across-experiment comparisons remain exploratory.

Required outputs: condition/evidence regime/graph F1 and EM, supporting-fact recall, actual payload tokens, full provider input/output usage, logical calls; paired-effect figures and separate payload/inference frontiers. Unknown provider usage remains missing. Retain all helped, harmed and unchanged cases. Gold-supported delivery is a diagnostic, not proof of the mechanism or source entailment.'''),code(BOOT),
code('''from reporting import report, physical_usage
from IPython.display import display, Image
RUN_DIR, _=freeze(CFG)
for experiment in ['E1','E2','E3','E4']:
    status=RUN_DIR/(experiment+'_status.json')
    if not status.exists() or json.loads(status.read_text()).get('status')!='COMPLETE':
        print(experiment+': NOT COMPLETED; no results generated')
        continue
    table,stats=report(RUN_DIR,experiment)
    display(table);display(pd.DataFrame(stats['effects'])) if 'pd' in globals() else print(stats['effects'])
    display(Image(filename=str(RUN_DIR/'reports'/experiment/'paired_effects.png')))
print('Physical usage:',physical_usage(RUN_DIR))'''),
md('''## Cost interpretation and reporting boundaries

Logical costs include each condition's shared initial calls. Physical billing sums successful real requests once from the call ledger; cache reads are not new completions. Error calls can be billable with unknown usage. Payload counts exclude JSON prompt framing; provider input/output counts include actual inference usage when reported. Do not infer monetary costs without the deployment's actual billing rates and usage records. Measured latency belongs to actual calls, not cached responses. Exported JSON includes exact raw record references for evidence-trace examples.'''),
code('''# Verify raw response files and freeze a shareable checksum index.
checks={str(p.relative_to(ROOT)):file_hash(p) for p in RUN_DIR.rglob('*') if p.is_file() and p.name!='export_checksums.json'}
write_json(RUN_DIR/'export_checksums.json',checks)
print('Checksum entries:',len(checks))''')])

save('07_Seed_Backbone_Robustness.ipynb',[
md('''# Prespecified partition seeds and independent backbone replication

Run at least three prespecified partition/role seeds on exactly the same question sample. This tests sensitivity to assigned evidence and role placement. It does not create independent questions or prove robustness to model sampling. Never choose the best seed. Aggregate interactions across seeds within question, then bootstrap questions. Report each seed separately as well.

For a second real backbone, copy the config and change model, immutable revision, tokenizer and tokenizer revision; keep the sample and evaluation protocol fixed. Run the same notebooks under that config. Keep per-backbone results separate. An independent backbone is the blueprint's affordable generalization option; this suite does not claim a second benchmark has been evaluated.'''),code(BOOT),
code('''from robustness import configs, replicate, analyze_seed_runs
sweep=configs(CFG)
for config in sweep:
    print(config['replicate_id'], preview(config))
print('Total E1 upper-bound calls across seeds:',sum(preview(c)['E1_max_new_calls'] for c in sweep))
print('Each config max_calls is a per-run cap. Set the total budget deliberately before execute.')'''),
code('''seed_runs=replicate(CFG,execute=EXECUTE)
print(seed_runs)
if EXECUTE:
    print(json.dumps(analyze_seed_runs(seed_runs),indent=2))
else:
    print('No seed replication inference was executed.')''')])

save('08_Run_All_And_Readiness.ipynb',[
md('''# Run sequence and evidence-based readiness

Execute setup → E0 → invariant pilot → E1 → E2 → E3 → E4 → reporting → seed/backbone robustness. E4 and replication are optional by scientific priority but implemented. Configure adequate costs before calling all phases. E2 exact prompts reuse E1 R, so they should create no additional physical calls. Separate experiment checkpoints remain available for interpretation.

A full 300-question default suite with E1 self-updates, E3 and depth-3 E4 has a conservative upper bound of 11,400 calls before cache reuse (E1 7,200 + E3 600 + E4 3,600; E2 reuses E1). Some E4 initial/first-update prompts also match E1. These are completions, not prices. Seeds multiply cost. Do not call a run complete until all intended questions, strict invariants, raw responses and scorer outputs validate.'''),code(BOOT),
code('''from audit import audit_legacy
from reporting import report, physical_usage
print('No-inference preflight:',preview(CFG))
if EXECUTE:
    for experiment in ['E1','E2','E3','E4']:
        run(experiment,CFG,execute=True)
        directory,_=freeze(CFG)
        table,stats=report(directory,experiment)
        display(table)
else:
    print('Full new-inference suite NOT RUN. Configure backend and budget, then enable execution.')'''),
code('''directory,_=freeze(CFG)
readiness={experiment:(json.loads((directory/(experiment+'_status.json')).read_text()).get('status')
                     if (directory/(experiment+'_status.json')).exists() else 'NOT_RUN')
           for experiment in ['E1','E2','E3','E4']}
print('Experiment status:',readiness)
print('Physical call/usage status:',physical_usage(directory))
print('Only actual complete results support paper claims. Null/negative outcomes must be retained.')''')])
print('Built',len(list(ROOT.glob('*.ipynb'))),'notebooks')
