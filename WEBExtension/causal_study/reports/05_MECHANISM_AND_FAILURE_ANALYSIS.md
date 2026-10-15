# Mechanism and failure analysis

E6 completed-block diagnostics: 8 unique questions, 32 question-cells. Across cells: 96 directed broadcast messages, 76 delivered source-fact instances, 52 newly accessible annotated-support fact instances, 1 EM-exact-to-nonexact agent transitions, 12 nonexact-to-exact transitions, 5 cells rejecting an exact candidate, 16 EM-nonexact revised agent outputs despite all annotated support being accessible, and 0 invalid citation references. EM misses are metric-defined, not a claim that every response is substantively wrong. Agent/cell sums repeat questions; they are descriptive events, not independent sample sizes.

Source packets are the frozen initial answer plus complete sentences matching valid sender-visible citations. Messages can carry an answer proposal with no source fact. The source-count manipulation and factual uptake must be reported separately. A citation to a delivered fact does not establish entailment or internal use; unsupported/hallucinated reasoning and contradictions are **UNOBSERVABLE_FROM_CONCISE_ANSWER_CITATION_OUTPUT** except for directly invalid citation indices. No paid judge was invoked. Source delivery is an accessibility measurement, not successful knowledge transfer.

## Proposal availability and what the intervention did

Across16 placement/question blocks, 16 broadcast final answers already occurred among the frozen initial proposals and 16 equal the best initial agent F1. Broadcast generated a new exact proposal on0 blocks lacking any initial exact proposal. Best-agent F1 uses gold and is an oracle diagnostic only; it was never used to select answers or sample questions. This pattern is consistent with propagating/selecting existing answers, rather than evidence that agents needed to combine complementary missing facts. The delivered source sentences and answer proposals remain bundled, so their causal contributions are not separated.

The source-only independent review in report06 finds that many single annotated support articles already contain the answer fact or share the answer attribute; annotation membership does not establish complementary necessity. Several distractors also mention the answer. The pilot was selected by frozen hashes and not by ease, but the eligible population still does not enforce the intended necessity intervention. A next source review must be blind to outputs and must not quietly replace this cohort based on outcomes. Treat this as a scientific design finding, not BM25 failure.

## Surface forms, abstentions, and the two nonzero interactions

Six of eight question interactions are exactly zero. The negative interaction is the Quinlan question: co-located private majority chooses noanswer while distributed private revision has a three-way tie selecting `Ernesto "Che" Guevara` (80F1); both broadcasts select `Che Guevara` (100F1). The positive interaction is the RosiGolan/Chicco question: broadcasts select `more than120` versus `more than120 countries`, scoring85.714 versus100 despite the same numerical answer. These observed response/tie/scoring paths make the small pilot direction fragile; they do not establish a causal explanation of population behavior.

For DannyJones/AleksiSihvonen, broadcast propagates `vocalists` to every agent, while the reference is `vocalist`; official EM and answerF1 are0 because the evaluator does not stem singular/plural. This is an observed surface-form limitation, not proof of failed factual reasoning. Preserve the prespecified official metric; no alternative post-hoc semantic score, revised gold answer, or paid regeneration is substituted. `Bi-fuel` and `Swiss Super League` also receive partial F1 against longer reference strings.

## Question-level trace register

### 5abace6b5542996cc5e49e99

Irish print journalist, Arthur Quinlan, was the only one to interviewed which major figure of the Cuban Revolution?
Reference answer: `Che Guevara`. Support articles: ['Che Guevara', 'Arthur Quinlan'].

- colocated: initial answers ['Che Guevara', 'noanswer', 'noanswer']; initial majority `noanswer`.
- colocated/nocomm: revised answers ['Che Guevara', 'noanswer', 'noanswer']; selected `noanswer`, F1=0.000, EM=0; new support by node=[0, 0, 0]; majority rejected exact=True.
- colocated/comm: revised answers ['Che Guevara', 'Che Guevara', 'Che Guevara']; selected `Che Guevara`, F1=100.000, EM=100; new support by node=[0, 1, 1]; majority rejected exact=False.
- distributed: initial answers ['Ernesto "Che" Guevara', 'Che Guevara', 'noanswer']; initial majority `Ernesto "Che" Guevara`.
- distributed/nocomm: revised answers ['Ernesto "Che" Guevara', 'Che Guevara', 'noanswer']; selected `Ernesto "Che" Guevara`, F1=80.000, EM=0; new support by node=[0, 0, 0]; majority rejected exact=True.
- distributed/comm: revised answers ['noanswer', 'Che Guevara', 'Che Guevara']; selected `Che Guevara`, F1=100.000, EM=100; new support by node=[0, 1, 1]; majority rejected exact=False.

### 5a8080365542992097ad2fed

A vehicle that uses hydrogen as its onboard fuel for motive power is cosidered which, Flex-fuel or Bi-fuel?
Reference answer: `bi-fuel vehicles`. Support articles: ['Flexible-fuel vehicle', 'Hydrogen vehicle'].

- colocated: initial answers ['Bi-fuel', 'Bi-fuel', 'noanswer']; initial majority `Bi-fuel`.
- colocated/nocomm: revised answers ['noanswer', 'Bi-fuel', 'noanswer']; selected `noanswer`, F1=0.000, EM=0; new support by node=[0, 0, 0]; majority rejected exact=False.
- colocated/comm: revised answers ['Bi-fuel', 'Bi-fuel', 'Bi-fuel']; selected `Bi-fuel`, F1=66.667, EM=0; new support by node=[2, 0, 2]; majority rejected exact=False.
- distributed: initial answers ['neither', 'Bi-fuel', 'noanswer']; initial majority `neither`.
- distributed/nocomm: revised answers ['neither', 'Bi-fuel', 'neither']; selected `neither`, F1=0.000, EM=0; new support by node=[0, 0, 0]; majority rejected exact=False.
- distributed/comm: revised answers ['Bi-fuel', 'Bi-fuel', 'Bi-fuel']; selected `Bi-fuel`, F1=66.667, EM=0; new support by node=[1, 1, 2]; majority rejected exact=False.

### 5abfca465542993fe9a41e65

which league is the 121th season of top-tier competitive football in Switzerland which Ulrich Forte coach FC Zürich in the league 
Reference answer: `2017–18 Swiss Super League`. Support articles: ['2017–18 Swiss Super League', 'Ulrich Forte'].

- colocated: initial answers ['noanswer', 'Swiss Super League', 'Swiss Super League']; initial majority `Swiss Super League`.
- colocated/nocomm: revised answers ['noanswer', 'Swiss Super League', 'Swiss Super League']; selected `Swiss Super League`, F1=85.714, EM=0; new support by node=[0, 0, 0]; majority rejected exact=False.
- colocated/comm: revised answers ['Swiss Super League', 'Swiss Super League', 'Swiss Super League']; selected `Swiss Super League`, F1=85.714, EM=0; new support by node=[1, 1, 0]; majority rejected exact=False.
- distributed: initial answers ['Swiss Super League', 'Swiss Super League', 'Swiss Super League']; initial majority `Swiss Super League`.
- distributed/nocomm: revised answers ['Swiss Super League', 'Swiss Super League', 'Swiss Super League']; selected `Swiss Super League`, F1=85.714, EM=0; new support by node=[0, 0, 0]; majority rejected exact=False.
- distributed/comm: revised answers ['Swiss Super League', 'Swiss Super League', 'Swiss Super League']; selected `Swiss Super League`, F1=85.714, EM=0; new support by node=[1, 2, 1]; majority rejected exact=False.

### 5ac3044f5542990b17b154dc

 Danny Jones and Aleksi Sihvonen are both what?
Reference answer: `vocalist`. Support articles: ['Aleksi Sihvonen', 'Danny Jones'].

- colocated: initial answers ['vocalists', 'noanswer', 'noanswer']; initial majority `noanswer`.
- colocated/nocomm: revised answers ['vocalists', 'noanswer', 'noanswer']; selected `noanswer`, F1=0.000, EM=0; new support by node=[0, 0, 0]; majority rejected exact=False.
- colocated/comm: revised answers ['vocalists', 'vocalists', 'vocalists']; selected `vocalists`, F1=0.000, EM=0; new support by node=[0, 2, 2]; majority rejected exact=False.
- distributed: initial answers ['vocalists', 'noanswer', 'noanswer']; initial majority `noanswer`.
- distributed/nocomm: revised answers ['vocalists', 'noanswer', 'noanswer']; selected `noanswer`, F1=0.000, EM=0; new support by node=[0, 0, 0]; majority rejected exact=False.
- distributed/comm: revised answers ['vocalists', 'vocalists', 'vocalists']; selected `vocalists`, F1=0.000, EM=0; new support by node=[0, 1, 1]; majority rejected exact=False.

### 5a7c651a55429935c91b5188

The mulga apple is is often eaten by people who genetic research has inferred a date of habitation as early as when?
Reference answer: `80,000 years BP`. Support articles: ['Mulga apple', 'Indigenous Australians'].

- colocated: initial answers ['noanswer', '80,000 years BP', 'noanswer']; initial majority `noanswer`.
- colocated/nocomm: revised answers ['noanswer', '80,000 years BP', 'noanswer']; selected `noanswer`, F1=0.000, EM=0; new support by node=[0, 0, 0]; majority rejected exact=True.
- colocated/comm: revised answers ['80,000 years BP', '80,000 years BP', '80,000 years BP']; selected `80,000 years BP`, F1=100.000, EM=100; new support by node=[2, 0, 2]; majority rejected exact=False.
- distributed: initial answers ['noanswer', 'noanswer', '80,000 years BP']; initial majority `noanswer`.
- distributed/nocomm: revised answers ['noanswer', 'noanswer', '80,000 years BP']; selected `noanswer`, F1=0.000, EM=0; new support by node=[0, 0, 0]; majority rejected exact=True.
- distributed/comm: revised answers ['80,000 years BP', '80,000 years BP', '80,000 years BP']; selected `80,000 years BP`, F1=100.000, EM=100; new support by node=[1, 1, 0]; majority rejected exact=False.

### 5a735e9155429901807db00d

What country is home to both Guadarrama National Park and Doñana National Park? 
Reference answer: `Spain`. Support articles: ['Guadarrama National Park', 'Doñana National Park'].

- colocated: initial answers ['Spain', 'noanswer', 'Spain']; initial majority `Spain`.
- colocated/nocomm: revised answers ['Spain', 'noanswer', 'Spain']; selected `Spain`, F1=100.000, EM=100; new support by node=[0, 0, 0]; majority rejected exact=False.
- colocated/comm: revised answers ['Spain', 'Spain', 'Spain']; selected `Spain`, F1=100.000, EM=100; new support by node=[2, 2, 0]; majority rejected exact=False.
- distributed: initial answers ['noanswer', 'Spain', 'Spain']; initial majority `Spain`.
- distributed/nocomm: revised answers ['Spain', 'noanswer', 'Spain']; selected `Spain`, F1=100.000, EM=100; new support by node=[0, 0, 0]; majority rejected exact=False.
- distributed/comm: revised answers ['Spain', 'Spain', 'Spain']; selected `Spain`, F1=100.000, EM=100; new support by node=[2, 1, 1]; majority rejected exact=False.

### 5ac540365542994611c8b470

Loughborough Students' Union serves the students at what public research university that has been a university since 1966, but the institution dates back to 1909?
Reference answer: `Loughborough University`. Support articles: ['Loughborough University', "Loughborough Students' Union"].

- colocated: initial answers ['Loughborough University', 'Loughborough University', 'Loughborough University']; initial majority `Loughborough University`.
- colocated/nocomm: revised answers ['Loughborough University', 'Loughborough University', 'Loughborough University']; selected `Loughborough University`, F1=100.000, EM=100; new support by node=[0, 0, 0]; majority rejected exact=False.
- colocated/comm: revised answers ['Loughborough University', 'Loughborough University', 'Loughborough University']; selected `Loughborough University`, F1=100.000, EM=100; new support by node=[0, 2, 2]; majority rejected exact=False.
- distributed: initial answers ['Loughborough University', 'Loughborough University', 'Loughborough University']; initial majority `Loughborough University`.
- distributed/nocomm: revised answers ['Loughborough University', 'Loughborough University', 'Loughborough University']; selected `Loughborough University`, F1=100.000, EM=100; new support by node=[0, 0, 0]; majority rejected exact=False.
- distributed/comm: revised answers ['Loughborough University', 'Loughborough University', 'Loughborough University']; selected `Loughborough University`, F1=100.000, EM=100; new support by node=[1, 2, 3]; majority rejected exact=False.

### 5abe4a365542993f32c2a0dc

Rosi Golan's songs were featured in commercials for a brand of Artsana, that's been working for over 50 years, and is present in how many countries?
Reference answer: `more than 120 countries`. Support articles: ['Chicco', 'Rosi Golan'].

- colocated: initial answers ['noanswer', 'more than 120', 'noanswer']; initial majority `noanswer`.
- colocated/nocomm: revised answers ['noanswer', 'more than 120', 'noanswer']; selected `noanswer`, F1=0.000, EM=0; new support by node=[0, 0, 0]; majority rejected exact=False.
- colocated/comm: revised answers ['more than 120', 'more than 120', 'more than 120']; selected `more than 120`, F1=85.714, EM=0; new support by node=[1, 0, 1]; majority rejected exact=False.
- distributed: initial answers ['noanswer', 'more than 120 countries', 'noanswer']; initial majority `noanswer`.
- distributed/nocomm: revised answers ['noanswer', 'more than 120 countries', 'noanswer']; selected `noanswer`, F1=0.000, EM=0; new support by node=[0, 0, 0]; majority rejected exact=True.
- distributed/comm: revised answers ['more than 120 countries', 'more than 120 countries', 'more than 120 countries']; selected `more than 120 countries`, F1=100.000, EM=100; new support by node=[1, 0, 1]; majority rejected exact=False.

## E8 old-study comparison

E8 is wholly offline and observational, not an E6 intervention estimate. In saved E5 P, initial any-agent exact67/300 contrasts with majority10/300; broadcast any-agent exact66/300 contrasts with majority34/300. Of25 questions whose initially incorrect vote becomes exact,23 already had an exact initial proposal. This supports a proposal-utilization bottleneck as an observed failure mode; it does not identify whether content, repeated computation, graph reach or agreement formation caused the old gains. Learned seed19/31 reject48/49 exact revised candidates versus32 for broadcast. No oracle selection is a practical model result. Full43-group transitions, EM/F1 separation and intervals are in `../offline/`.

Earlier E1 partitioned gains and its imprecise interaction do not predict the same-union intervention. E6 conditions on support availability; older300 include224 cases with at least one annotated article missing, mostly absent before ranking. No estimates are pooled across these protocols. Corrected E3-v2 tests redundant shared-evidence presentation and does not establish novel transfer. E4 is replay stopping. Learned routing is unnecessary for the current fixed-communication question.

E7 paid outcomes are **NOT_RUN**. Six provisional exact-token-matched source triples were generated locally; no semantic non-support or treatment effect is asserted. A future E7 needs blind content verification, whole-passage treatment validation, and adequate frozen receiver states; no outcome-based receiver selection.
