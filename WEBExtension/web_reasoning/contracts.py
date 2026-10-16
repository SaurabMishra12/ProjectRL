"""Immutable evidence records and explicit inference allowlists."""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Atom:
    atom_id: str
    document_id: str
    sentence_id: int
    kind: str
    text: str
    title: str
    context: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.kind not in ('text', 'table_row') or type(self.sentence_id) is not int or self.sentence_id < 0:
            raise ValueError('invalid_source_locator')
        if not all(isinstance(x, str) and x for x in (self.atom_id, self.document_id, self.text, self.title)):
            raise ValueError('invalid_source_text')
        object.__setattr__(self, 'context', tuple(tuple(p) for p in self.context))

    def to_public_dict(self):
        return dict(atom_id=self.atom_id, document_id=self.document_id, sentence_id=self.sentence_id,
                    kind=self.kind, text=self.text, title=self.title, context=dict(self.context))


@dataclass(frozen=True)
class Source:
    document_id: str
    title: str
    atoms: tuple[Atom, ...]

    def __post_init__(self):
        object.__setattr__(self, 'atoms', tuple(self.atoms))
        if any(a.document_id != self.document_id or a.title != self.title for a in self.atoms):
            raise ValueError('source_identity_mismatch')

    def to_public_dict(self):
        return dict(document_id=self.document_id, title=self.title,
                    atoms=[a.to_public_dict() for a in self.atoms])


@dataclass(frozen=True)
class PublicExample:
    key: str
    dataset: str
    question: str
    sources: tuple[Source, ...]

    def __post_init__(self):
        object.__setattr__(self, 'sources', tuple(self.sources))
        if self.dataset not in ('2wiki', 'finqa') or not self.key or not self.question.strip():
            raise ValueError('invalid_public_example')
        ids = [a.atom_id for s in self.sources for a in s.atoms]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate_atom_identity')


@dataclass(frozen=True)
class PrivateTargets:
    answer: str
    aliases: tuple[str, ...] = ()
    supporting_locators: tuple[tuple[str, int], ...] = ()
    program: str = ''
    strata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Example:
    public: PublicExample
    targets: PrivateTargets


@dataclass(frozen=True)
class Need:
    entity: str
    relation: str


@dataclass(frozen=True)
class Fact:
    atom_id: str
    quote: str
    claim: str


@dataclass(frozen=True)
class Proposal:
    status: str
    answer: str | None
    facts: tuple[Fact, ...] = ()
    entity_tags: tuple[str, ...] = ()
    needs: tuple[Need, ...] = ()
    derivation: tuple = ()
    conflicts: tuple = ()
    program: tuple[str, ...] = ()
    operand_sources: tuple = ()

    def __post_init__(self):
        for name in ('facts','entity_tags','needs','derivation','conflicts','program','operand_sources'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.status not in ('supported','partial','no_evidence','conflict'):
            raise ValueError('unknown_status')
        if self.status == 'supported':
            if not isinstance(self.answer, str) or not self.answer.strip() or not self.facts or self.answer.strip().lower() == 'noanswer':
                raise ValueError('supported_requires_answer_and_fact')
        elif self.answer is not None:
            raise ValueError('unresolved_answer_must_be_null')


@dataclass(frozen=True)
class Outcome:
    kind: str
    proposal: Proposal | None = None
    raw_sha256: str = ''
    usage: Any = None
    reason: str = ''

    def __post_init__(self):
        if self.kind not in ('valid','refusal','invalid','unresolved'):
            raise ValueError('unknown_outcome')
        if (self.kind == 'valid') != (self.proposal is not None):
            raise ValueError('outcome_proposal_mismatch')


@dataclass(frozen=True)
class AgentState:
    agent_id: int
    visible_atoms: tuple[Atom, ...]
    ledger: tuple[Fact, ...]
    outcome: Outcome


def runtime_object(public):
    if not isinstance(public, PublicExample):
        raise ValueError('public_example_required')
    return {'question': public.question, 'visible_sources': [s.to_public_dict() for s in public.sources]}
