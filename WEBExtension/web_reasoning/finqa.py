"""Bounded typed financial programs and source-visible operand checks."""
from dataclasses import dataclass
import json
import math
import operator
import re

BINARY={'add':operator.add,'subtract':operator.sub,'multiply':operator.mul,'divide':operator.truediv}
TABLE={'table_max':max,'table_min':min,'table_sum':sum,'table_average':lambda x:sum(x)/len(x)}
ALLOWED=set(BINARY)|set(TABLE)|{'exp','greater'}


@dataclass(frozen=True)
class ProgramResult:
    valid:bool
    value:object=None
    reason:str=''
    steps:int=0


def number(token):
    value=token.replace(',','')
    if value.startswith('const_'):value=value[6:];value='-1' if value=='m1' else value
    percent=value.endswith('%');value=value[:-1] if percent else value
    result=float(value)/(100 if percent else 1)
    if not math.isfinite(result) or abs(result)>1e15:raise ValueError('nonfinite_or_oversized_operand')
    return result


def steps(program,max_steps=32):
    if not isinstance(program,(list,tuple)) or not program or program[-1]!='EOF' or (len(program)-1)%4 or not 1< len(program)<=4*max_steps+1:
        raise ValueError('invalid_program_structure')
    for index in range(0,len(program)-1,4):
        op,left,right,end=program[index:index+4]
        if not all(isinstance(t,str) for t in (op,left,right,end)) or not op.endswith('(') or op[:-1] not in ALLOWED:
            raise ValueError('unsupported_operation')
        if end!=')':raise ValueError('invalid_program_structure')
        yield op[:-1],left,right


def visible_table(atoms):
    rows={}
    for atom in atoms:
        if atom.kind!='table_row':continue
        context=dict(atom.context)
        if not {'header','row','report_notes'}<=set(context):raise ValueError('table_header_or_unit_context_missing')
        row=json.loads(context['row']);header=json.loads(context['header'])
        if len(row)!=len(header) or not row:raise ValueError('invalid_table_shape')
        if row[0] in rows and rows[row[0]]!=row[1:]:raise ValueError('ambiguous_table_row')
        rows[row[0]]=row[1:]
    return rows


def execute_program(tokens,visible_table_atoms,max_steps=32):
    results=[]
    try:
        instructions=list(steps(tokens,max_steps));table=visible_table(visible_table_atoms)
        def arg(token):
            if token.startswith('#'):
                if not re.fullmatch(r'#\d+',token) or int(token[1:])>=len(results):raise ValueError('invalid_previous_step_reference')
                value=results[int(token[1:])]
                if isinstance(value,str):raise ValueError('nonnumeric_previous_step')
                return value
            return number(token)
        for op,left,right in instructions:
            if op in TABLE:
                if left not in table or right!='none':raise ValueError('unknown_table_row_or_placeholder')
                values=[number(str(x).replace('$','').split('(')[0].strip()) for x in table[left]]
                if not values:raise ValueError('empty_table_row')
                result=TABLE[op](values)
            else:
                a,b=arg(left),arg(right)
                if op in BINARY:
                    if op=='divide' and b==0:raise ValueError('division_by_zero')
                    result=BINARY[op](a,b)
                elif op=='exp':
                    if abs(b)>20:raise ValueError('exponent_ceiling')
                    result=a**b
                else:result='yes' if a>b else 'no'
            if not isinstance(result,str) and (isinstance(result,complex) or not math.isfinite(result) or abs(result)>1e15):raise ValueError('numerical_ceiling')
            results.append(result)
        final=results[-1]
        return ProgramResult(True,round(final,5) if not isinstance(final,str) else final,'',len(results))
    except (ValueError,TypeError,ZeroDivisionError,OverflowError,KeyError) as error:
        return ProgramResult(False,None,str(error),len(results))


def source_numbers(atom):
    if atom.kind=='table_row':
        context=dict(atom.context)
        if not {'header','row','report_notes'}<=set(context):raise ValueError('table_header_or_unit_context_missing')
        texts=[str(x) for x in json.loads(context['row'])[1:]]
    else:texts=[atom.text]
    result=set()
    for text in texts:
        # Financial parentheses preserve a negative sign for literal operands.
        text=re.sub(r'\(([\d,.]+)\)',r'-\1',text)
        for item in re.findall(r'(?<![\w.])-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)%?',text):
            try:result.add(number(item))
            except ValueError:continue
    return result


def validate_operands(program,operand_sources,visible_atoms):
    visible={a.atom_id:a for a in visible_atoms};mappings={}
    for raw in operand_sources:
        item=json.loads(raw) if isinstance(raw,str) else raw
        if not isinstance(item,dict) or set(item)!={'operand','fact_ids'} or not item['fact_ids'] or any(i not in visible for i in item['fact_ids']):raise ValueError('hidden_operand_source')
        mappings.setdefault(item['operand'],set()).update(item['fact_ids'])
    for op,left,right in steps(program):
        for operand in ((left,) if op in TABLE else (left,right)):
            if operand.startswith(('#','const_')):continue
            if operand not in mappings:raise ValueError('operand_citation_required')
            atoms=[visible[i] for i in mappings[operand]]
            if op in TABLE:
                if operand not in visible_table(atoms):raise ValueError('table_operand_not_visible')
            elif number(operand) not in set().union(*(source_numbers(a) for a in atoms)):
                raise ValueError('operand_not_present_in_cited_source')


def finqa_prediction(proposal,question_key=''):
    return {'id':question_key,'predicted':list(proposal.program)}


def score_official_program(proposal,targets,public,module,visible_atoms=None):
    if public is None:return {'execution_accuracy':None,'program_accuracy':None,'status':'INCOMPLETE','reason':'source_registry_required'}
    if visible_atoms is None:return {'execution_accuracy':None,'program_accuracy':None,'status':'INCOMPLETE','reason':'endpoint_visibility_required'}
    if not targets.program or not targets.answer:return {'execution_accuracy':None,'program_accuracy':None,'status':'INCOMPLETE','reason':'reference_labels_missing'}
    registry={a.atom_id:a for s in public.sources for a in s.atoms};visible=tuple(visible_atoms)
    if any(registry.get(a.atom_id)!=a for a in visible):raise ValueError('endpoint_source_registry_mismatch')
    citation_atoms={f.atom_id for f in proposal.facts}
    try:
        if any(f.atom_id not in {a.atom_id for a in visible} or registry[f.atom_id].text!=f.quote for f in proposal.facts):raise ValueError('invalid_citation')
        validate_operands(proposal.program,proposal.operand_sources,tuple(a for a in visible if a.atom_id in citation_atoms))
    except ValueError as error:
        return {'execution_accuracy':None,'program_accuracy':None,'status':'INCOMPLETE','reason':str(error),'operand_provenance_valid':False}
    predicted=execute_program(proposal.program,visible)
    if not predicted.valid:
        bounded=predicted.reason in ('exponent_ceiling','numerical_ceiling','nonfinite_or_oversized_operand')
        return {'execution_accuracy':None if bounded else 0.,'program_accuracy':None if bounded else 0.,
                'status':'INCOMPLETE' if bounded else 'COMPLETE','program_error':predicted.reason}
    table_atoms=[a for a in visible if a.kind=='table_row'];table=[]
    if table_atoms:
        table=[json.loads(dict(table_atoms[0].context)['header'])]+[json.loads(dict(a.context)['row']) for a in table_atoms]
    invalid,value=module.eval_program(list(proposal.program),table)
    if not invalid and value!=predicted.value:raise ValueError('official_execution_parity_failure')
    try:gold=float(targets.answer)
    except ValueError:gold=targets.answer
    correct=not invalid and value==gold
    gold_tokens=module.program_tokenization(targets.program)
    # Validate the trusted reference structure too before invoking symbolic equivalence.
    valid_gold=execute_program(gold_tokens,tuple(registry.values())).valid
    program_correct=bool(module.equal_program(gold_tokens,list(proposal.program))) if valid_gold else False
    if program_correct and not correct:raise ValueError('inconsistent_official_reference')
    return {'execution_accuracy':100.*correct,'program_accuracy':100.*program_correct,
            'status':'COMPLETE','operand_provenance_valid':True}
