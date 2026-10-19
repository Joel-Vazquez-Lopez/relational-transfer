"""BX-005: adversarial fix for BX-004's question-changing false positive.

Offline developmental protocol validator, no LLM inference. Source fragments and their
hashes still require human/source verification; self-declared checksums do not prove rights.
"""
import hashlib
import json
from bx004_evidence_utility import CONDITIONS, validate, evaluate

V1_FIELDS={'event_id','condition','prompt','prompt_sha256','source_text_sha256'}
REQUIRED=V1_FIELDS|{'question','instruction','base_spans','additional_spans'}

def digest(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def canonical_prompt(instruction, base, additional, question):
    """One immutable query/output interface; only approved context may vary."""
    if not all(isinstance(t,str) and t.strip() for t in (instruction,base,question)):
        raise ValueError('instruction, base context, question must be nonempty')
    if not isinstance(additional,str):
        raise ValueError('additional context must be a string')
    context=base if not additional else base+'\n'+additional
    return instruction+'\n\nPASSAGE:\n'+context+'\n\nQUESTION:\n'+question+'\nANSWER:'

def assemble(spans, authorized_ids, commitments, tag):
    if not isinstance(spans,list) or (not spans and tag=='base'):
        raise ValueError(tag+': missing required spans')
    if not isinstance(authorized_ids,list) or not isinstance(commitments,dict):
        raise ValueError(tag+': missing source manifest')
    texts=[]; ids=[]
    for span in spans:
        if not isinstance(span,dict) or set(span)!={'span_id','text','sha256'}:
            raise ValueError(tag+': invalid or extra span fields')
        sid=span['span_id']; txt=span['text']
        if not isinstance(sid,str) or not isinstance(txt,str) or not txt.strip():
            raise ValueError(tag+': invalid span text/id')
        if span['sha256']!=digest(txt) or commitments.get(sid)!=digest(txt):
            raise ValueError(tag+': span sha mismatch: '+sid)
        texts.append(txt);ids.append(sid)
    if len(set(ids))!=len(ids) or ids!=authorized_ids:
        raise ValueError(tag+': exact ordered source span ids do not match the manifest')
    return '\n'.join(texts)

def validate_v2(manifest, gold, rich_inputs, responses, min_docs=10, families=2):
    """Run v1 scientific eligibility gate, then verify exact prompt construction.

    Crucial: This does not itself verify that hashes correspond to upstream text,
    that human reviewer IDs represent real independent humans, or that rights were
    cleared. Those require separate third-party verification and manual authorization.
    """
    errors=[]; meta={r['event_id']:r for r in manifest}
    bare=[]; grouped={}
    for row in rich_inputs:
        k=(row.get('event_id'),row.get('condition'))
        if k in grouped: errors.append(f'{k}:DUPLICATE_MODEL_INPUT')
        grouped[k]=row
        if set(row)!=REQUIRED:
            errors.append(f'{k}:MISSING_OR_EXTRA_RICH_INPUT_FIELDS:{sorted(REQUIRED-set(row))}:{sorted(set(row)-REQUIRED)}')
        bare.append({f:row[f] for f in V1_FIELDS if f in row})
    try: base=validate(manifest,gold,bare,responses,min_docs=min_docs,families=families)
    except (ValueError,KeyError,TypeError) as e:
        return {'status':'BLOCKED','errors':['V1_VALIDATION_ERROR:'+str(e)],'baseline':None}
    errors+=base['errors']
    for eid,m in meta.items():
        triple=[grouped.get((eid,c)) for c in CONDITIONS]
        if any(x is None or not REQUIRED.issubset(x) for x in triple):
            errors.append(f'{eid}:INCOMPLETE_CONTEXT_TRIPLET');continue
        questions={r['question'] for r in triple}
        instructions={r['instruction'] for r in triple}
        if len(questions)!=1:errors.append(f'{eid}:QUESTION_CHANGED_ACROSS_CONDITIONS')
        if len(instructions)!=1:errors.append(f'{eid}:OUTPUT_INTERFACE_CHANGED_ACROSS_CONDITIONS')
        try:
            b=[]
            for c,r in zip(CONDITIONS,triple):
                commitments=m.get('source_span_sha256')
                if not isinstance(commitments,dict):raise ValueError('source span commitments missing')
                base_txt=assemble(r['base_spans'],m['local_span_ids'],commitments,'base')
                expected=m['informative_addition_span_ids'] if c=='informative' else m['irrelevant_addition_span_ids'] if c=='irrelevant' else []
                extra_txt=assemble(r['additional_spans'],expected,commitments,'addition')
                if c=='local' and r['additional_spans']:
                    raise ValueError('local arm contains additional context')
                prompt=canonical_prompt(r['instruction'],base_txt,extra_txt,r['question'])
                if r['prompt']!=prompt:raise ValueError('prompt does not match canonical serialization')
                context=base_txt+('\n'+extra_txt if extra_txt else '')
                if r['source_text_sha256']!=digest(context):
                    raise ValueError('context SHA mismatch')
                if r['prompt_sha256']!=digest(prompt):
                    raise ValueError('prompt SHA mismatch')
                b.append((base_txt,tuple(r['base_spans'])))
            if len({x[0] for x in b})!=1:
                errors.append(f'{eid}:BASE_CONTEXT_CHANGED_ACROSS_CONDITIONS')
        except (KeyError,ValueError,TypeError) as e:
            errors.append(f'{eid}:CONTEXT_CONTRACT_FAILURE:{e}')
    base['errors']=sorted(set(errors))
    return {'status':'BLOCKED' if base['errors'] else 'CONTRACT_PASS_PENDING_MANUAL_SOURCE_REVIEW',
            'errors':base['errors'],'baseline':base}

def score_v2(manifest,gold,inputs,responses,min_docs=10,families=2):
    outcome=validate_v2(manifest,gold,inputs,responses,min_docs,families)
    if outcome['errors']:return {'status':'BLOCKED','errors':outcome['errors']}
    report=evaluate(outcome['baseline'])
    report['status']='SYNTHETIC_SMOKE_ONLY_UNLESS_MANUALLY_VERIFIED'
    report['mandatory_manual_gates']=['source text versus pinned original','per-document rights and sensitive-data approval','two actual human annotators plus independent adjudicator','context independent informativity and genuine irrelevance','model runner sends ONLY prompt string']
    return report

def model_visible_request(row):
    """The only approved handoff payload: never IDs, gold, spans or hashes."""
    if set(row)!=REQUIRED:raise ValueError('rich model input schema invalid')
    return {'prompt':row['prompt']}
