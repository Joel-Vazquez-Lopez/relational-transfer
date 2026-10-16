#!/usr/bin/env python3
"""BX-004: gold-isolated, document-clustered evidence-utility audit.

Does NO inference. Requires preadjudicated human semantic labels and source-clearance;
synthetic smoke fixtures are not valid empirical evidence. Python 3.9+ stdlib.
"""
import argparse, collections, csv, hashlib, json, random, re, statistics
from pathlib import Path

CONDITIONS = ('local','informative','irrelevant')
SECRET_FIELDS = {'gold','agent','patient','actor','correct_answer','source_graph','generator_edges','split','document_id','doc_id','event_id','item_id','role_labels','annotation','annotator_id'}

def load_jsonl(path):
    result=[]
    for n,line in enumerate(Path(path).read_text(encoding='utf8').splitlines(),1):
        if line.strip():
            try: obj=json.loads(line)
            except Exception as e: raise ValueError(f'{path}:{n} invalid JSON: {e}') from e
            if not isinstance(obj,dict): raise ValueError(f'{path}:{n} object expected')
            result.append(obj)
    return result

def unique(seq,key,label):
    d={}
    for row in seq:
        k=key(row)
        if k in d: raise ValueError(f'{label}: duplicate {k}')
        d[k]=row
    return d

def validate(manifest,gold,model_inputs,predictions, min_docs=10, families=2):
    errors=[]
    M=unique(manifest,lambda r:r['event_id'],'manifest')
    G=unique(gold,lambda r:r['event_id'],'gold')
    I=unique(model_inputs,lambda r:(r['event_id'],r['condition']),'model inputs')
    P=unique(predictions,lambda r:(r['model_family'],r['model_revision'],r['event_id'],r['condition']),'predictions')
    if not M: errors.append('NO_EVENTS')
    if set(M)!=set(G): errors.append('GOLD_MANIFEST_ID_MISMATCH')
    docs={r['document_id'] for r in manifest}
    if len(docs)<min_docs: errors.append(f'INSUFFICIENT_DOCUMENT_CLUSTERS:{len(docs)}<{min_docs}')
    models={(r['model_family'],r['model_revision']) for r in predictions}
    if len({m[0] for m in models})<families: errors.append('INSUFFICIENT_DISTINCT_MODEL_FAMILIES')
    for event_id, row in M.items():
        for f in ('document_id','source_commit','source_rights','contrast_adjudication','gold_adjudication','local_evidence_status'):
            if not row.get(f): errors.append(f'{event_id}:MISSING_{f.upper()}')
        if row.get('source_rights') != 'approved_for_internal_model_evaluation': errors.append(f'{event_id}:RIGHTS_NOT_CLEARED')
        if row.get('context_provenance') != 'unaltered_original_document_subspans':
            errors.append(f'{event_id}:CONTEXT_PROVENANCE_UNVERIFIED')
        spans=[row.get(name) for name in ('local_span_ids','informative_addition_span_ids','irrelevant_addition_span_ids')]
        if any(not isinstance(x,list) or not x or len(x)!=len(set(x)) for x in spans):
            errors.append(f'{event_id}:MISSING_OR_DUPLICATE_SOURCE_SPANS')
        elif (set(spans[0]) & set(spans[1])) or (set(spans[0]) & set(spans[2])) or (set(spans[1]) & set(spans[2])):
            errors.append(f'{event_id}:SOURCE_SPAN_CONDITION_OVERLAP')
        if row.get('contrast_adjudication') != 'human_verified_independently_informative_vs_irrelevant': errors.append(f'{event_id}:CONTEXT_NOT_HUMAN_VERIFIED')
        if row.get('gold_adjudication') != 'completed_two_blind_humans_plus_independent_adjudicator': errors.append(f'{event_id}:GOLD_NOT_HUMAN_ADJUDICATED')
        if row.get('question_role') not in ('actor','patient'): errors.append(f'{event_id}:QUESTION_ROLE_UNSET')
        if row.get('local_evidence_status') not in ('sufficient','insufficient','uncertain'):
            errors.append(f'{event_id}:LOCAL_EVIDENCE_UNCLASSIFIED')
        if event_id not in G: continue
        g=G[event_id]
        if not g.get('actor') or not g.get('patient') or g.get('actor')==g.get('patient'):
            errors.append(f'{event_id}:GOLD_AMBIGUOUS_OR_NOT_BINARY')
        raters=g.get('annotators',[])
        if len(raters)!=2 or len(set(raters))!=2 or not g.get('adjudicator') or g.get('adjudicator') in raters:
            errors.append(f'{event_id}:NO_INDEPENDENT_HUMAN_REVIEW')
        if g.get('document_id')!=row.get('document_id'): errors.append(f'{event_id}:GOLD_DOC_MISMATCH')
        for condition in CONDITIONS:
            k=event_id,condition
            if k not in I:
                errors.append(f'{event_id}:MISSING_INPUT_{condition}')
                continue
            inp=I[k]
            unknown=set(inp)-{'event_id','condition','prompt','source_text_sha256','prompt_sha256'}
            if unknown: errors.append(f'{event_id}:{condition}:MODEL_INPUT_EXTRA_FIELDS:{sorted(unknown)}')
            if not isinstance(inp.get('prompt'),str) or not inp['prompt'].strip(): errors.append(f'{event_id}:{condition}:EMPTY_PROMPT')
            if inp.get('prompt_sha256')!=hashlib.sha256(inp.get('prompt','').encode()).hexdigest(): errors.append(f'{event_id}:{condition}:PROMPT_HASH_MISMATCH')
        if all((event_id,c) in I for c in CONDITIONS):
            inputs=[I[event_id,c]['prompt'] for c in CONDITIONS]
            if len(set(inputs))!=3: errors.append(f'{event_id}:NON_DISTINCT_CONTEXT_CONDITIONS')
    for model in models:
        for eid in M:
            for condition in CONDITIONS:
                if (model[0],model[1],eid,condition) not in P:
                    errors.append(f'{model}:{eid}:MISSING_RESPONSE_{condition}')
    for k,p in P.items():
        if p.get('input_sha256')!=I.get((p['event_id'],p['condition']),{}).get('prompt_sha256'):
            errors.append(f'{k}:RESPONSE_PROMPT_HASH_MISMATCH')
        if not isinstance(p.get('response'),str): errors.append(f'{k}:RESPONSE_NOT_STRING')
    return {'errors':sorted(set(errors)),'events':len(M),'documents':len(docs),'model_families':sorted(set(m[0] for m in models)),
            'models':sorted([list(x) for x in models]),'manifest':M,'gold':G,'inputs':I,'predictions':P}

def score_response(response,gold,question_role):
    """Parser-only validity is independent of gold participant list."""
    answer=response.strip()
    valid=bool(re.fullmatch(r"[^\W\d_][\w '\-]{0,79}",answer,flags=re.UNICODE))
    if not valid:return False,False
    return answer.casefold()==gold[question_role].strip().casefold(),True

def _boot_delta(units,seed=161604,repetitions=2000):
    if not units:return None
    rng=random.Random(seed)
    diffs=[v[0]-v[1] for v in units.values()]
    means=[]
    for _ in range(repetitions):
        samples=[diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(statistics.mean(samples))
    means.sort()
    return {'documents':len(diffs),'ci95': [means[int(.025*repetitions)],means[int(.975*repetitions)]],
            'method':'document_cluster_percentile_bootstrap_descriptive_unadjusted'}

def evaluate(v,seed=161604):
    if v['errors']:return {'status':'BLOCKED','reason':'failed_intake_validation','errors':v['errors'],
                           'events':v['events'],'documents':v['documents'],'families':v['model_families']}
    out=[]
    for family,revision in map(tuple,v['models']):
        rows=[]
        for eid,meta in v['manifest'].items():
            g=v['gold'][eid]
            answers={};valid={}
            for c in CONDITIONS:
                p=v['predictions'][family,revision,eid,c]
                answers[c],valid[c]=score_response(p['response'],g,meta['question_role'])
            rows.append(dict(doc=meta['document_id'],event=eid,local_status=meta['local_evidence_status'],correct=answers,valid=valid))
        bydoc=collections.defaultdict(list)
        for row in rows:bydoc[row['doc']].append(row)
        # Treat documents rather than dependent event prompts as sampling units.
        doc_scores={doc:(statistics.mean(int(r['correct']['informative']) for r in rs),statistics.mean(int(r['correct']['irrelevant']) for r in rs)) for doc,rs in bydoc.items()}
        summary={'model_family':family,'model_revision':revision,'n_documents':len(bydoc),'n_events':len(rows),
                 'accuracy':{c:sum(int(r['correct'][c]) for r in rows)/len(rows) for c in CONDITIONS},
                 'format_valid':{c:sum(int(r['valid'][c]) for r in rows)/len(rows) for c in CONDITIONS},
                 'net_informative_minus_irrelevant':statistics.mean(v0-v1 for v0,v1 in doc_scores.values()),
                 'paired_informative_vs_irrelevant':{
                      'rescue':sum(r['correct']['informative'] and not r['correct']['irrelevant'] for r in rows),
                      'harm':sum(r['correct']['irrelevant'] and not r['correct']['informative'] for r in rows),
                      'both_correct':sum(r['correct']['irrelevant'] and r['correct']['informative'] for r in rows)},
                 'local_status':{},'descriptive_ci':_boot_delta(doc_scores,seed)}
        for status in ('sufficient','insufficient','uncertain'):
            sub=[r for r in rows if r['local_status']==status]
            if sub:summary['local_status'][status]={c:sum(int(r['correct'][c]) for r in sub)/len(sub) for c in CONDITIONS} | {'n':len(sub)}
        out.append(summary)
    return {'status':'EXPLORATORY_NOT_MECHANISM', 'disclaimer':'Toy examples, synthetic smoke and non-human reviewed records must not be presented as authentic evaluation.',
            'models':out,'no_internal_causal_mechanism_claim':True}

def candidate_intake(rows):
    """Metadata-only feasibility audit for an unadjudicated corpus candidate manifest.
    Never interprets syntactic roles as agent/patient and never returns source text.
    """
    genres=collections.Counter(str(r.get('genre','unknown')) for r in rows)
    docs={r.get('document_id') for r in rows if r.get('document_id')}
    ready=sum(bool(r.get('human_verified')) and r.get('rights')=='approved_for_internal_model_evaluation'
              and bool(r.get('informative_vs_irrelevant_human_adjudicated')) for r in rows)
    return {'status':'NOT_READY','candidate_sentences':len(rows),'documents':len(docs),
            'genres':dict(sorted(genres.items())),
            'human_semantic_labels_recorded':sum(bool(r.get('human_verified')) for r in rows),
            'source_rights_cleared':sum(r.get('rights')=='approved_for_internal_model_evaluation' for r in rows),
            'human_context_contrast_verified':sum(bool(r.get('informative_vs_irrelevant_human_adjudicated')) for r in rows),
            'fully_eligible_for_bx004':ready,
            'warning':'This metadata-only gate cannot establish human reviewer identity, accuracy of adjudication, corpus rights or actual semantic usefulness of context.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit-candidates');p.add_argument('--manifest');p.add_argument('--gold');p.add_argument('--model-inputs');p.add_argument('--responses')
    p.add_argument('--min-documents',type=int,default=10);p.add_argument('--min-families',type=int,default=2);p.add_argument('--out');a=p.parse_args()
    if a.audit_candidates:
        result=candidate_intake(load_jsonl(a.audit_candidates))
    else:
        if not all((a.manifest,a.gold,a.model_inputs,a.responses)):
            p.error('provide --manifest, --gold, --model-inputs and --responses for actual evaluation')
        v=validate(load_jsonl(a.manifest),load_jsonl(a.gold),load_jsonl(a.model_inputs),load_jsonl(a.responses),min_docs=a.min_documents,families=a.min_families)
        result=evaluate(v)
    txt=json.dumps(result,indent=2)+'\n'
    if a.out:Path(a.out).write_text(txt)
    print(txt)
    if result['status'] in ('BLOCKED','NOT_READY'):raise SystemExit(2)
if __name__=='__main__':main()
