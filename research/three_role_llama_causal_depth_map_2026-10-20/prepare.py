"""CPU-only prospective causal protocol inputs. Never opens an inference engine."""
import itertools, json, math, re
from transformers import AutoTokenizer
from core import *
PARENT=ROOT/'research/three_role_llama_native_2026-10-20'
ROLES=['sender','sent_person','destination_person']
ORDERS={'active':[0,1,2],'passive':[1,2,0],'goal_fronted':[2,0,1]}

def key(r):return '|'.join([r['family'],''.join(map(str,r['assignment'])),r['form']])
def predicts(rec,donor,invariant):
    q=ROLES.index(rec['query_role']);dn=donor['names'];da=donor['assignment']
    mention=[dn[da[i]] for i in ORDERS[donor['form']]]
    role=dn[da[q]]
    return dict(role_table=role,mention_order=mention[ORDERS[rec['form']].index(q)],fixed_one_answer=dn[da[ROLES.index(invariant)]],syntax_aware_parser=role,complete_future_answer_table=role)

def main():
    assert not (HERE/'LOCK.json').exists()
    parent=json.loads((PARENT/'LOCK.json').read_text())
    results=json.loads((PARENT/'RESULTS.json').read_text());assert results['baseline_ready_for_causal_protocol']
    audit=json.loads((PARENT/'INDEPENDENT_AUDIT.json').read_text());assert audit['passed']
    rows=json.loads((PARENT/'INPUTS.json').read_text())
    native={r['id']:r for r in map(json.loads,(PARENT/'ANSWERS.jsonl').read_text().splitlines())}
    tok=AutoTokenizer.from_pretrained(str(ROOT/parent['model']['path']),local_files_only=True,trust_remote_code=False)
    for row in rows:
        offsets=tok(row['prompt'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
        start=row['prompt'].index('Sentence: ')+len('Sentence: ')
        spans=[(start+m.start(),start+m.end()) for m in re.finditer(r'\b(?:'+'|'.join(row['names'])+r')\b',row['source'])]
        namesites={i for i,(a,b) in enumerate(offsets) if any(a<hi and b>lo for lo,hi in spans)}
        neighbors=[]
        for lo,hi in spans:
            first=min(i for i,(a,b) in enumerate(offsets) if a<hi and b>lo)
            neighbors.append(max(i for i in range(first) if i not in namesites))
        assert len(set(neighbors))==3 and not set(neighbors)&namesites
        assert all(x<row['period'] for x in neighbors)
        row['neighbors']=neighbors;row['person_span_tokens']=sorted(namesites)
    lookup={(r['family'],tuple(r['assignment']),r['form'],r['query_role']):r for r in rows}
    pairs=json.loads((PARENT/'PAIR_LEDGER.json').read_text());units=[]
    for fam,inv in itertools.product(['t01','t02'],ROLES):
        choices=[p for p in pairs if p['family']==fam and p['invariant_role']==inv and p['eligible'] and p['recipient_form']!=p['donor_form']]
        pick=sorted(choices,key=lambda p:(p['recipient'],p['donor'],p['recipient_form'],p['donor_form']))[0]
        units.append(dict(unit=f'{fam}_{inv}',**pick))
    jobs=[];identity={}
    for u in units:
        for direction in ['forward','reverse']:
            ra,da,rf,df=(u['recipient'],u['donor'],u['recipient_form'],u['donor_form']) if direction=='forward' else (u['donor'],u['recipient'],u['donor_form'],u['recipient_form'])
            for q in ROLES:
                rec=lookup[u['family'],tuple(ra),rf,q];main=lookup[u['family'],tuple(da),df,q]
                same=lookup[u['family'],tuple(ra),df,q]
                other='t02' if u['family']=='t01' else 't01'
                foreign=lookup[other,tuple(da),df,q]
                identity[rec['id']]=rec
                for condition,donor in [('main',main),('same_role_voice',same),('neighbor',main),('cross_lexical',foreign)]:
                    predictions=predicts(rec,donor,u['invariant_role'])
                    jobs.append(dict(id=f"{u['unit']}_{direction}_{q}_{condition}",unit=u['unit'],direction=direction,condition=condition,recipient_id=rec['id'],donor_context=key(donor),donor_input_id=donor['id'],query_role=q,invariant_role=u['invariant_role'],predictions=predictions,recipient_native_correct=native[rec['id']]['native_correct'],donor_native_correct=native[donor['id']]['native_correct']))
    jobs=[dict(j,id=f"b{block:02d}_"+j['id'],block=block) for block in [7,14,21] for j in jobs]
    caches=sorted([r for r in rows if r['query_role']=='sender'],key=key)
    assert len(jobs)==432 and len(units)==6 and len(caches)==36
    save(HERE/'INPUTS.json',rows);save(HERE/'JOBS.json',jobs);save(HERE/'UNITS.json',units)
    save(HERE/'CACHE_INPUTS.json',[r['id'] for r in caches]);save(HERE/'IDENTITY_INPUTS.json',[dict(input_id=i,block=b) for b in [7,14,21] for i in sorted(identity)])
    save(HERE/'TOKENIZER_AUDIT.json',[{k:r[k] for k in ['id','anchors','neighbors','person_span_tokens','period']} for r in rows])
    from test_design import check
    save(HERE/'CPU_PREFLIGHT.json',check(rows,jobs,units,native))
    nonrestart=36+3*len(identity)+sum(next(r for r in rows if r['id']==j['recipient_id'])['max_unique_scientific_forwards'] for j in jobs)
    files=['core.py','prepare.py','replacement.py','worker.py','runtime.py','run.py','analyze.py','test_design.py','PROTOCOL.md','INPUTS.json','JOBS.json','UNITS.json','CACHE_INPUTS.json','IDENTITY_INPUTS.json','TOKENIZER_AUDIT.json','CPU_PREFLIGHT.json','SOURCE_MANIFEST.json','RESEARCH_RATIONALE.md','RESTART_AMENDMENT.md','CONTROL_INPUT.json','REFERENCE_CONTROL.npz']
    sourcepaths=['LOCK.json','INPUTS.json','ANSWERS.jsonl','PAIR_LEDGER.json','RESULTS.json','INDEPENDENT_AUDIT.json','FORWARDS.jsonl']
    save(HERE/'SOURCE_MANIFEST.json',dict(parent_directory=str(PARENT.relative_to(ROOT)),parent_files={n:sha(PARENT/n) for n in sourcepaths},annotation='AI-only template annotation; no human labels or corpus generalization.',selection='Lexicographically first native-eligible cross-form pair for each family and invariant role, using (recipient assignment, donor assignment, recipient form literal string, donor form literal string).',candidate='Prespecified depth profile at outgoing blocks 7,14,21, three person last subtokens in surface order. All depths reported; no adaptive additional site/depth.',no_heldout=True))
    save(HERE/'LOCK.json',dict(files={n:sha(HERE/n) for n in files},native_sources=parent['native_sources'],model=parent['model'],runtime=parent['runtime'],max_model_forwards=nonrestart+math.ceil(nonrestart/(WORKER_CAP-1)),max_nonrestart_forwards=nonrestart,development_only=True,first_causal_outcome_seen=False,blocks=[7,14,21],site='joint_person_last_subtokens',technical_logit_tolerance=.05,technical_trace_tolerance=1e-6,independent_probability_replay_tolerance=1e-8))
    print(json.dumps(dict(jobs=len(jobs),identity_inputs=3*len(identity),caches=len(caches),max_model_forwards=json.loads((HERE/'LOCK.json').read_text())['max_model_forwards']),indent=2))
if __name__=='__main__':main()
