"""Construct and freeze exactly32 untouched behavioral cases; no forwards."""
import json,tarfile,collections,importlib.metadata
from pathlib import Path
from transformers import AutoTokenizer
from core import *
def main():
    assert not (HERE/'LOCK.json').exists()
    parent=ROOT/'research/voice_query_code_factorial_checkpointed_2026-10-19'
    old=json.loads((parent/'LOCK.json').read_text());model=old['model']
    tok=AutoTokenizer.from_pretrained(str(ROOT/model['path']),local_files_only=True,trust_remote_code=False)
    pad=tok.encode('x',add_special_tokens=False);assert len(pad)==1
    rows=[]
    for family,a,b,past,base in FAMILIES:
        for role in [0,1]:
            agent,patient=(a,b) if role==0 else (b,a)
            sentence=f'{agent} {past} {patient}.'
            for query in [0,1]:
                sub,obj=(a,b) if query==0 else (b,a)
                question=f'Did {sub} {base} {obj}?'
                for mapping in [0,1]:
                    ent,non=('A','B') if mapping==0 else ('B','A')
                    for order in ['entailed_first','nonentailed_first']:
                        parts=[f'entailed = {ent}',f'not entailed = {non}']
                        if order=='nonentailed_first':parts.reverse()
                        legend='Response legend: '+'; '.join(parts)+'.'
                        user=BACKGROUND+'Sentence: '+sentence+'\nQuestion: '+question+'\n'+legend
                        prompt=tok.apply_chat_template([dict(role='system',content=SYSTEM),dict(role='user',content=user)],tokenize=False,add_generation_prompt=True)
                        enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];offsets=enc['offset_mapping'];assert len(ids)<LENGTH
                        at=lambda pos:next(i for i,(lo,hi) in enumerate(offsets) if lo<=pos<hi)
                        sc=prompt.index('Sentence: ')+len('Sentence: ');pc=sc+len(sentence)-1;qc=prompt.index('Question: ')+len('Question: ')+len(question)-1
                        aliases={}
                        for cls,word in [('entailed',ent),('not_entailed',non)]:
                            aa=[tok.encode(w,add_special_tokens=False) for w in [word,' '+word]];assert all(len(v)==1 for v in aa);aliases[cls]=sorted({v[0] for v in aa})
                        expected='entailed' if role==query else 'not_entailed'
                        rows.append(dict(id=f'{family}_r{role}_q{query}_m{mapping}_{order}',family=family,role=role,query=query,mapping=mapping,legend_order=order,last_class='not_entailed' if order=='entailed_first' else 'entailed',last_symbol=non if order=='entailed_first' else ent,expected=expected,expected_symbol=ent if expected=='entailed' else non,sentence=sentence,question=question,legend=legend,prompt=prompt,token_ids=ids,offsets=offsets,source_start=at(sc),period=at(pc),question_end=at(qc),prompt_end=len(ids)-1,pad=pad[0],aliases=aliases))
    assert len(rows)==32 and len({r['id'] for r in rows})==32
    import random
    random.Random(20261020).shuffle(rows)
    checks=[]
    for r in rows:
        other=next(x for x in rows if all(x[k]==r[k] for k in ['family','role','query','mapping']) and x['legend_order']!=r['legend_order'])
        assert r['token_ids'][:r['question_end']+1]==other['token_ids'][:other['question_end']+1]
        assert (r['period'],r['question_end'],r['prompt_end'])==(other['period'],other['question_end'],other['prompt_end'])
        assert collections.Counter(r['token_ids'])==collections.Counter(other['token_ids'])
        checks.append(dict(id=r['id'],source_question_prefix_exact=True,token_count_and_multiset_equal=True,end_positions_equal=True))
    known=set()
    def visit(x):
        if isinstance(x,dict):
            if isinstance(x.get('sentence'),str):known.add(x['sentence'])
            for v in x.values():visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    for p in (ROOT/'research').glob('**/INPUTS.json'):
        if HERE not in p.parents:visit(json.loads(p.read_text()))
    assert not {r['sentence'] for r in rows}&known,'Already used complete sentence'
    prior=[json.loads(s) for s in (parent/'qwen7b/RAW.jsonl').read_text().splitlines()]
    ref=next(r for r in prior if r['kind']=='validation_unwrapped')
    control=next(r for r in json.loads((parent/'INPUTS.json').read_text()) if r['id']==ref['id'])
    index=json.loads((parent/'PAYLOAD_BUNDLES.json').read_text());entry=index['members'][ref['payload']]
    with tarfile.open(parent/entry['bundle'],'r:') as tar:b=tar.extractfile(ref['payload']).read()
    import hashlib
    assert hashlib.sha256(b).hexdigest()==ref['payload_sha256']
    (HERE/'REFERENCE_CONTROL.npz').write_bytes(b)
    save(HERE/'CONTROL_INPUT.json',control);save(HERE/'INPUTS.json',rows);save(HERE/'TOKENIZER_AUDIT.json',checks)
    save(HERE/'SOURCE_MANIFEST.json',dict(annotation='AI-only authored and reviewed. No human validation. Two asymmetric directed-predicate clauses, both role orientations. Unsupported converse is not entailed, not asserted false. Names and predicate vary jointly across families; no population inference.',families=FAMILIES,new_complete_sentences=True,control=dict(parent_commit='33cd9e4974e82a8545bcefac4bc7f3364b321c69',record=ref,payload_reference_sha256=ref['payload_sha256']),limits='Only active sentences/questions, A/B, two authored families, same instruction and background. Legend order also changes category recency and symbol position together; last-option following does not identify a literal copying circuit.'))
    from test_design import check
    save(HERE/'CPU_PREFLIGHT.json',check(rows))
    for n,d in model['files'].items():assert sha(ROOT/n)==d['sha256'],n
    files=['core.py','prepare.py','run.py','analyze.py','test_design.py','PROTOCOL.md','INPUTS.json','CONTROL_INPUT.json','REFERENCE_CONTROL.npz','TOKENIZER_AUDIT.json','SOURCE_MANIFEST.json','CPU_PREFLIGHT.json']
    save(HERE/'LOCK.json',dict(files={n:sha(HERE/n) for n in files},model=model,native_sources=old['native_sources'],runtime={p:importlib.metadata.version(p) for p in ['mlx','mlx-lm','numpy','transformers']},max_forwards=33,scientific=32,exposed_control=1,physical_length=128,first_new_outcome_seen=False))
    print('Frozen32 untouched cases plus1 exposed control; no inference.')
if __name__=='__main__':main()
