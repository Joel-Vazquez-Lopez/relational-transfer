"""CPU-only tokenizer/source preflight and source/model lock; no inference."""
import importlib.metadata
import itertools
import json
import math
import random
import re
import shutil
from transformers import AutoTokenizer
from core import *


def main():
    assert not (HERE/'LOCK.json').exists()
    prior=ROOT/'research/legend_order_diagnostic_2026-10-20'
    previous=json.loads((prior/'LOCK.json').read_text())
    model=previous['model']
    tok=AutoTokenizer.from_pretrained(str(ROOT/model['path']),local_files_only=True,trust_remote_code=False)
    inputs=json.loads((ROOT/'research/three_role_design_2026-10-20/CANDIDATE_INPUTS.json').read_text())
    rows=[r for r in inputs if r['split']=='development']
    assert len(rows)==108
    pad=tok.encode('x',add_special_tokens=False);assert len(pad)==1
    eos=sorted({tok.convert_tokens_to_ids('<|im_end|>'),tok.convert_tokens_to_ids('<|endoftext|>')})
    assert all(isinstance(x,int) for x in eos) and len(eos)==2
    checks=[]
    for row in rows:
        names=sorted({r['gold'] for r in rows if r['family']==row['family']});assert len(names)==3
        source=row['source']
        if row['form']=='active':
            m=re.fullmatch(r'(\w+) sent (\w+) to (\w+)\.',source);roles=list(m.groups())
        elif row['form']=='passive':
            m=re.fullmatch(r'(\w+) was sent to (\w+) by (\w+)\.',source);roles=[m[3],m[1],m[2]]
        else:
            m=re.fullmatch(r'To (\w+), (\w+) sent (\w+)\.',source);roles=[m[2],m[3],m[1]]
        expected=dict(zip(['sender','sent_person','destination_person'],roles))[row['query_role']]
        assert expected==row['gold']
        user='Sentence: '+source+'\nQuestion: '+row['question']
        prompt=tok.apply_chat_template([dict(role='system',content=row['system']),dict(role='user',content=user)],tokenize=False,add_generation_prompt=True)+'Answer:'
        enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];offsets=enc['offset_mapping']
        assert len(ids)+MAX_NEW<LENGTH
        at=lambda pos: next(i for i,(lo,hi) in enumerate(offsets) if lo<=pos<hi)
        start=prompt.index('Sentence: ')+len('Sentence: ')
        mentions=list(re.finditer(r'\b(?:'+'|'.join(names)+r')\b',source));assert len(mentions)==3
        anchors=[at(start+m.end()-1) for m in mentions]
        sequences={}
        for name in names:
            paths=[]
            for text in [' '+name]:
                seq=tok.encode(text,add_special_tokens=False)
                joint=tok.encode(prompt+text,add_special_tokens=False)
                assert joint[:len(ids)]==ids and joint[len(ids):]==seq,('Candidate boundary',name,text)
                assert seq and tok.decode(seq).strip()==name
                paths.append(seq)
            sequences[name]=sorted({tuple(x) for x in paths})
        flat=[(name,seq) for name,seqs in sequences.items() for seq in seqs]
        assert all(not (len(a)<=len(b) and tuple(b[:len(a)])==tuple(a)) for n,a in flat for m,b in flat if n!=m),'Candidate prefix overlap'
        proper={tuple(seq[:j]) for _,seq in flat for j in range(len(seq))}
        row.update(prompt=prompt,token_ids=ids,pad=pad[0],period=at(start+len(source)-1),anchors=anchors,eos_ids=eos,names=names,candidate_sequences=sequences,max_unique_scientific_forwards=len(proper)+MAX_NEW-1)
        checks.append(dict(id=row['id'],parsed_gold=expected,candidate_boundary_verified=True,offset_anchors=anchors,period=row['period'],prompt_tokens=len(ids),max_unique_forwards=row['max_unique_scientific_forwards']))
    keys=sorted({(r['family'],tuple(r['assignment']),r['form']) for r in rows});random.Random(20261020).shuffle(keys)
    ordered=[]
    for key in keys:
        group=[r for r in rows if (r['family'],tuple(r['assignment']),r['form'])==key]
        assert len(group)==3
        ref=group[0]
        assert all(r['token_ids'][:r['period']+1]==ref['token_ids'][:ref['period']+1] and r['period']==ref['period'] for r in group)
        ordered+=sorted(group,key=lambda r:['sender','sent_person','destination_person'].index(r['query_role']))
    # Exact source freshness relative to prior INPUTS files, excluding this proposal.
    prior_sources=set()
    def visit(x):
        if isinstance(x,dict):
            for k in ['source','sentence']:
                if isinstance(x.get(k),str):prior_sources.add(x[k])
            for value in x.values():visit(value)
        elif isinstance(x,list):
            for value in x:visit(value)
    for path in (ROOT/'research').glob('**/INPUTS.json'):
        if HERE not in path.parents:visit(json.loads(path.read_text()))
    assert not {r['source'] for r in ordered}&prior_sources
    save(HERE/'INPUTS.json',ordered);save(HERE/'TOKENIZER_AUDIT.json',checks)
    shutil.copyfile(prior/'CONTROL_INPUT.json',HERE/'CONTROL_INPUT.json')
    shutil.copyfile(prior/'REFERENCE_CONTROL.npz',HERE/'REFERENCE_CONTROL.npz')
    from test_scoring import check
    save(HERE/'CPU_PREFLIGHT.json',check(ordered))
    nonrestart=sum(r['max_unique_scientific_forwards'] for r in ordered)+3
    max_calls=nonrestart+math.ceil(nonrestart/(WORKER_CAP-1))
    for name,value in model['files'].items():assert sha(ROOT/name)==value['sha256'],name
    save(HERE/'SOURCE_MANIFEST.json',dict(parent_design_commit='5979355249416f967d447a02bff7040a2993f57e',proposal_sha256=sha(ROOT/'research/three_role_design_2026-10-20/CANDIDATE_INPUTS.json'),scientific_families=['t01','t02'],scientific_prompts=108,held_out_families_not_executed=['t03','t04'],annotation='AI-only authored and independently template-parsed; sender/sent-person/destination are grammatical task roles. Not human-validated or a corpus sample.',freshness='No exact development source sentence in inspected prior research/**/INPUTS.json files; proposal itself was already exposed. This is not a global corpus/train contamination claim.',control=dict(source='research/legend_order_diagnostic_2026-10-20',reference_sha256=sha(HERE/'REFERENCE_CONTROL.npz'))))
    files=['core.py','prepare.py','worker.py','runtime.py','run.py','analyze.py','test_scoring.py','PROTOCOL.md','INPUTS.json','TOKENIZER_AUDIT.json','CPU_PREFLIGHT.json','SOURCE_MANIFEST.json','CONTROL_INPUT.json','REFERENCE_CONTROL.npz']
    save(HERE/'LOCK.json',dict(files={n:sha(HERE/n) for n in files},native_sources=previous['native_sources'],model=model,max_model_forwards=max_calls,max_nonrestart_forwards=nonrestart,max_worker_forwards=WORKER_CAP,physical_length=LENGTH,max_new_tokens=MAX_NEW,runtime={p:importlib.metadata.version(p) for p in ['mlx','mlx-lm','numpy','transformers']},development_only=True,first_new_model_outcome_seen=False))
    print(json.dumps(dict(development_prompts=108,token_lengths=sorted({len(r['token_ids']) for r in ordered}),max_model_forwards=max_calls,max_nonrestart_forwards=nonrestart,cpu_checks=json.loads((HERE/'CPU_PREFLIGHT.json').read_text())),indent=2))

if __name__=='__main__':main()
