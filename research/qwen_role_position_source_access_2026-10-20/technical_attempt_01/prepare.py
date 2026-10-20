"""Outcome-blind exact inputs, annotations, tokenizer parity and resource budget."""
import itertools
import importlib.metadata
import re
import shutil
from transformers import AutoTokenizer
from core import *

def main():
    assert not (HERE/'LOCK.json').exists()
    prior=ROOT/'research/three_role_native_development_2026-10-20';previous=json.loads((prior/'LOCK.json').read_text())
    model=previous['model'];tok=AutoTokenizer.from_pretrained(str(ROOT/model['path']),local_files_only=True,trust_remote_code=False)
    questions=[['Who sent someone to another person?','Who was sent to another person?','Who was someone sent to?'],['Which person did the sending?','Which person was sent by someone?','To which person was someone sent?']]
    eos=json.loads((ROOT/model['path']/'config.json').read_text())['eos_token_id'];rows=[];checks=[]
    system="Read the passage and answer the question about the sending event. Return only the person's name, copied exactly from the passage. Do not explain."
    for fi,(family,names) in enumerate([('p01',['Grace','Helen','Jack','Kate']),('p02',['Nora','Luke','Paul','Mark'])]):
        candidate_ids={n:tok.encode(' '+n,add_special_tokens=False)[0] for n in names}
        assert all(tok.encode(' '+n,add_special_tokens=False)==[candidate_ids[n]] for n in names)
        for rotation,form in itertools.product(range(4),FORMS):
            base=[(i+rotation)%4 for i in range(4)]
            for variant in ['base','swap0','swap1','swap2','sham']:
                assignment=base.copy()
                if variant.startswith('swap'):
                    ri=int(variant[-1]);assignment[ri],assignment[3]=assignment[3],assignment[ri]
                event=source(names,assignment,form)
                passage=event+' '+names[assignment[3]]+(' stood nearby.' if variant=='sham' else ' waited nearby.')
                if form=='active':
                    m=re.fullmatch(r'(\w+) sent (\w+) to (\w+)\.',event);parsed=[m[1],m[2],m[3]]
                elif form=='passive':
                    m=re.fullmatch(r'(\w+) was sent to (\w+) by (\w+)\.',event);parsed=[m[3],m[1],m[2]]
                else:
                    m=re.fullmatch(r'To (\w+), (\w+) sent (\w+)\.',event);parsed=[m[2],m[3],m[1]]
                assert parsed==[names[i] for i in assignment[:3]]
                for qi,question in enumerate(questions[fi]):
                    user='Passage: '+passage+'\nQuestion: '+question
                    prompt=tok.apply_chat_template([dict(role='system',content=system),dict(role='user',content=user)],tokenize=False,add_generation_prompt=True)+'Answer:'
                    enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];offsets=enc['offset_mapping']
                    start=prompt.index('Passage: ')+len('Passage: ')
                    at=lambda p:next(i for i,(lo,hi) in enumerate(offsets) if lo<=p<hi)
                    mentions=list(re.finditer(r'\b(?:'+'|'.join(names)+r')\b',passage))
                    assert len(mentions)==4 and {m.group() for m in mentions}==set(names)
                    anchors=[at(start+m.end()-1) for m in mentions]
                    assert all(ids[a]==candidate_ids[m.group()] for a,m in zip(anchors,mentions))
                    for n,t in candidate_ids.items():assert tok.encode(prompt+' '+n,add_special_tokens=False)==ids+[t]
                    assert len(ids)+MAX_NEW<LENGTH
                    row=dict(id=f'{family}_r{rotation}_{form}_{variant}_q{qi}',family=family,rotation=rotation,form=form,variant=variant,assignment=assignment,source=passage,event=event,question=question,query_index=qi,role_index=qi,gold=parsed[qi],names=names,candidate_ids=candidate_ids,token_ids=ids,prompt=prompt,pad=87,eos_ids=eos,period=at(start+len(passage)-1),event_period=at(start+len(event)-1),anchors=anchors,target_token_index=anchors[surface_roles(form).index(qi)],position=surface_roles(form).index(qi),question_tokens=len(tok.encode(question,add_special_tokens=False)))
                    rows.append(row);checks.append(dict(id=row['id'],length=len(ids),period=row['period'],target_token_index=row['target_token_index'],question_tokens=row['question_tokens'],parsed_gold=parsed[qi],four_single_token_mentions=True))
    assert len(rows)==360
    lookup={(r['family'],r['rotation'],r['form'],r['variant'],r['query_index']):r for r in rows}
    units=[];shams=[]
    for family,rotation,form in itertools.product(['p01','p02'],range(4),FORMS):
        for ri in range(3):
            queries=[]
            for qi in range(3):
                a=lookup[family,rotation,form,'base',qi];b=lookup[family,rotation,form,f'swap{ri}',qi]
                assert len(a['token_ids'])==len(b['token_ids']) and a['period']==b['period']
                different=[j for j,(x,y) in enumerate(zip(a['token_ids'],b['token_ids'])) if x!=y]
                assert len(different)==2 and set(different)<=set(a['anchors'])
                assert (a['gold']!=b['gold'])==(qi==ri)
                queries.append(dict(a=a['id'],b=b['id'],query_index=qi,changed=qi==ri))
            units.append(dict(id=f'{family}_r{rotation}_{form}_role{ri}',family=family,rotation=rotation,form=form,role_index=ri,position=surface_roles(form).index(ri),queries=queries))
        for qi in range(3):
            a=lookup[family,rotation,form,'base',qi];b=lookup[family,rotation,form,'sham',qi]
            assert len(a['token_ids'])==len(b['token_ids']) and a['anchors']==b['anchors'] and a['period']==b['period']
            assert sum(x!=y for x,y in zip(a['token_ids'],b['token_ids']))==1 and a['gold']==b['gold']
            shams.append(dict(a=a['id'],b=b['id']))
    save(HERE/'INPUTS.json',rows);save(HERE/'UNITS.json',units);save(HERE/'SHAMS.json',shams);save(HERE/'TOKENIZER_AUDIT.json',checks)
    shutil.copyfile(prior/'CONTROL_INPUT.json',HERE/'CONTROL_INPUT.json');shutil.copyfile(prior/'REFERENCE_CONTROL.npz',HERE/'REFERENCE_CONTROL.npz')
    from test_design import main as test
    test()
    for n,v in model['files'].items():assert sha(ROOT/n)==v['sha256']
    save(HERE/'SOURCE_MANIFEST.json',dict(annotation='AI-only authored, independently regex-parsed; all arguments are people and task roles are sender/sent-person/destination. No human or corpus validation.',families=['p01','p02'],native_prompts=360,primary_native_prompts=288,sham_native_prompts=72,role_isolated_edges=72,held_out_prediction_rule='Other lexical quartet, other two constructions; six fixed folds',names_balanced='Each cyclic rotation places every name in each grammatical role and distractor status. Ordered pairs are not exhaustive.',question_paraphrase_scope='One wording set per quartet; joint lexical/paraphrase transfer, not separately identifiable factors.',old_evaluation_families_not_used=['t03','t04'],previous_stopped_study_commit='012658caccf8252383a528b126b75d586416a28c',model_escalation_after='a608f7989a731d38b3474a3a6e5dafbf31a152a6 Llama native gate failed166/288; no causal responses',model_choice_adaptive=True))
    names=['core.py','predictors.py','prepare.py','worker.py','runtime.py','run.py','analyze.py','test_design.py','PROTOCOL.md','PROTOCOL_GRADED.md','SUPERSESSION_BEFORE_INFERENCE.json','LITERATURE_SCOPE.md','MODEL_ESCALATION.md','stream.py','INPUTS.json','UNITS.json','SHAMS.json','TOKENIZER_AUDIT.json','CPU_PREFLIGHT.json','SOURCE_MANIFEST.json','CONTROL_INPUT.json','REFERENCE_CONTROL.npz']
    save(HERE/'LOCK.json',dict(files={n:sha(HERE/n) for n in names},native_sources={**previous['native_sources'],'research/stronger_model_development_2026-10-15/.venv/lib/python3.11/site-packages/mlx_lm/utils.py':sha(ROOT/'research/stronger_model_development_2026-10-15/.venv/lib/python3.11/site-packages/mlx_lm/utils.py')},model=model,max_model_forwards=4650,max_nonrestart_forwards=4503,physical_length=LENGTH,max_new_tokens=MAX_NEW,primary_measurement='Cross-fitted bias-cancelling role-contrast source access',runtime={p:importlib.metadata.version(p) for p in ['mlx','mlx-lm','numpy','transformers']},outcomes_seen=False,model_selection_after_llama_failure=True))
    print(json.dumps(dict(native_prompts=360,edges=72,shams=72,token_lengths=sorted({len(r['token_ids']) for r in rows}),max_model_forwards=4650)))

if __name__=='__main__':main()
