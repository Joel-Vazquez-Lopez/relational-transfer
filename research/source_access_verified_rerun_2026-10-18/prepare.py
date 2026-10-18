"""Tokenizer/static-source preflight only; never imports MLX or loads a model."""
import importlib.metadata
import json
from pathlib import Path
import sysconfig
import numpy as np
from transformers import AutoTokenizer
from core import BLOCKS, LENGTH, CONDITIONS, mask, save, sha, toy_calibration

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent.parent
MODELS={'qwen7b':'research/stronger_model_development_2026-10-15/models/qwen7b',
        'llama3b':'research/llama_gemma_cross_family_2026-10-15/model'}
SYSTEM="Answer the question using only the sentence. Reply with only the person's name."
BACKGROUND='Background: The wooden table stood beside the quiet window in the empty room.\nSentence: '
FAMILIES=[('d1','development','Nina','Luke','followed','active'),
          ('d2','development','Emma','Jack','visited','active'),
          ('e1','evaluation','Helen','Martin','followed','passive'),
          ('e2','evaluation','Sarah','James','visited','passive')]

def main():
    assert not (HERE/'LOCK.json').exists(), 'Already frozen'
    rows=[]; sources={}; audits=[]
    for model,path in MODELS.items():
        p=ROOT/path; tok=AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False)
        assert tok.is_fast
        pad=tok.encode('x',add_special_tokens=False); assert len(pad)==1
        cfg=json.loads((p/'config.json').read_text()); assert cfg['num_hidden_layers']==BLOCKS
        files=sorted([*p.glob('*.safetensors'),*p.glob('*.json'),*p.glob('*.jinja')])
        sources[model]=dict(path=path,pad=pad[0],files={str(f.relative_to(ROOT)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in files})
        for family,split,a,b,verb,voice in FAMILIES:
            names=[a,b]; aliases={}; sequences={}
            for n in names:
                forms={s:tok.encode(s,add_special_tokens=False) for s in [n,' '+n]}
                aliases[n]=sorted({ids[0] for ids in forms.values() if len(ids)==1 and tok.decode(ids).strip()==n})
                assert forms[n] and all(0<len(ids)<=2 for ids in forms.values()),(model,n,forms)
                sequences[n]=[list(x) for x in sorted({tuple(ids) for ids in forms.values()})]
                audits.append(dict(model=model,name=n,forms=forms,accepted_ids=aliases[n]))
            assert not set(aliases[a])&set(aliases[b])
            for orientation in [0,1]:
                agent,patient=(a,b) if orientation==0 else (b,a)
                first=agent if voice=='active' else patient
                sentence=f'{agent} {verb} {patient}.' if voice=='active' else f'{patient} was {verb} by {agent}.'
                for q,question,gold in [('agent',f'Who {verb} someone?',agent),('first','Who was mentioned first in the sentence?',first)]:
                    user=BACKGROUND+sentence+'\nQuestion: '+question
                    prompt=tok.apply_chat_template([dict(role='system',content=SYSTEM),dict(role='user',content=user)],tokenize=False,add_generation_prompt=True)
                    enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
                    ids=enc['input_ids']; offsets=enc['offset_mapping']; assert len(ids)+4<=LENGTH
                    event_char=prompt.index(BACKGROUND)+len(BACKGROUND)
                    qchar=prompt.index('\nQuestion: ',event_char)+len('\nQuestion: ')
                    # Tokens overlapping the first character belong to that region, including preceding whitespace.
                    start=next(i for i,(lo,hi) in enumerate(offsets) if hi>event_char and lo<=event_char)
                    suffix=next(i for i,(lo,hi) in enumerate(offsets) if hi>qchar and lo<=qchar)
                    width=suffix-start; assert width>0
                    bgchar=prompt.index(BACKGROUND)
                    assert offsets[start-width][0]>=bgchar, 'Insufficient pre-event irrelevant source for matched cut'
                    r=dict(id=f'{family}_{orientation}_{q}',model=model,family=family,split=split,voice=voice,
                        orientation=orientation,query=q,names=names,expected=gold,agent=agent,patient=patient,
                        sentence=sentence,question=question,prompt=prompt,token_ids=ids,pad=pad[0],
                        offsets=offsets,source_span=[start,suffix],irrelevant_span=[start-width,start],
                        suffix_start=suffix,prompt_end=len(ids)-1,aliases=aliases,candidate_sequences=sequences,eos_ids=[int(tok.eos_token_id)])
                    rows.append(r)
                    for c in CONDITIONS:
                        for layer in range(BLOCKS):
                            m=mask(r,c,layer)
                            assert not m[np.triu_indices(LENGTH,1)].any()
                            assert np.array_equal(m[:suffix],np.tri(LENGTH,dtype=bool)[:suffix])
        # Equal-length swapped prompts are essential for the zero-access causal null.
        selected=[r for r in rows if r['model']==model]
        for family,*_ in FAMILIES:
            for q in ['agent','first']:
                pair=[r for r in selected if r['family']==family and r['query']==q]
                assert len(pair)==2 and pair[0]['source_span']==pair[1]['source_span']
                s=pair[0]['suffix_start']
                assert pair[0]['token_ids'][:pair[0]['source_span'][0]]==pair[1]['token_ids'][:pair[1]['source_span'][0]]
                assert pair[0]['token_ids'][s:]==pair[1]['token_ids'][s:]
    save(HERE/'INPUTS.json',rows); save(HERE/'TOKENIZER_AUDIT.json',audits)
    save(HERE/'TOY_CALIBRATION.json',toy_calibration())
    from test_integration import check
    save(HERE/'INTEGRATION_TESTS.json',check(rows))
    pkg=Path(sysconfig.get_paths()['purelib'])/'mlx_lm/models'
    import ast
    original=ROOT/'research/source_access_feasibility_2026-10-18'
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    previous_functions=functions(original/'core.py');new_functions=functions(HERE/'core.py')
    old_tail=(original/'run.py').read_text().split('    def stop(',1)[1]
    new_tail=(HERE/'run.py').read_text().split('    def stop(',1)[1].replace(':04d}',':03d}')
    invariance=dict(native_mask_wrapper_byte_identical=sha(original/'instrument.py')==sha(HERE/'instrument.py'),
        matched_summary_AST_identical=previous_functions['matched_summary']==new_functions['matched_summary'],
        selection_rule_AST_identical=previous_functions['select_prediction']==new_functions['select_prediction'],
        gate_and_phase_controller_text_identical_except_payload_filename_padding=old_tail==new_tail)
    assert all(invariance.values()),invariance
    save(HERE/'SCIENTIFIC_INVARIANCE.json',invariance)
    scripts=['SCIENTIFIC_INVARIANCE.json','scoring.py','engine.py','AMENDMENT.md','core.py','instrument.py','prepare.py','run.py','test_integration.py','analyze.py','PROTOCOL.md','ANNOTATION.md','PRIOR_WORK.md','INPUTS.json','TOKENIZER_AUDIT.json','TOY_CALIBRATION.json','INTEGRATION_TESTS.json']
    lock=dict(status='SCORING_AMENDMENT_FROZEN_BEFORE_RERUN_OUTPUTS_DEVELOPMENT_EXPOSED',authorization='User: can you verify it and run it again?',
        annotation='AI-only, per persistent explicit user authorization; no independent human review',
        files={n:sha(HERE/n) for n in scripts},models=sources,
        native_sources={str(f.relative_to(ROOT)):sha(f) for f in [pkg/'qwen2.py',pkg/'llama.py',pkg/'base.py']},
        runtime={p:importlib.metadata.version(p) for p in ['mlx','mlx-lm','numpy','transformers']},
        max_forwards=2000,max_per_model=1000,length=LENGTH,blocks=BLOCKS,
        conditions=CONDITIONS,source_branch='research/causal-developmental-relations-2026-10-17',
        source_parent='f35e167716a4b59be71fb5cf7a4185462f8a5f90',prior_attempt_forwards=11)
    old=ROOT/'research/source_access_feasibility_2026-10-18'
    previous=json.loads((old/'INPUTS.json').read_text())
    assert json.loads(json.dumps([{k:v for k,v in r.items() if k not in ['candidate_sequences','eos_ids']} for r in rows]))==previous
    max_candidate_prefixes=max(len({tuple(seq[:j]) for seqs in r['candidate_sequences'].values() for seq in seqs for j in range(1,len(seq))}) for r in rows)
    assert max_candidate_prefixes<=1, 'Revise the explicit worst-case call bound before freezing'
    lock['maximum_candidate_only_prefixes_per_record']=max_candidate_prefixes
    lock['original_inputs_sha256']=sha(old/'INPUTS.json')
    lock['original_lock_sha256']=sha(old/'LOCK.json')
    save(HERE/'LOCK.json',lock)
    print('FROZEN',len(rows),'tokenized rows; zero model calls',flush=True)

if __name__=='__main__': main()
