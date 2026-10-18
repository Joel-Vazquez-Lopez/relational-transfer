"""Static/tokenizer preflight, no MLX import or model forward."""
import importlib.metadata
import json
import shutil
import sysconfig
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from core import HERE, ROOT, MODELS, FAMILIES, LENGTH, BLOCKS, CUTS, MODES, mask, sha, save, calibration

SYSTEM = 'Answer the question using only the sentence. Reply with only Yes or No.'
BACKGROUND = ('Background: The shelf holds a plain folder beside a closed notebook. '
              'The wall is white and the room is quiet.\nSentence: ')

def main():
    assert not (HERE/'LOCK.json').exists(), 'Do not overwrite a frozen study'
    rows=[]; sources={}; audits=[]
    previous=json.loads((ROOT/'research/source_access_verified_rerun_2026-10-18/LOCK.json').read_text())
    for model,path in MODELS.items():
        p=ROOT/path; tok=AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False)
        assert tok.is_fast
        aliases={k:sorted({tok.encode(v,add_special_tokens=False)[0] for v in [k,' '+k]}) for k in ['Yes','No']}
        for k in aliases:
            for v in [k,' '+k]:
                ids=tok.encode(v,add_special_tokens=False)
                assert len(ids)==1 and tok.decode(ids).strip()==k,(model,v,ids)
        assert not set(aliases['Yes']) & set(aliases['No'])
        pad=tok.encode('x',add_special_tokens=False); assert len(pad)==1
        assert json.loads((p/'config.json').read_text())['num_hidden_layers']==BLOCKS
        # Verify exact existing model files; no inference, download or substitution.
        files={}
        for name,old in previous['models'][model]['files'].items():
            f=ROOT/name; h=sha(f); assert h==old['sha256'],('model changed',name)
            files[name]=dict(bytes=f.stat().st_size,sha256=h)
        sources[model]=dict(path=path,files=files,pad=pad[0])
        for family,split,a,b,past,base in FAMILIES:
            for role in [0,1]:
                agent,patient=(a,b) if role==0 else (b,a)
                for voice in ['active','passive']:
                    sentence=(f'{agent} {past} {patient}.' if voice=='active' else f'{patient} was {past} by {agent}.')
                    for q in [0,1]:
                        subject,obj=(a,b) if q==0 else (b,a)
                        question=f'Did {subject} {base} {obj}?'
                        expected='Yes' if subject==agent else 'No'
                        prompt=tok.apply_chat_template([dict(role='system',content=SYSTEM),
                            dict(role='user',content=BACKGROUND+sentence+'\nQuestion: '+question)],
                            tokenize=False,add_generation_prompt=True)
                        enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
                        ids,off=enc['input_ids'],enc['offset_mapping']; assert len(ids)<LENGTH
                        sc=prompt.index(BACKGROUND)+len(BACKGROUND); pc=sc+len(sentence)-1
                        s=next(i for i,(lo,hi) in enumerate(off) if hi>sc and lo<=sc)
                        period=next(i for i,(lo,hi) in enumerate(off) if hi>pc and lo<=pc)
                        assert tok.decode([ids[period]]).strip()=='.',(model,sentence,period)
                        assert period>s+1
                        width=period-s+1; lo=s-width
                        assert off[lo][0]>=prompt.index(BACKGROUND),'Neutral span escaped background'
                        row=dict(id=f'{family}_r{role}_{voice}_q{q}',model=model,family=family,split=split,
                            role=role,voice=voice,query=q,names=[a,b],agent=agent,patient=patient,
                            past=past,base=base,sentence=sentence,question=question,expected=expected,
                            prompt=prompt,token_ids=ids,offsets=off,pad=pad[0],aliases=aliases,
                            source_start=s,period=period,neutral_span=[lo,s],prompt_end=len(ids)-1)
                        rows.append(row)
                        native=np.tri(LENGTH,dtype=bool)
                        for mode in MODES:
                            for k in CUTS+[28]:
                                for block in [0,6,7,13,14,20,21,27]:
                                    m=mask(row,mode,k,block)
                                    if block<k: assert np.array_equal(m,native)
                                    if mode=='irrelevant':
                                        other=mask(row,'closed',k,block)
                                        assert np.count_nonzero(native & ~m)==np.count_nonzero(native & ~other)
                        # Exact pairwise intervention differences, not informal labels.
                        f=mask(row,'frozen',0,0); c=mask(row,'closed',0,0); live=mask(row,'live',0,0)
                        assert np.array_equal(np.argwhere(f & ~c),np.array([[i,period] for i in range(period+1,LENGTH)]))
                        assert np.array_equal(np.argwhere(live & ~f),np.array([[period,j] for j in range(s,period)]))
            rr=[r for r in rows if r['model']==model and r['family']==family]
            assert sum(r['expected']=='Yes' for r in rr)==4
            for voice in ['active','passive']:
                for q in [0,1]:
                    pair=[r for r in rr if r['voice']==voice and r['query']==q]
                    x,y=pair; assert (x['source_start'],x['period'],x['prompt_end'])==(y['source_start'],y['period'],y['prompt_end'])
                    assert x['token_ids'][:x['source_start']]==y['token_ids'][:y['source_start']]
                    assert x['token_ids'][x['period']:]==y['token_ids'][y['period']:]
            audits.append(dict(model=model,family=family,n=8,balanced_yes_no=True,
                               offsets_valid=True,disconnected_surviving_tokens_identical=True))
    save(HERE/'INPUTS.json',rows)
    save(HERE/'TOKENIZER_ANNOTATION_AUDIT.json',audits)
    save(HERE/'KNOWN_ROUTE_CALIBRATION.json',calibration())
    # Fit only exposed previous d1 data; no fresh outcome enters this rule.
    calibrators={}
    for model in MODELS:
        source=ROOT/'research/causal_carrier_isolation_2026-10-18'/model/'RAW.json'
        rr=[r for r in json.loads(source.read_text()) if r['kind']=='scientific']
        yes=[r['class_logits']['Yes']-r['class_logits']['No'] for r in rr if r['expected']=='Yes']
        no=[r['class_logits']['Yes']-r['class_logits']['No'] for r in rr if r['expected']=='No']
        assert min(yes)>max(no)
        calibrators[model]=dict(threshold=(min(yes)+max(no))/2,source=str(source.relative_to(ROOT)),
            source_sha256=sha(source),yes_scores=yes,no_scores=no,fitting_cases=8,
            rule='One constant midpoint threshold from prior exposed d1; never refit on fresh data')
    save(HERE/'READOUT_CALIBRATION.json',calibrators)
    old=ROOT/'research/causal_carrier_isolation_2026-10-18'
    oldinputs=json.loads((old/'INPUTS.json').read_text());oldprompts={r['prompt'] for r in oldinputs}
    assert not any(r['prompt'] in oldprompts for r in rows)
    save(HERE/'SOURCE_REUSE.json',dict(previous_study_unchanged=True,
        native_wrapper_byte_identical=sha(HERE/'instrument.py')==sha(old/'instrument.py'),
        prior_lock_sha256=sha(old/'LOCK.json'),new_prompts_disjoint_from_all_prior_carrier_inputs=True,
        engine_change='Added embedding capture only; exact original model forward and dense-mask mechanism reused',
        prior_engine_source_sha256=sha(old/'run.py')))
    from test_pipeline import check
    save(HERE/'PREFLIGHT_TESTS.json',check(rows,calibrators))
    pkg=Path(sysconfig.get_paths()['purelib'])/'mlx_lm/models'
    native={str(f.relative_to(ROOT)):sha(f) for f in [pkg/'qwen2.py',pkg/'llama.py',pkg/'base.py']}
    storage={}
    for model,path in MODELS.items():
        cfg=json.loads((ROOT/path/'config.json').read_text()); width=cfg['hidden_size']; vocab=cfg['vocab_size']
        storage[model]=dict(planned_calls=879,expected_lossless_float16_bytes=879*(vocab+29*2*width)*2)
    required=int(sum(v['expected_lossless_float16_bytes'] for v in storage.values())*1.25)+int(1.5*1024**3)
    available=shutil.disk_usage(HERE).free
    assert available>required,('Insufficient storage',available,required)
    save(HERE/'STORAGE_BUDGET.json',dict(models=storage,required_bytes=required,available_bytes=available,
        rule='Expected native-value lossless16 storage +25percent overhead +1.5GiB reserve; every call checks1GiB remaining'))
    files=['core.py','instrument.py','engine.py','prepare.py','run.py','test_pipeline.py','PROTOCOL.md','PREPARATION.md',
           'INPUTS.json','READOUT_CALIBRATION.json','TOKENIZER_ANNOTATION_AUDIT.json','KNOWN_ROUTE_CALIBRATION.json',
           'PREFLIGHT_TESTS.json','SOURCE_REUSE.json','STORAGE_BUDGET.json','DRAFT_LOCK_BEFORE_SUFFIX_REVIEW.json']
    lock=dict(status='FROZEN_BEFORE_ALL_NEW_OUTCOMES',
        authorization='Joel explicitly directed continuation after the closed pilot; broad independent local-resource research authorization retained',
        annotation='AI-only deterministic template annotation; no human validation',
        files={n:sha(HERE/n) for n in files},models=sources,native_sources=native,
        runtime={p:importlib.metadata.version(p) for p in ['mlx','mlx-lm','numpy','transformers']},
        max_forwards=1800,max_per_model=900,planned_max_per_model=879,
        source_parent='2c29bc72b98530482b5fd7ddf0bbc3596c266c04',source_branch='research/causal-developmental-relations-2026-10-17')
    save(HERE/'LOCK.json',lock)
    print('FROZEN',len(rows),'tokenized rows; zero new model calls',flush=True)

if __name__=='__main__': main()
