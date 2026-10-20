"""One frozen candidate, development only, with every forward journaled."""
import json,time,traceback
import numpy as np
from transformers import AutoTokenizer
from core import *
from runtime import Engine,resources
from prepare import key,PARENT

def load_payload(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].astype(np.float32) for k in z.files}

def main():
    lock=json.loads((HERE/'LOCK.json').read_text())
    for name,h in lock['files'].items():assert sha(HERE/name)==h,('Frozen source',name)
    for name,h in lock['native_sources'].items():assert sha(ROOT/name)==h
    for name,v in lock['model']['files'].items():assert sha(ROOT/name)==v['sha256']
    manifest=json.loads((HERE/'SOURCE_MANIFEST.json').read_text())
    for name,h in manifest['parent_files'].items():assert sha(PARENT/name)==h
    assert (HERE/'FREEZE_RECEIPT.json').exists() and not (HERE/'STARTED.json').exists()
    resources();save(HERE/'STARTED.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),lock_sha256=sha(HERE/'LOCK.json')))
    rows={r['id']:r for r in json.loads((HERE/'INPUTS.json').read_text())}
    parent={r['id']:r for r in map(json.loads,(PARENT/'ANSWERS.jsonl').read_text().splitlines())}
    parentcalls={r['call']:r for r in map(json.loads,(PARENT/'FORWARDS.jsonl').read_text().splitlines())}
    tok=AutoTokenizer.from_pretrained(str(ROOT/lock['model']['path']),local_files_only=True,trust_remote_code=False)
    engine=Engine(lock);completed=[];caches={};prefixes={}
    shared=[0,1,2,6,7]
    def check(name,ok,**extra):
        append(HERE/'CONTROLS.jsonl',dict(check=name,passed=bool(ok),**extra));assert ok,(name,extra)
    try:
        for ident in json.loads((HERE/'CACHE_INPUTS.json').read_text()):
            row=rows[ident];old=parent[ident];assert sha(PARENT/old['base_payload'])==old['base_payload_sha256']
            data,rec=engine.dispatch(row,(),True,'donor_source_recapture')
            prior=load_payload(PARENT/old['base_payload'])
            le=float(np.max(np.abs(data['logits']-prior['logits'])))
            te=float(np.max(np.abs(data['trace'][:,shared]-prior['trace'])))
            check('donor_recapture_equivalence',le<=.05 and te<=1e-6 and np.argmax(data['logits'])==np.argmax(prior['logits']) and rec['prefix_hashes']==parentcalls[old['base_call']]['prefix_hashes'],id=ident,call=rec['call'],logit_error=le,trace_error=te)
            check('eight_anchor_shape',data['trace'].shape==(29,8,3584),id=ident,call=rec['call'])
            caches[key(row)]=rec;append(HERE/'DONOR_CACHE.jsonl',dict(context=key(row),input_id=ident,**rec))
            print('CACHE',len(caches),'of36',ident,flush=True)
        for identity in json.loads((HERE/'IDENTITY_INPUTS.json').read_text()):
            ident=identity['input_id'];block=identity['block']
            row=dict(rows[ident]);cache=caches[key(row)]
            row['edit']=dict(block=block,recipient_positions=row['anchors'],donor_payload=cache['payload'],donor_sites=[0,1,2])
            data,rec=engine.dispatch(row,(),True,'identity_patch')
            old=parent[ident];prior=load_payload(PARENT/old['base_payload'])
            le=float(np.max(np.abs(data['logits']-prior['logits'])));te=float(np.max(np.abs(data['trace'][:,shared]-prior['trace'])))
            check('identity_patch_equivalence',le<=.05 and te<=1e-6 and np.argmax(data['logits'])==np.argmax(prior['logits']) and rec['prefix_hashes']==parentcalls[old['base_call']]['prefix_hashes'],id=ident,block=block,call=rec['call'],logit_error=le,trace_error=te)
        jobs=json.loads((HERE/'JOBS.json').read_text())
        for job in jobs:
            original=rows[job['recipient_id']];row=dict(original);row['id']=job['id']
            donor=caches[job['donor_context']];assert sha(HERE/donor['payload'])==donor['payload_sha256']
            neighbor=job['condition']=='neighbor'
            block=job['block']
            row['edit']=dict(block=block,recipient_positions=row['neighbors'] if neighbor else row['anchors'],donor_payload=donor['payload'],donor_sites=[3,4,5] if neighbor else [0,1,2])
            base,rec=engine.dispatch(row,(),True,'causal_baseline')
            old=parent[job['recipient_id']];prior=load_payload(PARENT/old['base_payload'])
            te=float(np.max(np.abs(base['trace'][:block,shared]-prior['trace'][:block])))
            check('before_cut_equivalence',te<=1e-6 and rec['prefix_hashes'][:block-1]==parentcalls[old['base_call']]['prefix_hashes'][:block-1],id=job['id'],call=rec['call'],trace_error=te)
            donorarrays=load_payload(HERE/donor['payload']);siteidx=[3,4,5] if neighbor else [0,1,2]
            pe=float(np.max(np.abs(base['trace'][block,siteidx]-donorarrays['trace'][block,siteidx])))
            check('exact_patch_target',pe<=1e-6,id=job['id'],call=rec['call'],trace_error=pe)
            group=(block,job['condition'],key(original),job['donor_context'])
            if group in prefixes:check('patched_source_future_query_invariance',prefixes[group]==rec['prefix_hashes'],id=job['id'],call=rec['call'])
            else:prefixes[group]=rec['prefix_hashes']
            logits={():base['logits']};calls={():rec['call']}
            def get(prefix):
                prefix=tuple(prefix)
                if prefix not in logits:
                    data,entry=engine.dispatch(row,prefix,False,'causal_continuation')
                    err=float(np.max(np.abs(data['original_logits']-logits[()])))
                    check('continuation_original_patch_invariance',err<=.05,id=job['id'],call=entry['call'],logit_error=err)
                    logits[prefix]=data['logits'];calls[prefix]=entry['call']
                return logits[prefix]
            result=full_score(row,get,tok.decode)
            drow=next(r for r in rows.values() if key(r)==job['donor_context'])
            text=result['generated_text'].strip();terminated=result['terminated']
            scored={**job,**result,'recipient_source':row['source'],'donor_source':drow['source'],
                    'hypothesis_matches':{h:bool(terminated and text==pred) for h,pred in job['predictions'].items()},
                    'donor_name_compliant':bool(terminated and text in drow['names']),
                    'base_call':rec['call'],'base_payload':rec['payload'],'base_payload_sha256':rec['payload_sha256'],
                    'donor_cache_call':donor['call'],'edit':row['edit'],
                    'logit_prefix_calls':[dict(prefix=list(p),call=calls[p]) for p in sorted(calls)]}
            append(HERE/'ANSWERS.jsonl',scored);completed.append(scored)
            print('ANSWER',len(completed),'of432',job['id'],repr(result['generated_text']),scored['hypothesis_matches'],flush=True)
            del base,prior,donorarrays,logits
        check('all_432_causal_answers',len(completed)==432)
        save(HERE/'COLLECTION.json',dict(status='COMPLETE_FIXED_DEPTH_PROFILE',answers=432,donor_recaptures=36,identity_inputs=len(json.loads((HERE/'IDENTITY_INPUTS.json').read_text())),actual_model_forwards=engine.calls,workers=engine.worker,held_out_forwards=0,llama_forwards=0))
    except BaseException:
        save(HERE/'ERROR.json',dict(traceback=traceback.format_exc(),completed_answers=len(completed),attempts=engine.calls,no_automatic_retry=True));raise
    finally:engine.close()
    from analyze import main as analyze
    analyze()
if __name__=='__main__':main()
