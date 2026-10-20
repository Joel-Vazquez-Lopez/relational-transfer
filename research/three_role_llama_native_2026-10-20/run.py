"""One bounded development collection; no held-out prompts or interventions."""
import json
import time
import traceback
import numpy as np
from transformers import AutoTokenizer
from core import *
from runtime import Engine, resources


def main():
    lock=json.loads((HERE/'LOCK.json').read_text())
    for name,h in lock['files'].items():assert sha(HERE/name)==h,('Frozen source',name)
    for name,h in lock['native_sources'].items():assert sha(ROOT/name)==h,('Native source',name)
    for name,value in lock['model']['files'].items():assert sha(ROOT/name)==value['sha256'],('Model source',name)
    assert (HERE/'FREEZE_RECEIPT.json').exists()
    assert not (HERE/'STARTED.json').exists(),'No automatic retries or overwrites'
    resources()
    save(HERE/'STARTED.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),lock_sha256=sha(HERE/'LOCK.json')))
    rows=json.loads((HERE/'INPUTS.json').read_text());assert len(rows)==108 and all(r['split']=='development' for r in rows)
    tok=AutoTokenizer.from_pretrained(str(ROOT/lock['model']['path']),local_files_only=True,trust_remote_code=False)
    engine=Engine(lock); completed=[];prefixes={}

    def check(name,ok,**metadata):
        append(HERE/'CONTROLS.jsonl',dict(check=name,passed=bool(ok),**metadata))
        assert ok,(name,metadata)

    try:
        first=rows[0]
        raw,a=engine.dispatch(first,(),False,'validation_unwrapped')
        wrapped,b=engine.dispatch(first,(),True,'validation_wrapper')
        repeat,c=engine.dispatch(first,(),True,'validation_repeat')
        err=float(np.max(np.abs(raw['logits']-wrapped['logits'])))
        check('native_wrapper_equivalence',err<=.05,error=err,calls=[a['call'],b['call']])
        err=float(np.max(np.abs(wrapped['logits']-repeat['logits'])))
        te=float(np.max(np.abs(wrapped['trace']-repeat['trace'])))
        check('repeat_equivalence',err<=.05 and te<=1e-6 and b['prefix_hashes']==c['prefix_hashes'],logit_error=err,trace_error=te,calls=[b['call'],c['call']])
        del raw,wrapped,repeat
        for row in rows:
            base,record=engine.dispatch(row,(),True,'scientific_baseline')
            cache={():base['logits']};calls={():record['call']}
            group=(row['family'],tuple(row['assignment']),row['form'])
            if group in prefixes:
                check('source_prefix_future_query_invariance',prefixes[group]==record['prefix_hashes'],id=row['id'],call=record['call'])
            else:prefixes[group]=record['prefix_hashes']
            check('native_trace_shape',base['trace'].shape==(29,5,3072),id=row['id'],call=record['call'])
            def get(prefix):
                prefix=tuple(prefix)
                if prefix not in cache:
                    data,entry=engine.dispatch(row,prefix,False,'scientific_continuation')
                    error=float(np.max(np.abs(data['original_logits']-cache[()])))
                    check('continuation_original_prefix_invariance',error<=.05,id=row['id'],call=entry['call'],error=error)
                    cache[prefix]=data['logits'];calls[prefix]=entry['call']
                return cache[prefix]
            result=full_score(row,get,tok.decode)
            scored={**{k:row[k] for k in ['id','family','split','assignment','form','source','question','query_role']},**result,
                    'base_call':record['call'],'logit_prefix_calls':[dict(prefix=list(p),call=calls[p]) for p in sorted(calls)],
                    'base_payload':record['payload'],'base_payload_sha256':record['payload_sha256']}
            append(HERE/'ANSWERS.jsonl',scored);completed.append(scored)
            print('ANSWER',len(completed),'of108',row['id'],repr(result['generated_text']),result['native_correct'],flush=True)
            del cache,base
        check('all_108_development_answers',len(completed)==108)
        save(HERE/'COLLECTION.json',dict(status='COMPLETE_108_DEVELOPMENT',answers=108,actual_model_forwards=engine.calls,workers=engine.worker,held_out_forwards=0,interventions=0))
    except BaseException:
        save(HERE/'ERROR.json',dict(traceback=traceback.format_exc(),completed_answers=len(completed),attempts=engine.calls,no_automatic_retry=True))
        raise
    finally:
        engine.close()
    from analyze import main as analyze
    analyze()

if __name__=='__main__':main()
