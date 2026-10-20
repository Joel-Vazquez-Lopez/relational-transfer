"""One fixed native screen and gated causal comparison, without adaptive rescue."""
import time
import traceback
from transformers import AutoTokenizer
from core import *
from predictors import native_gate
from runtime import Engine,resources

def main():
    lock=json.loads((HERE/'LOCK.json').read_text())
    for n,h in lock['files'].items():assert sha(HERE/n)==h,('Frozen source',n)
    for n,h in lock['native_sources'].items():assert sha(ROOT/n)==h
    for n,v in lock['model']['files'].items():assert sha(ROOT/n)==v['sha256']
    assert (HERE/'FREEZE_RECEIPT.json').exists() and not (HERE/'STARTED.json').exists()
    resources();save(HERE/'STARTED.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),lock_sha256=sha(HERE/'LOCK.json')))
    rows=json.loads((HERE/'INPUTS.json').read_text());index={r['id']:r for r in rows}
    units=json.loads((HERE/'UNITS.json').read_text());shams=json.loads((HERE/'SHAMS.json').read_text())
    tok=AutoTokenizer.from_pretrained(str(ROOT/lock['model']['path']),local_files_only=True)
    engine=Engine(lock);native={};completed=0;sham_count=0
    def check(name,ok,**kw):
        append(HERE/'CONTROLS.jsonl',dict(check=name,passed=bool(ok),**kw));assert ok,(name,kw)
    def raw(path,key):
        with np.load(HERE/path,allow_pickle=False) as z:return z[key].astype(np.float32)
    def dispatch(row,prefix,capture,kind):
        data,entry=engine.dispatch(row,prefix,capture,kind)
        if row.get('edit'):
            e=row['edit'];before=raw(native[row['id']]['base_payload'],'states')[CUTS.index(e['cut'])]
            donor=raw(e['donor_payload'],'states')[CUTS.index(e['cut'])]
            left,right=(0,row['period']+1) if e['region']=='prefix' else (row['period']+1,len(row['token_ids'])) if e['region']=='suffix' else (0,len(row['token_ids']))
            after=before.copy();after[left:right]=donor[left:right];p=entry['patch_proof']
            check('exact_region_interchange',p['pre_sha256']==array_sha(before) and p['post_sha256']==array_sha(after) and p['tail_pre_sha256']==p['tail_post_sha256'],id=row['id'],call=entry['call'],edit=e)
        return data,entry
    def measure(row,kind,capture=False,full=True):
        data,entry=dispatch(row,(),capture,kind);cache={():data['logits']};calls={():entry['call']}
        def get(prefix):
            if prefix not in cache:
                z,r=dispatch(row,prefix,False,kind+'_continuation')
                err=float(np.max(np.abs(z['original_logits']-cache[()])))
                check('original_prompt_continuation_invariance',err<=.05,id=row['id'],call=r['call'],error=err)
                cache[prefix]=z['logits'];calls[prefix]=r['call']
            return cache[prefix]
        if full:result=score_generation(row,get,tok.decode)
        else:
            t=int(data['logits'].argmax())
            result=dict(first_token=t,first_token_text=tok.decode([t]),first_token_name=next((n for n,v in row['candidate_ids'].items() if v==t),None),name_logits={n:float(data['logits'][v]) for n,v in row['candidate_ids'].items()},generated_text=None,generated_tokens=None,terminated=None,actual=None)
        result.update(id=row['id'],gold=row['gold'],measurement='full_generation' if full else 'next_token_only',base_call=entry['call'],base_payload=entry['payload'],base_payload_sha256=entry['payload_sha256'],calls=[dict(prefix=list(p),call=c) for p,c in calls.items()],prefix_hashes=entry['prefix_hashes'])
        return result
    try:
        first=rows[0]
        plain,a=dispatch(first,(),False,'validation_unwrapped');wrapped,b=dispatch(first,(),True,'validation_wrapped');repeat,c=dispatch(first,(),True,'validation_repeat')
        error=float(np.max(abs(plain['logits']-wrapped['logits'])))
        check('wrapper_equivalence',error<=.05,error=error)
        check('repeat_equivalence',np.array_equal(wrapped['states'],repeat['states']) and np.array_equal(wrapped['readout'],repeat['readout']) and np.max(abs(wrapped['logits']-repeat['logits']))<=.05)
        del plain,wrapped,repeat
        prefixes={}
        for row in rows:
            result=measure(row,'native',capture=True)
            result.update({k:row[k] for k in ['family','rotation','form','variant','assignment','query_index','role_index','source','question','position']})
            native[row['id']]=result;append(HERE/'NATIVE.jsonl',result)
            key=(row['family'],row['rotation'],row['form'],row['variant'])
            if key in prefixes:check('future_question_source_invariance',prefixes[key]==result['prefix_hashes'],id=row['id'])
            else:prefixes[key]=result['prefix_hashes']
            print('NATIVE',len(native),'/360',row['id'],repr(result['generated_text']),result['native_correct'],flush=True)
        gate=native_gate(native,rows,units);save(HERE/'NATIVE_GATE.json',gate)
        save(HERE/'NATIVE_COLLECTION.json',dict(complete=True,native_responses=360,actual_model_forwards=engine.calls,workers=engine.worker))
        if not gate['passed']:
            save(HERE/'COLLECTION.json',dict(status='INCONCLUSIVE_NATIVE',native_responses=360,hybrid_measurements=0,actual_model_forwards=engine.calls,workers=engine.worker))
            print('STOP_NATIVE_GATE',gate['correct'],'/288',flush=True)
            from analyze import native_summary
            native_summary();return
        engine.started=time.monotonic()
        for u in units:
            if u['rotation']!=0 or u['role_index']!=0:continue
            for q in u['queries']:
                a,b=index[q['a']],index[q['b']]
                for cut in [0,28]:
                    for region in ['prefix','suffix']:
                        row={**a,'edit':dict(donor_payload=native[b['id']]['base_payload'],cut=cut,region=region)}
                        z,r=dispatch(row,(),False,'endpoint_control')
                        donor=b if (cut==0 and region=='prefix') or (cut==28 and region=='suffix') else a
                        err=float(np.max(abs(z['logits']-raw(native[donor['id']]['base_payload'],'logits'))))
                        check('forced_endpoint_full_vocab',err<=.05,call=r['call'],error=err,target=donor['id'])
                for cut in [7,14,21]:
                    for donor,label in [(a,'identity'),(b,'joint_swap')]:
                        row={**a,'edit':dict(donor_payload=native[donor['id']]['base_payload'],cut=cut,region='both')}
                        z,r=dispatch(row,(),False,label+'_control')
                        err=float(np.max(abs(z['logits']-raw(native[donor['id']]['base_payload'],'logits'))))
                        check(label+'_full_vocab',err<=.05,call=r['call'],error=err)
        save(HERE/'TECHNICAL_GATE.json',dict(passed=True,endpoint_forwards=72,intermediate_forwards=108))
        for cut in [7,14,21]:
            for u in units:
                qq=u['queries'] if cut==14 else [u['queries'][u['role_index']]]
                for q in qq:
                    a,b=index[q['a']],index[q['b']]
                    for hybrid,region in [('BA','prefix'),('AB','suffix')]:
                        row={**a,'edit':dict(donor_payload=native[b['id']]['base_payload'],cut=cut,region=region)}
                        result=measure(row,'hybrid',full=cut==14)
                        if 'native_correct' in result:result['full_generation_a_correct']=result.pop('native_correct')
                        if 'first_token_correct' in result:result['first_token_a_correct']=result.pop('first_token_correct')
                        result.update(unit=u['id'],family=u['family'],rotation=u['rotation'],form=u['form'],target_role=u['role_index'],position=u['position'],query_index=q['query_index'],changed=q['changed'],a=a['id'],b=b['id'],gold_a=a['gold'],gold_b=b['gold'],cut=cut,hybrid=hybrid,edit=row['edit'])
                        append(HERE/'HYBRIDS.jsonl',result);completed+=1
                        print('HYBRID',completed,'/720',cut,u['id'],q['query_index'],hybrid,repr(result['generated_text'] if cut==14 else result['first_token_text']),flush=True)
        for pair in shams:
            a,b=index[pair['a']],index[pair['b']]
            row={**a,'edit':dict(donor_payload=native[b['id']]['base_payload'],cut=14,region='prefix')}
            result=measure(row,'sham',full=True)
            result.update(a=a['id'],b=b['id'],family=a['family'],rotation=a['rotation'],form=a['form'],query_index=a['query_index'],edit=row['edit'])
            append(HERE/'SHAM_RESULTS.jsonl',result);sham_count+=1
            print('SHAM',sham_count,'/72',a['id'],repr(result['generated_text']),flush=True)
        assert completed==720 and sham_count==72
        save(HERE/'COLLECTION.json',dict(status='COMPLETE',native_responses=360,hybrid_measurements=720,hybrid_full_generations=432,sham_full_generations=72,actual_model_forwards=engine.calls,workers=engine.worker,qwen_forwards=0,paid_api_calls=0))
    except BaseException:
        save(HERE/'ERROR.json',dict(traceback=traceback.format_exc(),native_responses=len(native),hybrid_measurements=completed,sham_responses=sham_count,actual_model_forwards=engine.calls,no_automatic_retry=True));raise
    finally:engine.close()
    from analyze import main as analyze
    analyze()

if __name__=='__main__':main()
