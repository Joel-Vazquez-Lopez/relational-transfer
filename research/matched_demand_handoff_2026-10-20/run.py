"""One fixed native screen, technical controls and factorial; no outcome adaptation."""
import time
import traceback
from transformers import AutoTokenizer
from core import *
from runtime import Engine, resources

def main():
    lock=json.loads((HERE/'LOCK.json').read_text())
    for n,h in lock['files'].items():assert sha(HERE/n)==h,('Frozen source',n)
    for n,h in lock['native_sources'].items():assert sha(ROOT/n)==h
    for n,v in lock['model']['files'].items():assert sha(ROOT/n)==v['sha256']
    assert (HERE/'FREEZE_RECEIPT.json').exists()
    assert not (HERE/'STARTED.json').exists(),'No automatic restart or overwrite'
    resources();save(HERE/'STARTED.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),lock_sha256=sha(HERE/'LOCK.json')))
    rows=json.loads((HERE/'INPUTS.json').read_text());lookup={r['id']:r for r in rows}
    units=json.loads((HERE/'UNITS.json').read_text())
    tok=AutoTokenizer.from_pretrained(str(ROOT/lock['model']['path']),local_files_only=True)
    engine=Engine(lock);native={};completed=0

    def check(name,ok,**metadata):
        append(HERE/'CONTROLS.jsonl',dict(check=name,passed=bool(ok),**metadata));assert ok,(name,metadata)

    def get_array(path,key):
        with np.load(HERE/path,allow_pickle=False) as z:return z[key].astype(np.float32)

    def dispatch(row,prefix,capture,kind):
        data,entry=engine.dispatch(row,prefix,capture,kind)
        if row.get('edit'):
            e=row['edit'];before=get_array(native[row['id']]['base_payload'],'states')[CUTS.index(e['cut'])]
            donor=get_array(e['donor_payload'],'states')[CUTS.index(e['cut'])]
            left,right=(0,row['period']+1) if e['region']=='prefix' else (row['period']+1,len(row['token_ids'])) if e['region']=='suffix' else (0,len(row['token_ids']))
            expected=before.copy();expected[left:right]=donor[left:right]
            proof=entry['patch_proof']
            check('exact_region_interchange',proof['pre_sha256']==array_sha(before) and proof['post_sha256']==array_sha(expected) and proof['tail_pre_sha256']==proof['tail_post_sha256'],call=entry['call'],id=row['id'],edit=e)
        return data,entry

    def generate(row,kind,capture):
        data,entry=dispatch(row,(),capture,kind)
        cache={():data['logits']};calls={():entry['call']}
        def get(prefix):
            if prefix not in cache:
                z,rec=dispatch(row,prefix,False,kind+'_continuation')
                error=float(np.max(np.abs(z['original_logits']-cache[()])))
                check('continuation_original_prompt_invariance',error<=.05,id=row['id'],call=rec['call'],error=error)
                cache[prefix]=z['logits'];calls[prefix]=rec['call']
            return cache[prefix]
        result=score_generation(row,get,tok.decode)
        return {**result,'base_call':entry['call'],'base_payload':entry['payload'],'base_payload_sha256':entry['payload_sha256'],'calls':[dict(prefix=list(p),call=c) for p,c in calls.items()],'prefix_hashes':entry['prefix_hashes']}

    try:
        first=rows[0]
        plain,a=dispatch(first,(),False,'validation_unwrapped')
        wrapped,b=dispatch(first,(),True,'validation_wrapped')
        repeat,c=dispatch(first,(),True,'validation_repeat')
        error=float(np.max(np.abs(plain['logits']-wrapped['logits'])))
        check('wrapper_equivalence',error<=.05,error=error)
        check('repeat_equivalence',np.array_equal(wrapped['states'],repeat['states']) and np.max(np.abs(wrapped['logits']-repeat['logits']))<=.05)
        del plain,wrapped,repeat
        groups={}
        for row in rows:
            result=generate(row,'native',True)
            result.update({k:row[k] for k in ['id','family','form','assignment','source','question','demand','query_index','role_index','gold']})
            native[row['id']]=result;append(HERE/'NATIVE.jsonl',result)
            group=(row['family'],row['form'],tuple(row['assignment']))
            if group in groups:check('future_question_source_invariance',groups[group]==result['prefix_hashes'],id=row['id'])
            else:groups[group]=result['prefix_hashes']
            print('NATIVE',len(native),'/144',row['id'],repr(result['generated_text']),result['native_correct'],flush=True)
        save(HERE/'NATIVE_COLLECTION.json',dict(complete=True,answers=len(native),native_correct=sum(r['native_correct'] for r in native.values()),actual_model_forwards=engine.calls))
        # The intervention stage has its own predeclared 45-minute window.
        engine.started=time.monotonic()
        for u in units:
            if u['edge']!=0:continue
            for q in u['queries']:
                a,b=lookup[q['a']],lookup[q['b']]
                for cut in [0,28]:
                    for region in ['prefix','suffix']:
                        row={**a,'edit':dict(donor_payload=native[b['id']]['base_payload'],cut=cut,region=region)}
                        data,rec=dispatch(row,(),False,'endpoint_control')
                        target=b if (cut==0 and region=='prefix') or (cut==28 and region=='suffix') else a
                        expected=get_array(native[target['id']]['base_payload'],'logits')
                        err=float(np.max(np.abs(data['logits']-expected)))
                        check('forced_endpoint_full_vocab',err<=.05,call=rec['call'],target=target['id'],error=err)
                for cut in [7,14,21]:
                    for donor,label in [(a,'identity'),(b,'joint_swap')]:
                        row={**a,'edit':dict(donor_payload=native[donor['id']]['base_payload'],cut=cut,region='both')}
                        data,rec=dispatch(row,(),False,label+'_control')
                        err=float(np.max(np.abs(data['logits']-get_array(native[donor['id']]['base_payload'],'logits'))))
                        check(label+'_full_vocab',err<=.05,call=rec['call'],error=err)
        save(HERE/'TECHNICAL_GATE.json',dict(passed=True,endpoint_forwards=96,intermediate_forwards=144))
        for cut in [7,14,21]:
            for u in units:
                for q in u['queries']:
                    a,b=lookup[q['a']],lookup[q['b']]
                    for hybrid,region in [('BA','prefix'),('AB','suffix')]:
                        row={**a,'edit':dict(donor_payload=native[b['id']]['base_payload'],cut=cut,region=region)}
                        result=generate(row,'hybrid',False)
                        result.update(unit=u['id'],family=u['family'],form=u['form'],cut=cut,hybrid=hybrid,a=a['id'],b=b['id'],demand=q['demand'],query_index=q['query_index'],role_index=q['role_index'],changed=q['changed'],gold_a=a['gold'],gold_b=b['gold'],edit=row['edit'])
                        append(HERE/'HYBRIDS.jsonl',result);completed+=1
                        print('HYBRID',completed,'/432',cut,u['id'],q['demand'],q['query_index'],hybrid,repr(result['generated_text']),flush=True)
        assert completed==432
        save(HERE/'COLLECTION.json',dict(status='COMPLETE',native_answers=144,hybrid_generations=completed,actual_model_forwards=engine.calls,workers=engine.worker,qwen_forwards=0,paid_api_calls=0))
    except BaseException:
        save(HERE/'ERROR.json',dict(traceback=traceback.format_exc(),native_answers=len(native),hybrid_generations=completed,actual_model_forwards=engine.calls,no_automatic_retry=True));raise
    finally:engine.close()
    from analyze import main as analyze
    analyze()

if __name__=='__main__':main()
