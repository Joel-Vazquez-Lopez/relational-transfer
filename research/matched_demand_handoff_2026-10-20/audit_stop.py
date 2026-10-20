"""Independent stopped-study audit; no model import, retry or source mutation."""
import csv
import re
from core import *

def read(name):return [json.loads(l) for l in (HERE/name).read_text().splitlines()]

def main():
    checks=0
    def check(ok, label):
        nonlocal checks
        assert ok,label
        checks+=1
    lock=json.loads((HERE/'LOCK.json').read_text())
    for name,h in lock['files'].items():check(sha(HERE/name)==h,('Frozen file',name))
    rows=json.loads((HERE/'INPUTS.json').read_text());index={r['id']:r for r in rows}
    native=read('NATIVE.jsonl');forwards=read('FORWARDS.jsonl');budget=read('BUDGET.jsonl');controls=read('CONTROLS.jsonl')
    callmap={r['call']:r for r in forwards}
    for rec in forwards:check(sha(HERE/rec['payload'])==rec['payload_sha256'],('Raw hash',rec['call']))
    def raw(call):
        path=HERE/'payloads'/f'CALL_{call:04d}.npz'
        with np.load(path,allow_pickle=False) as z:return {k:z[k].astype(np.float32) for k in z.files}
    prefix_hashes={};ledger=[]
    for result in native:
        row=index[result['id']];sentence=row['source']
        if row['form']=='active':
            m=re.fullmatch(r'(\w+) sent (\w+) to (\w+)\.',sentence);roles=[m[1],m[2],m[3]];mentions=roles
        else:
            m=re.fullmatch(r'To (\w+), (\w+) sent (\w+)\.',sentence);roles=[m[2],m[3],m[1]];mentions=[m[1],m[2],m[3]]
        gold=roles[row['query_index']] if row['demand']=='role' else mentions[row['query_index']]
        check(gold==row['gold']==result['gold'],('Independent source annotation',row['id']))
        base=raw(result['base_call'])
        check(base['states'].shape==(5,len(row['token_ids']),3072),('States shape',row['id']))
        check(all(np.isfinite(a).all() for a in base.values()),('Finite native arrays',row['id']))
        for n,t in row['candidate_ids'].items():check(float(base['logits'][t])==result['name_logits'][n],('Candidate score',row['id'],n))
        calls={tuple(z['prefix']):z['call'] for z in result['calls']}
        generated=result['generated_tokens']
        for i,t in enumerate(generated):check(int(raw(calls[tuple(generated[:i])])['logits'].argmax())==t,('Unconstrained greedy token',row['id'],i))
        check(result['terminated'] and int(raw(calls[tuple(generated)])['logits'].argmax())==result['eos_id'] and result['eos_id'] in row['eos_ids'],('EOS',row['id']))
        # All observed answers are exactly one complete source-name token.
        actual=next((n for n,t in row['candidate_ids'].items() if generated==[t]),None)
        check(actual==result['actual'] and result['generated_text'].strip()==actual,('Whole-name output',row['id']))
        check(result['native_correct']==(actual==gold),('Native correctness',row['id']))
        check(result['first_token_correct']==(base['logits'].argmax()==row['candidate_ids'][gold]),('First-token correctness',row['id']))
        key=(row['family'],row['form'],tuple(row['assignment']))
        hashes=[array_sha(a[:row['period']+1]) for a in base['states']]
        check(hashes==result['prefix_hashes'],('Source capture hash',row['id']))
        if key in prefix_hashes:check(hashes==prefix_hashes[key],('Future-question causality',row['id']))
        else:prefix_hashes[key]=hashes
        ledger.append({**{k:result[k] for k in ['id','family','form','source','question','demand','query_index','gold','actual','generated_text','native_correct','first_token_correct','base_call']},'observed_gold_role':ROLES[roles.index(actual)]})
    for c in controls:check(c['passed'],('Production control',c['check']))
    check(all(b['edit'] is None for b in budget),'No intervention requested')
    check(not (HERE/'HYBRIDS.jsonl').exists(),'No hybrid results')
    failures=sum(not r['first_token_correct'] for r in native)
    check(144-failures<.95*144,'Frozen prerequisite impossible even with perfect remainder')
    check(len(forwards)==188 and len(budget)==189,'Stopped call accounting')
    orphan=raw(189);b=budget[-1];row=index[b['id']]
    check(b['call']==189 and b['kind']=='native_continuation','Final dispatched continuation')
    parent=next(z for z in forwards if z['id']==row['id'] and z['kind']=='native')
    parentdata=raw(parent['call'])
    error=float(np.max(np.abs(orphan['original_logits']-parentdata['logits'])))
    check(error<=.05,'Recovered original-prefix equivalence')
    eos=int(orphan['logits'].argmax());check(eos in row['eos_ids'],'Recovered EOS')
    recovered_name=next(n for n,t in row['candidate_ids'].items() if b['prefix']==[t])
    check(int(parentdata['logits'].argmax())==b['prefix'][0],'Recovered preceding greedy token')
    recovery=dict(call=189,id=row['id'],payload='payloads/CALL_0189.npz',payload_sha256=sha(HERE/'payloads/CALL_0189.npz'),original_prompt_full_vocab_error=error,eos_id=eos,generated_tokens=b['prefix'],actual=recovered_name,gold=row['gold'],native_correct=recovered_name==row['gold'],status='Worker completed and saved raw NPZ while SIGINT interrupted controller read. Not appended to the original forward/native journals. Metrics unavailable.',parent_call=parent['call'])
    save(HERE/'RECOVERED_FORWARD.json',recovery)
    breakdown=[]
    for d in ['role','ordinal']:
        for q in range(3):
            z=[r for r in native if r['demand']==d and r['query_index']==q]
            breakdown.append(dict(demand=d,query_index=q,total=len(z),correct=sum(r['native_correct'] for r in z)))
    complete_sources=[]
    for key in sorted({(r['family'],r['form'],tuple(r['assignment'])) for r in native}):
        z=[r for r in native if (r['family'],r['form'],tuple(r['assignment']))==key]
        complete_sources.append(dict(family=key[0],form=key[1],assignment=list(key[2]),observed_questions=len(z),fully_observed=len(z)==6,complete_correct=len(z)==6 and all(r['native_correct'] for r in z)))
    summary=dict(decision='INCONCLUSIVE_COMPETENCE',schedule='EARLY_PREREQUISITE_STOP',checks=checks,checks_passed=True,frozen_files_unchanged=True,native_completed=89,native_correct=74,first_token_failures=15,maximum_possible_correct_out_of_144=129,first_token_required_at_least=137,full_response_required_at_least=130,maximum_possible_correct_fraction=129/144,primary_contrasts=None,interventions=0,controller_journaled_forwards=188,completed_raw_payloads=189,recovered_unjournaled_forwards=1,recovered_complete_response_separate=recovery,workers=max(z['worker'] for z in budget),max_mlx_peak_bytes=max(z['metrics']['peak_mlx_bytes'] for z in forwards),max_swap_bytes=max(z['swap_used_bytes'] for z in budget),min_free_disk_bytes=min(z['free_disk_bytes'] for z in budget),production_controls=len(controls),production_controls_failed=sum(not z['passed'] for z in controls),by_question=breakdown,source_cases=complete_sources,failures=[x for x in ledger if not x['native_correct']],limitation='Incomplete fixed-order native panel, not an unbiased accuracy estimate or a primary causal result. Early scheduling stop documented after outcomes; no frozen code, threshold, source, prompt, score, site or hypothesis changed. No automatic retry.')
    recovered_row={**row,**recovery}
    observed=native+[recovered_row]
    observed_source_keys={(r['family'],r['form'],tuple(r['assignment'])) for r in observed}
    summary['all_observed_including_separate_recovery']=dict(answers=len(observed),correct=sum(r['native_correct'] for r in observed),unexecuted_prompts=144-len(observed),fully_observed_source_cases=sum(sum((r['family'],r['form'],tuple(r['assignment']))==key for r in observed)==6 for key in observed_source_keys),complete_correct_source_cases=sum(all(r['native_correct'] for r in observed if (r['family'],r['form'],tuple(r['assignment']))==key) for key in observed_source_keys),by_demand=[dict(demand=d,total=sum(r['demand']==d for r in observed),correct=sum(r['native_correct'] for r in observed if r['demand']==d)) for d in ['role','ordinal']])
    save(HERE/'STOP_RESULTS.json',summary)
    with (HERE/'NATIVE_LEDGER.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    print(json.dumps({k:summary[k] for k in ['decision','checks','native_completed','native_correct','controller_journaled_forwards','completed_raw_payloads','production_controls','by_question']},indent=2))

if __name__=='__main__':main()
