"""Independent postcollection full-vector replay. No LLM calls or selection changes."""
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    checks=[]
    def check(name,ok,**metadata):
        checks.append(dict(check=name,passed=bool(ok),**metadata))
        assert ok,(name,metadata)
    lock=json.loads((HERE/'LOCK.json').read_text())
    for f,h in lock['files'].items():check('frozen_file',digest(HERE/f)==h,file=f)
    inputs=json.loads((HERE/'INPUTS.json').read_text());lookup={r['id']:r for r in inputs}
    tok=AutoTokenizer.from_pretrained(str(ROOT/lock['model']['path']),local_files_only=True,trust_remote_code=False)
    answers=[json.loads(s) for s in (HERE/'ANSWERS.jsonl').read_text().splitlines()]
    forwards=[json.loads(s) for s in (HERE/'FORWARDS.jsonl').read_text().splitlines()]
    budget=[json.loads(s) for s in (HERE/'BUDGET.jsonl').read_text().splitlines()]
    check('complete108',len(answers)==108 and len({a['id'] for a in answers})==108)
    check('all_attempts_completed',len(forwards)==len(budget)<=lock['max_model_forwards'])
    check('contiguous_calls',[f['call'] for f in forwards]==list(range(1,len(forwards)+1)))
    flookup={f['call']:f for f in forwards};workers=defaultdict(list)
    for f,b in zip(forwards,budget):
        check('journal',all(f[k]==v for k,v in b.items()),call=f['call'])
        check('payload_hash',digest(HERE/f['payload'])==f['payload_sha256'],call=f['call'])
        with np.load(HERE/f['payload'],allow_pickle=False) as a:
            check('finite_full_vectors',a['logits'].shape==a['original_logits'].shape==(128256,) and all(np.isfinite(a[k]).all() for k in a.files),call=f['call'])
        workers[f['worker']].append(f)
    for w,group in workers.items():
        check('worker_bound',len(group)<=32 and group[0]['kind']=='exposed_restart_control',worker=w)
        with np.load(HERE/group[0]['payload'],allow_pickle=False) as a,np.load(HERE/'REFERENCE_CONTROL.npz',allow_pickle=False) as b:
            error=float(np.max(np.abs(a['logits'].astype(np.float32)-b['logits'].astype(np.float32))))
        check('restart_reference',error<=.05,worker=w,error=error)
    source_states={}
    for answer in answers:
        row=lookup[answer['id']]
        mapping={tuple(x['prefix']):x['call'] for x in answer['logit_prefix_calls']}
        arrays={}
        for prefix,call in mapping.items():
            f=flookup[call]
            check('prefix_call_identity',f['id']==row['id'] and f['prefix']==list(prefix),call=call)
            with np.load(HERE/f['payload'],allow_pickle=False) as z:
                arrays[prefix]=z['logits'].astype(np.float64)
                if not prefix:trace=z['trace'].astype(np.float32)
        base=arrays[()]
        for prefix,call in mapping.items():
            with np.load(HERE/flookup[call]['payload'],allow_pickle=False) as z:
                err=float(np.max(np.abs(z['original_logits'].astype(np.float64)-base)))
            check('continuation_invariance',err<=.05,call=call,error=err)
        group=(row['family'],tuple(row['assignment']),row['form'])
        if group in source_states:
            error=float(np.max(np.abs(trace[:,:4]-source_states[group])))
            check('saved_source_anchor_invariance',error<=1e-6,id=row['id'],error=error)
        else:source_states[group]=trace[:,:4].copy()
        probabilities={}
        for name,paths in row['candidate_sequences'].items():
            values=[]
            for path in paths:
                terms=[]
                for j,t in enumerate(path):
                    z=arrays[tuple(path[:j])];peak=float(z.max())
                    terms.append(float(z[t])-peak-math.log(float(np.exp(z-peak).sum())))
                values.append(sum(terms))
            peak=max(values);probabilities[name]=peak+math.log(sum(math.exp(v-peak) for v in values))
            error=abs(probabilities[name]-answer['candidate_log_probabilities'][name])
            check('full_name_probability_replay',error<1e-8,id=row['id'],candidate=name,error=error)
        generated=[];eos=None
        for _ in range(12):
            t=int(arrays[tuple(generated)].argmax())
            if t in row['eos_ids']:eos=t;break
            generated.append(t)
        text=tok.decode(generated)
        actual=next((n for n in row['names'] if text.strip()==n),None)
        correct=eos is not None and actual==row['gold']
        margin=probabilities[row['gold']]-max(v for n,v in probabilities.items() if n!=row['gold'])
        check('native_generation_replay',generated==answer['generated_tokens'] and text==answer['generated_text'] and eos==answer['eos_id'] and correct==answer['native_correct'],id=row['id'])
        check('margin_replay',abs(margin-answer['full_name_margin'])<1e-8 and (margin>0)==answer['score_correct'],id=row['id'])
    pairs=json.loads((HERE/'PAIR_LEDGER.json').read_text());alook={r['id']:r for r in answers}
    check('162_pairs',len(pairs)==162)
    for pair in pairs:
        group=[alook[i] for i in pair['answer_ids']]
        check('pair_six_answer_eligibility',len(group)==6 and sum(x['native_correct'] for x in group)==pair['native_correct'] and all(x['native_correct'] for x in group)==pair['eligible'],family=pair['family'])
    result=json.loads((HERE/'RESULTS.json').read_text())
    check('all_native_answers_counted',result['native_correct']==sum(r['native_correct'] for r in answers))
    check('held_out_absent',all(f['kind']=='exposed_restart_control' or f['id'] in lookup for f in forwards) and result['held_out_answers']==0)
    check('no_interventions',result['scientific_interventions']==0 and all(f['kind'] in ['exposed_restart_control','validation_unwrapped','validation_wrapper','validation_repeat','scientific_baseline','scientific_continuation'] for f in forwards))
    output=dict(passed=True,n_checks=len(checks),checks=checks,actual_model_forwards=len(forwards),workers=len(workers),answers=108,scope='Independent saved-vector score/generation, source-anchor, control and six-answer eligibility replay; no new model forwards.')
    (HERE/'INDEPENDENT_AUDIT.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='checks'},indent=2))

if __name__=='__main__':main()
