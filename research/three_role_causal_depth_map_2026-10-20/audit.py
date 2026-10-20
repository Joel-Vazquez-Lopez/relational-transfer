"""Independent postcollection replay using raw saved vectors. No model inference."""
import hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
PARENT=ROOT/'research/three_role_native_development_2026-10-20'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arr(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].astype(np.float64) for k in z.files}
def context(r):return '|'.join([r['family'],''.join(map(str,r['assignment'])),r['form']])

def main():
    checks=[]
    def check(name,ok,**extra):
        checks.append(dict(check=name,passed=bool(ok),**extra));assert ok,(name,extra)
    lock=json.loads((HERE/'LOCK.json').read_text())
    for f,h in lock['files'].items():check('frozen_file',digest(HERE/f)==h,file=f)
    inputs={r['id']:r for r in json.loads((HERE/'INPUTS.json').read_text())};jobs={j['id']:j for j in json.loads((HERE/'JOBS.json').read_text())}
    answers=[json.loads(s) for s in (HERE/'ANSWERS.jsonl').read_text().splitlines()]
    forwards=[json.loads(s) for s in (HERE/'FORWARDS.jsonl').read_text().splitlines()]
    budget=[json.loads(s) for s in (HERE/'BUDGET.jsonl').read_text().splitlines()]
    native={r['id']:r for r in map(json.loads,(PARENT/'ANSWERS.jsonl').read_text().splitlines())}
    oldcalls={r['call']:r for r in map(json.loads,(PARENT/'FORWARDS.jsonl').read_text().splitlines())}
    check('complete432',len(answers)==len(jobs)==432 and {a['id'] for a in answers}==set(jobs))
    check('all_attempts_completed',len(forwards)==len(budget)<=lock['max_model_forwards'])
    check('contiguous_calls',[f['call'] for f in forwards]==list(range(1,len(forwards)+1)))
    flookup={f['call']:f for f in forwards};workers=defaultdict(list)
    for f,b in zip(forwards,budget):
        check('journal',all(f[k]==v for k,v in b.items()),call=f['call'])
        check('payload_hash',digest(HERE/f['payload'])==f['payload_sha256'],call=f['call'])
        a=arr(HERE/f['payload'])
        check('finite_full_vectors',a['logits'].shape==a['original_logits'].shape==(152064,) and all(np.isfinite(v).all() for v in a.values()),call=f['call'])
        check('resources',f['free_disk_bytes']>8*1024**3 and f['swap_used_bytes']<4*1024**3 and f['swap_used_bytes']-f['initial_swap_bytes']<1024**3 and f['memory_pressure_level']==1 and f['metrics']['active_mlx_bytes']<6*1024**3,call=f['call'])
        workers[f['worker']].append(f)
    for w,g in workers.items():
        check('worker_bound',len(g)<=32 and g[0]['kind']=='exposed_restart_control',worker=w)
        a,b=arr(HERE/g[0]['payload']),arr(HERE/'REFERENCE_CONTROL.npz')
        check('restart_reference',np.max(np.abs(a['logits']-b['logits']))<=.05,worker=w)
    caches={r['context']:r for r in map(json.loads,(HERE/'DONOR_CACHE.jsonl').read_text().splitlines())}
    check('36_source_caches',len(caches)==36)
    shared=[0,1,2,6,7]
    for c in caches.values():
        old=native[c['input_id']];a,b=arr(HERE/c['payload']),arr(PARENT/old['base_payload'])
        check('cache_native_replay',a['trace'].shape==(29,8,3584) and np.max(np.abs(a['logits']-b['logits']))<=.05 and a['logits'].argmax()==b['logits'].argmax() and np.max(np.abs(a['trace'][:,shared]-b['trace']))<=1e-6 and c['prefix_hashes']==oldcalls[old['base_call']]['prefix_hashes'],id=c['input_id'])
    identities=[f for f in forwards if f['kind']=='identity_patch']
    check('all_identity_inputs',{(f['id'],f['edit']['block']) for f in identities}=={(r['input_id'],r['block']) for r in json.loads((HERE/'IDENTITY_INPUTS.json').read_text())} and len(identities)==90)
    for f in identities:
        old=native[f['id']];a,b=arr(HERE/f['payload']),arr(PARENT/old['base_payload'])
        check('identity_native_replay',np.max(np.abs(a['logits']-b['logits']))<=.05 and a['logits'].argmax()==b['logits'].argmax() and np.max(np.abs(a['trace'][:,shared]-b['trace']))<=1e-6 and f['prefix_hashes']==oldcalls[old['base_call']]['prefix_hashes'],id=f['id'])
    tok=AutoTokenizer.from_pretrained(str(ROOT/lock['model']['path']),local_files_only=True,trust_remote_code=False)
    role=['sender','sent_person','destination_person'];order={'active':[0,1,2],'passive':[1,2,0],'goal_fronted':[2,0,1]};source_states={}
    for answer in answers:
        j=jobs[answer['id']];block=j['block'];row=inputs[j['recipient_id']];donor=inputs[j['donor_input_id']]
        q=role.index(row['query_role']);rolepred=donor['names'][donor['assignment'][q]]
        mention=order[donor['form']][order[row['form']].index(q)]
        pred=dict(role_table=rolepred,mention_order=donor['names'][donor['assignment'][mention]],fixed_one_answer=donor['names'][donor['assignment'][role.index(j['invariant_role'])]],syntax_aware_parser=rolepred,complete_future_answer_table=rolepred)
        check('independent_symbolic_predictions',pred==j['predictions']==answer['predictions'],id=j['id'])
        check('native_admission_record',native[row['id']]['native_correct'] and native[donor['id']]['native_correct']==answer['donor_native_correct'],id=j['id'])
        mapping={tuple(x['prefix']):x['call'] for x in answer['logit_prefix_calls']};arrays={}
        for p,call in mapping.items():
            f=flookup[call];check('prefix_call_identity',f['id']==j['id'] and f['prefix']==list(p),call=call)
            a=arr(HERE/f['payload']);arrays[p]=a['logits']
        base=arr(HERE/answer['base_payload']);old=native[row['id']];prior=arr(PARENT/old['base_payload']);don=arr(HERE/caches[j['donor_context']]['payload'])
        f=flookup[answer['base_call']]
        check('before_cut_replay',np.max(np.abs(base['trace'][:block,shared]-prior['trace'][:block]))<=1e-6 and f['prefix_hashes'][:block-1]==oldcalls[old['base_call']]['prefix_hashes'][:block-1],id=j['id'])
        s=[3,4,5] if j['condition']=='neighbor' else [0,1,2]
        check('exact_patch_replay',np.max(np.abs(base['trace'][block,s]-don['trace'][block,s]))<=1e-6,id=j['id'])
        # At the outgoing patched boundary, unselected captured native sites remain untouched.
        unchanged=list(range(5)) if j['condition']=='neighbor' else [3,4]
        check('unselected_boundary_sites',np.max(np.abs(base['trace'][block,[shared[i] for i in unchanged]]-prior['trace'][block,unchanged]))<=1e-6,id=j['id'])
        group=(block,j['condition'],context(row),j['donor_context'])
        if group in source_states:
            states,hashes=source_states[group]
            check('future_query_source_invariance',np.max(np.abs(base['trace'][:,:7]-states))<=1e-6 and hashes==f['prefix_hashes'],id=j['id'])
        else:source_states[group]=(base['trace'][:,:7].copy(),f['prefix_hashes'])
        for p,call in mapping.items():
            a=arr(HERE/flookup[call]['payload']);check('continuation_patch_replay',np.max(np.abs(a['original_logits']-arrays[()]))<=.05,call=call)
        probabilities={}
        for name,paths in row['candidate_sequences'].items():
            vals=[]
            for path in paths:
                terms=[]
                for i,t in enumerate(path):
                    z=arrays[tuple(path[:i])];peak=float(z.max());terms.append(float(z[t])-peak-math.log(float(np.exp(z-peak).sum())))
                vals.append(sum(terms))
            peak=max(vals);probabilities[name]=peak+math.log(sum(math.exp(v-peak) for v in vals))
            check('probability_replay',abs(probabilities[name]-answer['candidate_log_probabilities'][name])<1e-8,id=j['id'],candidate=name)
        generated=[];eos=None
        for _ in range(12):
            t=int(arrays[tuple(generated)].argmax())
            if t in row['eos_ids']:eos=t;break
            generated.append(t)
        text=tok.decode(generated);correct=eos is not None and text.strip()==row['gold']
        check('native_generation_replay',generated==answer['generated_tokens'] and text==answer['generated_text'] and eos==answer['eos_id'] and correct==answer['native_correct'],id=j['id'])
        matches={h:bool(eos is not None and text.strip()==v) for h,v in pred.items()}
        check('all_account_matches_replay',matches==answer['hypothesis_matches'],id=j['id'])
        margin=probabilities[row['gold']]-max(v for n,v in probabilities.items() if n!=row['gold'])
        check('margin_replay',abs(margin-answer['full_name_margin'])<1e-8 and (margin>0)==answer['score_correct'],id=j['id'])
    result=json.loads((HERE/'RESULTS.json').read_text())
    qualified=[]
    for g in result['depth_gates']:
        endpoint=[a for a in answers if a['block']==g['block'] and a['condition'] in ['main','same_role_voice']]
        exact={h:len(endpoint)==72 and all(a['hypothesis_matches'][h] for a in endpoint) for h in pred}
        check('joint_account_gate',g['joint_72_answer_accounts']==exact,block=g['block'])
        neighbors=sum(u['complete_account']['role_table'] for u in result['units'] if u['block']==g['block'] and u['condition']=='neighbor')
        q=bool(result['technical_pass'] and exact['role_table'] and neighbors<=2)
        check('candidate_gate_replay',g['role_candidate_qualified']==q,block=g['block'])
        if q:qualified.append(g['block'])
    check('earliest_depth_selection',result['qualifying_blocks']==qualified and result['selected_block']==(min(qualified) if qualified else None))
    for u in result['units']:
        rows=[a for a in answers if (a['block'],a['unit'],a['condition'])==(u['block'],u['unit'],u['condition'])]
        check('unit_complete_replay',len(rows)==6 and u['complete_account']=={h:all(a['hypothesis_matches'][h] for a in rows) for h in pred},block=u['block'],unit=u['unit'],condition=u['condition'])
    check('heldout_absent',all(f['kind']=='exposed_restart_control' or f['id'] in inputs or f['id'] in jobs for f in forwards) and all(r['family'] in ['t01','t02'] for r in inputs.values()))
    out=dict(passed=True,n_checks=len(checks),checks=checks,actual_model_forwards=len(forwards),answers=432,workers=len(workers),scope='Independent full-vector scores/generations, donor/identity/patch/source invariance, literal predictions and unit gates; no model inference.')
    (HERE/'INDEPENDENT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='checks'},indent=2))
if __name__=='__main__':main()
