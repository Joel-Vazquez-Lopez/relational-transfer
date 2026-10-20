"""Independent raw-vector, annotation, intervention and prediction audit.

No MLX imports or model calls. Supports a native-gated complete stop as well as
the complete causal panel; unfinished resource interruptions require recovery.
"""
import itertools
import re
from transformers import AutoTokenizer
from core import *

def read(name):return [json.loads(l) for l in (HERE/name).read_text().splitlines()]

def main():
    count=0
    def check(ok,label):
        nonlocal count
        assert ok,label;count+=1
    lock=json.loads((HERE/'LOCK.json').read_text())
    for name,h in lock['files'].items():check(sha(HERE/name)==h,('Frozen file',name))
    inputs={r['id']:r for r in json.loads((HERE/'INPUTS.json').read_text())}
    native={r['id']:r for r in read('NATIVE.jsonl')};forward=read('FORWARDS.jsonl');budget=read('BUDGET.jsonl')
    collection=json.loads((HERE/'COLLECTION.json').read_text())
    check(len(forward)==len(budget)==collection['actual_model_forwards'],'Complete attempt accounting')
    callmap={r['call']:r for r in forward}
    tok=AutoTokenizer.from_pretrained(str(ROOT/lock['model']['path']),local_files_only=True)
    def raw(path,key):
        with np.load(HERE/path,allow_pickle=False) as z:return z[key].astype(np.float32)
    def scorefile(call):return raw(callmap[call]['payload'],'logits')
    controls=read('CONTROLS.jsonl')
    for c in controls:check(c['passed'],('Production control',c['check']))
    for rec,b in zip(forward,budget):
        check(rec['call']==b['call'] and rec['id']==b['id'] and rec['prefix']==b['prefix'],'Forward journal correspondence')
        check(sha(HERE/rec['payload'])==rec['payload_sha256'],('Payload checksum',rec['call']))
        check(np.isfinite(scorefile(rec['call'])).all(),'Finite full vocabulary')
        if rec.get('edit'):
            row=inputs[rec['id']];edit=rec['edit'];ci=CUTS.index(edit['cut'])
            before=raw(native[rec['id']]['base_payload'],'states')[ci]
            donor=raw(edit['donor_payload'],'states')[ci]
            after=before.copy();p=row['period']+1
            if edit['region']=='prefix':after[:p]=donor[:p]
            elif edit['region']=='suffix':after[p:]=donor[p:]
            else:check(edit['region']=='both','Declared region');after[:]=donor
            proof=rec['patch_proof']
            check(proof['pre_sha256']==array_sha(before),'Native pre-interchange state')
            check(proof['post_sha256']==array_sha(after),'Entire realized interchange state')
            check(proof['tail_pre_sha256']==proof['tail_post_sha256'],'Unselected generated/pad tail')
            if rec['kind'] in ['identity_control','joint_swap_control']:
                err=float(np.max(abs(scorefile(rec['call'])-raw(edit['donor_payload'],'logits'))))
                check(err<=.05,'Full-vocabulary identity/joint equivalence')
            if rec['kind']=='endpoint_control':
                target=edit['donor_payload'] if (edit['cut']==0 and edit['region']=='prefix') or (edit['cut']==28 and edit['region']=='suffix') else native[rec['id']]['base_payload']
                check(np.max(abs(scorefile(rec['call'])-raw(target,'logits')))<=.05,'Forced endpoint full vocabulary')
        if rec['kind']=='exposed_restart_control':
            check(np.max(abs(scorefile(rec['call'])-raw('REFERENCE_CONTROL.npz','logits')))<=.05,'Restart reference')
    source_hashes={}
    for result in native.values():
        row=inputs[result['id']];event=row['event']
        if row['form']=='active':
            m=re.fullmatch(r'(\w+) sent (\w+) to (\w+)\.',event);roles=[m[1],m[2],m[3]]
        elif row['form']=='passive':
            m=re.fullmatch(r'(\w+) was sent to (\w+) by (\w+)\.',event);roles=[m[3],m[1],m[2]]
        else:
            m=re.fullmatch(r'To (\w+), (\w+) sent (\w+)\.',event);roles=[m[2],m[3],m[1]]
        check(roles[row['query_index']]==result['gold']==row['gold'],'Independent grammatical annotation')
        check(tok.encode(row['prompt'],add_special_tokens=False)==row['token_ids'],'Input token/prompt equality')
        states=raw(result['base_payload'],'states');readout=raw(result['base_payload'],'readout')
        check(states.shape==(5,len(row['token_ids']),3072) and readout.shape==(29,3072),'Native capture shapes')
        check(np.array_equal(states[:,-1],readout[CUTS]),'Saved full-depth readout equals bank anchor')
        hashes=[array_sha(a[:row['period']+1]) for a in states]
        check(hashes==result['prefix_hashes'],'Source hashes reproduce')
        key=(row['family'],row['rotation'],row['form'],row['variant'])
        if key in source_hashes:check(hashes==source_hashes[key],'Source prefix cannot depend on future query')
        else:source_hashes[key]=hashes
    measured=list(native.values())
    if collection['status']=='COMPLETE':measured+=read('HYBRIDS.jsonl')+read('SHAM_RESULTS.jsonl')
    for r in measured:
        row=inputs[r['id']];base=scorefile(r['base_call'])
        for n,t in row['candidate_ids'].items():check(float(base[t])==r['name_logits'][n],'Candidate logits reproduce')
        check(int(base.argmax())==r['first_token'],'Vocabulary-wide first token')
        if r['measurement']=='next_token_only':
            check(r['generated_text'] is None and r['terminated'] is None,'No invented full response at secondary cuts');continue
        calls={tuple(x['prefix']):x['call'] for x in r['calls']};tokens=r['generated_tokens']
        for i,t in enumerate(tokens):check(int(scorefile(calls[tuple(tokens[:i])]).argmax())==t,'Unconstrained generated token')
        check(tok.decode(tokens)==r['generated_text'],'Exact generated decoding')
        actual=next((n for n in row['names'] if r['generated_text'].strip()==n),None)
        check(actual==r['actual'],'No repaired output name')
        if r['terminated']:
            eos=int(scorefile(calls[tuple(tokens)]).argmax());check(eos==r['eos_id'] and eos in row['eos_ids'],'EOS completion')
        else:check(len(tokens)==MAX_NEW and r['eos_id'] is None,'Truncation retained')
        for p,c in calls.items():check(np.max(abs(raw(callmap[c]['payload'],'original_logits')-base))<=.05,'Continuation original-prefix invariance')
        if r['id'] in native and r is native.get(r['id']):
            check(r['native_correct']==bool(r['terminated'] and actual==row['gold']),'Native exact-name correctness')
    if collection['status']=='COMPLETE':
        # Recompute every cross-fitted predictor independently. Ridge uses an
        # augmented least-squares formulation, not the original normal solve.
        result=json.loads((HERE/'RESULTS.json').read_text());profiles=result['profiles'];original=result['primary_prediction']
        units={u['id']:u for u in json.loads((HERE/'UNITS.json').read_text())}
        hybrids={(r['unit'],r['cut'],r['query_index'],r['hybrid']):r for r in read('HYBRIDS.jsonl')}
        for point in profiles:
            u=units[point['unit']];q=u['queries'][u['role_index']]
            a,b=native[q['a']],native[q['b']];row=inputs[q['a']]
            at,bt=row['candidate_ids'][a['gold']],row['candidate_ids'][b['gold']]
            def difference(z):return float(z[bt])-float(z[at])
            aa=difference(scorefile(a['base_call']));bb=difference(scorefile(b['base_call']))
            ba=difference(scorefile(hybrids[u['id'],point['cut'],u['role_index'],'BA']['base_call']))
            ab=difference(scorefile(hybrids[u['id'],point['cut'],u['role_index'],'AB']['base_call']))
            for k,v in [('aa',aa),('bb',bb),('ba',ba),('ab',ab)]:check(abs(point[k]-v)<1e-10,'Raw causal logit contrast')
            if bb-aa>=1:check(abs(point['authority']-(ba-ab)/(bb-aa))<1e-10,'Source-authority arithmetic')
            else:check(point['authority'] is None,'Undefined normalization retained')
            eligible=all(r['native_correct'] and r['first_token_correct'] for r in [a,b]) and bb-aa>=1
            check(point['eligible']==eligible,'Native-only support eligibility')
            mentions=[m.group() for m in re.finditer(r'\b(?:'+'|'.join(row['names'])+r')\b',row['event'])]
            check(mentions.index(a['gold'])==point['position'],'Role versus source-position label')
            for key,r in [('entropy_a',a),('entropy_b',b)]:
                z=scorefile(r['base_call']);x=np.array([float(z[t]) for t in row['candidate_ids'].values()]);p=np.exp(x-x.max());p/=p.sum()
                check(abs(float(-sum(p*np.log(p+1e-300)))-point[key])<1e-10,'Candidate entropy definition')
        for fit in original['fits']:
            train=[r for r in profiles if r['eligible'] and r['family']!=fit['family'] and r['form']!=fit['form'] and r['cut']==fit['cut']]
            test=[r for r in profiles if r['eligible'] and r['family']==fit['family'] and r['form']==fit['form'] and r['cut']==fit['cut']]
            check([r['unit'] for r in train]==fit['training_units'],'No held-out train contamination')
            def feat(r):return [r['aa'],r['bb'],r['entropy_a'],r['entropy_b'],r['target_token_index'],r['period']-r['target_token_index'],r['readout_index']-r['period']]+[float(r['position']==p) for p in range(3)]
            x=np.array([feat(r) for r in train]);y=np.array([r['authority'] for r in train]);mu=x.mean(0);sd=x.std(0);sd[sd<1e-8]=1
            xx=np.column_stack([np.ones(len(x)),(x-mu)/sd]);reg=np.eye(xx.shape[1]);reg[0,0]=0
            weights=np.linalg.lstsq(np.vstack([xx,reg]),np.r_[y,np.zeros(xx.shape[1])],rcond=None)[0]
            check(np.max(abs(weights-np.array(fit['weights'])))<1e-8,'Independent ridge solution')
            for r in test:
                pred=next(v for v in original['predictions'] if v['unit']==r['unit'] and v['cut']==r['cut'])
                rp=float(np.mean([z['authority'] for z in train if z['role_index']==r['role_index']]))
                pp=float(np.mean([z['authority'] for z in train if z['position']==r['position']]))
                npred=float(np.r_[1.,(np.array(feat(r))-mu)/sd]@weights)
                expected=dict(role=rp,position=pp,generic=float(y.mean()),nuisance=npred)
                for k,v in expected.items():check(abs(v-pred['predicted'][k])<1e-8,'Held-out predicted score')
        losses={k:float(np.mean([(p['predicted'][k]-p['observed'])**2 for p in original['predictions']])) for k in ['role','position','generic','nuisance']}
        for k,v in losses.items():check(abs(v-original['pooled_mse'][k])<1e-10,'Primary pooled prediction loss')
    receipt=dict(passed=True,checks=count,actual_model_forwards=len(forward),native_responses=len(native),status=collection['status'],frozen_files_unchanged=True,production_checks=len(controls),all_production_checks_passed=True)
    save(HERE/'INDEPENDENT_AUDIT.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':main()
