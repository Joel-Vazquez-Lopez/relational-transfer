"""Independent audit of saved arrays and paired endpoints; no model execution."""
from pathlib import Path
import collections,csv,hashlib,json
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
def main():
    checks=0
    def check(ok,message):
        nonlocal checks;assert bool(ok),message;checks+=1
    lock=json.loads((HERE/'FROZEN_LOCK.json').read_text())
    for name,h in lock['frozen_files'].items():check(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h,'Frozen bytes: '+name)
    runs=[];errors=[];initials=collections.defaultdict(set);positive_banks={};positive_rows={};base_banks={};rng_states={}
    for path in sorted((HERE/'factorial_runs').rglob('RESULT.json')):
        run=path.parent;r=json.loads(path.read_text());meta=json.loads((run/'INPUT_METADATA.json').read_text());seed=r['seed'];initials[(r['phase'],seed)].add(r['initial_state_sha256'])
        check(r['initial_state_sha256']==meta['initial_state_sha256'],'Initial metadata')
        checkpoint=torch.load(run/'checkpoint_00000.pt',map_location='cpu',weights_only=True)
        digest=hashlib.sha256()
        for name,value in checkpoint['model'].items():digest.update(name.encode());digest.update(value.numpy().tobytes())
        check(digest.hexdigest()==r['initial_state_sha256'],'Actual initial checkpoint tensor hash')
        for step in lock['checkpoints']:
            ck=torch.load(run/f'checkpoint_{step:05}.pt',map_location='cpu',weights_only=True)
            rr=json.dumps(ck['numpy_rng'],sort_keys=True)
            key=(r['phase'],seed,step)
            if key in rng_states:check(rr==rng_states[key],'Exact paired minibatch RNG state')
            else:rng_states[key]=rr
        with np.load(run/'TRAIN_INPUTS.npz',allow_pickle=False) as tr:
            rows=tr['rows'];ids=tr['messages'];base=tr['base'];key=(r['phase'],seed)
            gold_indices=rows[np.arange(len(rows)),2+rows[:,1]]
            signature=hashlib.sha256(rows[:,:2].tobytes()+gold_indices.tobytes()).hexdigest()
            check(len(rows)==290304,'Training row count')
            check(np.array_equal(gold_indices%48,rows[:,0]),'Positive scene belongs to query')
            qsig=hashlib.sha256(ids.tobytes()).hexdigest();bsig=hashlib.sha256(base.tobytes()).hexdigest()
            if key in positive_rows:
                check(signature==positive_rows[key],'Batchwise positive query/label/scene coupling')
                check(qsig==positive_banks[key] and bsig==base_banks[key],'Shared exact query/scene banks')
            else:positive_rows[key]=signature;positive_banks[key]=qsig;base_banks[key]=bsig
        with np.load(run/'INPUTS.npz',allow_pickle=False) as inp:
            labels=inp['held_labels'];queries=inp['held_queries'];fam_labels=inp['familiar_labels'];records=meta['held_records'];imap=meta['identity_map'];inverse={v:i for i,v in enumerate(imap)}
            check(len(labels)==3072 and len(records)==3072,'Held sample coverage')
            check(set(queries[:,3])=={8},'Dedicated PAD')
            check(np.ptp(np.bincount(labels,minlength=4))==0,'Held answer balance')
            for i,rec in enumerate(records):
                canonical=(inverse[int(queries[i,0])],inverse[int(queries[i,1])],int(queries[i,2])-6)
                check(tuple(rec['meaning'])==canonical,'Held query semantic join')
                check(canonical[0] in (0,1,2) and canonical[1] in (0,1,2) and canonical[0]!=canonical[1],'No observed relation in primary held panel')
            for metric in r['metrics']:
                step=metric['step']
                with np.load(run/f'scores_{step:05}.npz',allow_pickle=False) as sc:
                    scores=sc['held'];fp=sc['familiar'];correct=scores.argmax(1)==labels
                    check(np.isfinite(scores).all() and np.isfinite(fp).all(),'Finite score arrays')
                    check(abs(np.mean(fp.argmax(1)==fam_labels)-metric['familiar_accuracy'])<1e-12,'Familiar replay')
                    grouped=collections.defaultdict(list)
                    for i,rec in enumerate(records):grouped[(tuple(rec['group']),rec['role'])].append(i)
                    for role in ['source','target']:
                        pairs=[ix for (g,rr),ix in grouped.items() if rr==role]
                        check(all(len(ix)==2 for ix in pairs),'Two directions per pair')
                        success=sum(bool(np.all(correct[ix])) for ix in pairs)
                        check(success==metric['held'][role]['paired_correct'],'Independent paired replay')
                        check(sum(int(correct[ix].sum()) for ix in pairs)==metric['held'][role]['correct_answers'],'Independent individual replay')
                    if step==lock['updates']:
                        for i in np.flatnonzero(~correct):
                            rec=records[int(i)];event,g,d,rotation=rec['group'];ms=((1,0,event),(2,0,event),(0,1,event),(0,2,event));order=np.roll(np.arange(4),rotation);pred=int(scores[i].argmax());predm=ms[int(order[pred])]
                            query=tuple(rec['meaning']);pr='source' if predm[1]==0 else 'target';kind='wrong_focal_identity' if pr==rec['role'] else 'wrong_argument_role'
                            errors.append(dict(phase=r['phase'],seed=seed,arm=r['arm'],index=int(i),role=rec['role'],query=str(query),prediction=str(predm),gold_position=int(labels[i]),predicted_position=pred,error_kind=kind))
        runs.append(dict(phase=r['phase'],seed=seed,arm=r['arm']))
    for k,values in initials.items():check(len(values)==1,'Paired initial parameters')
    with (HERE/'WRONG_ANSWERS.csv').open('w',newline='') as f:
        fields=['phase','seed','arm','index','role','query','prediction','gold_position','predicted_position','error_kind'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(errors)
    result=dict(status='PASS_SAVED_ARRAY_AUDIT',checks=checks,runs_audited=len(runs),wrong_final_answers=len(errors),errors_by_kind=dict(collections.Counter(r['error_kind'] for r in errors)),errors_by_arm=dict(collections.Counter(r['arm'] for r in errors)),scope='Independent array/metadata/endpoints arithmetic; not an independent model replication.',model_forwards=0)
    (HERE/'INDEPENDENT_FACTORIAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
