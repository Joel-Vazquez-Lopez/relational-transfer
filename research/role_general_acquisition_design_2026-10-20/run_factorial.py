"""Execute the frozen, bounded role-demand experiment. CPU only; no downloads."""
from pathlib import Path
import hashlib,json,time,resource,shutil,sys,traceback
import numpy as np
import torch
import torch.nn.functional as F
import factorial_data as data
from legacy_model import Agents

HERE=Path(__file__).resolve().parent
RUNS=HERE/'factorial_runs'
COUNTS=dict(training_updates=0,training_examples=0,evaluation_batches=0,evaluation_examples=0,trace_batches=0,trace_examples=0)
START=None;LOCK=None
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify():
    lock=json.loads((HERE/'FROZEN_LOCK.json').read_text())
    receipt=json.loads((HERE/'GITHUB_FREEZE.json').read_text())
    assert receipt['all_selected_local_bytes_match_remote'] and receipt['frozen_lock_sha256']==sha(HERE/'FROZEN_LOCK.json')
    assert lock['preflight_status']=='PASS_PRETRAINING_DESIGN'
    for p,h in lock['frozen_files'].items():assert sha(HERE/p)==h,('Frozen code changed',p)
    return lock
def guard():
    assert shutil.disk_usage(HERE).free>=LOCK['resources']['disk_reserve_bytes'],'DISK_RESERVE'
    assert time.time()-START<LOCK['resources']['wall_seconds'],'WALL_CAP'
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform!='darwin':rss*=1024
    assert rss<=LOCK['resources']['peak_rss_bytes'],'MEMORY_CAP'
    used=sum(p.stat().st_size for p in RUNS.rglob('*') if p.is_file()) if RUNS.exists() else 0
    assert used<=LOCK['resources']['artifact_bytes'],'ARTIFACT_CAP'

def messages(ids):return F.one_hot(ids,data.CFG['vocab_size']).float()
def state_hash(model):
    h=hashlib.sha256()
    for n,p in model.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def capture(model,ids,candidates):
    msg=messages(ids);batch=len(msg)
    q=torch.cat([model.mcls.expand(batch,-1,-1),msg@model.sym],1)+model.mpos
    qs=[q.detach().numpy().copy()]
    for block in model.msg_blocks:q=block(q);qs.append(q.detach().numpy().copy())
    cv=candidates.reshape(-1,3,14)
    c=torch.cat([model.candidate.cls.expand(len(cv),-1,-1),model.candidate.inp(cv)],1)+model.candidate.pos
    cs=[c.detach().numpy().reshape(batch,4,4,-1).copy()]
    for block in model.candidate.blocks:c=block(c);cs.append(c.detach().numpy().reshape(batch,4,4,-1).copy())
    q=F.normalize(model.msg_norm(q[:,0]),dim=-1)
    c=F.normalize(model.candidate.norm(c[:,0]),dim=-1).reshape(batch,4,-1)
    scores=torch.einsum('bd,bkd->bk',q,c)*10
    COUNTS['trace_batches']+=1;COUNTS['trace_examples']+=batch
    return dict(query_states=np.stack(qs),candidate_states=np.stack(cs),query_final=q.numpy(),candidate_final=c.numpy(),scores=scores.numpy())

@torch.no_grad()
def score(model,ids,candidates):
    out=[];model.eval()
    for st in range(0,len(ids),128):
        out.append(model.listen(messages(ids[st:st+128]),candidates[st:st+128]).numpy())
        COUNTS['evaluation_batches']+=1;COUNTS['evaluation_examples']+=len(ids[st:st+128])
    return np.concatenate(out)

def required_masks(ds):
    ms=np.asarray(data.meanings());rows=ds['rows'];q=ms[rows[:,0]];c=ms[rows[:,2:]%len(ms)]
    source=np.isin(q[:,0],data.FOCAL)&np.any((c[:,:,0]==(3-q[:,0,None]))&(c[:,:,1]==q[:,1,None])&(c[:,:,2]==q[:,2,None]),axis=1)
    target=np.isin(q[:,1],data.FOCAL)&np.any((c[:,:,1]==(3-q[:,1,None]))&(c[:,:,0]==q[:,0,None])&(c[:,:,2]==q[:,2,None]),axis=1)
    return source,target

def fresh_panel(arm,imap,seed):
    ds=data.build_training(arm,imap,seed,n_geometry=6,split='fresh_training')
    s,t=required_masks(ds);regular=np.linspace(0,len(s)-1,1536,dtype=int)
    take=np.unique(np.r_[regular,np.flatnonzero(s|t)])
    rows=ds['rows'][take]
    return dict(candidates=ds['base'][rows[:,2:]],queries=ds['messages'][rows[:,0]],labels=rows[:,1].astype(np.int64),source_required=s[take],target_required=t[take])

def summarize_held(scores,ds):
    pred=scores.argmax(1);correct=pred==ds['labels'];groups={}
    for i,r in enumerate(ds['records']):groups.setdefault((tuple(r['group']),r['role']),[]).append(i)
    out={}
    for role in ['source','target']:
        pairs=[ix for (g,rr),ix in groups.items() if rr==role]
        assert all(len(ix)==2 for ix in pairs)
        out[role]=dict(correct_answers=int(sum(correct[ix].sum() for ix in pairs)),answers=2*len(pairs),paired_correct=int(sum(correct[ix].all() for ix in pairs)),pairs=len(pairs),paired_rate=float(np.mean([correct[ix].all() for ix in pairs])))
    complete={}
    for i,r in enumerate(ds['records']):complete.setdefault(tuple(r['group']),[]).append(i)
    out['complete_four_rate']=float(np.mean([correct[ix].all() for ix in complete.values()]))
    return out

def tensor_inputs(ds):return torch.from_numpy(ds['queries'].astype(np.int64)),torch.from_numpy(ds['candidates'].astype(np.float32))

def technical_gate():
    torch.manual_seed(LOCK['technical_seed']);net=Agents(data.CFG).eval()
    panel=data.familiar(tuple(range(6)),LOCK['technical_seed'],n_geometry=1)
    ids,c=tensor_inputs(panel);ids=ids[:32];c=c[:32]
    a=score(net,ids,c);b=capture(net,ids,c)['scores'];identity=score(net,ids.clone(),c.clone())
    perm=np.asarray([2,0,3,1]);pp=score(net,ids,c[:,perm]);back=pp[:,np.argsort(perm)]
    checks=dict(trace_max_error=float(np.max(np.abs(a-b))),identity_max_error=float(np.max(np.abs(a-identity))),candidate_permutation_max_error=float(np.max(np.abs(a-back))),candidate_argmax_agreement=bool(np.array_equal(a.argmax(1),back.argmax(1))))
    dump(HERE/'TECHNICAL_GATE.json',checks)
    assert max(checks['trace_max_error'],checks['identity_max_error'],checks['candidate_permutation_max_error'])<=1e-5 and checks['candidate_argmax_agreement']
    del net

def train_one(phase,seed,imap,arm,reference_hash):
    guard();verify();run=RUNS/phase/f'seed_{seed}'/arm
    assert not run.exists(),('Refuse overwrite',str(run));run.mkdir(parents=True)
    torch.manual_seed(seed);net=Agents(data.CFG)
    for name,p in net.named_parameters():
        if name.startswith(('speaker.','emit.')):p.requires_grad_(False)
    initial=state_hash(net)
    if reference_hash is not None:assert initial==reference_hash,'INITIALIZATION_MISMATCH'
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.0008,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    rng=np.random.default_rng(data.stable_seed(seed,'batch'))
    ds=data.build_training(arm,imap,seed)
    np.savez_compressed(run/'TRAIN_INPUTS.npz',**ds)
    rows=torch.from_numpy(ds['rows'].astype(np.int64));base=torch.from_numpy(ds['base']);qbank=torch.from_numpy(ds['messages'])
    held=data.evaluation(imap,seed,n_geometry=24);fam=data.familiar(imap,seed,n_geometry=6)
    hid,hc=tensor_inputs(held);fid,fc=tensor_inputs(fam)
    fresh=fresh_panel(arm,imap,seed);nid,nc=tensor_inputs(fresh)
    np.savez_compressed(run/'INPUTS.npz',held_candidates=held['candidates'],held_queries=held['queries'],held_labels=held['labels'],familiar_candidates=fam['candidates'],familiar_queries=fam['queries'],familiar_labels=fam['labels'],fresh_candidates=fresh['candidates'],fresh_queries=fresh['queries'],fresh_labels=fresh['labels'],fresh_source_required=fresh['source_required'],fresh_target_required=fresh['target_required'])
    dump(run/'INPUT_METADATA.json',dict(seed=seed,identity_map=imap,arm=arm,held_records=held['records'],initial_state_sha256=initial,train_rows=len(rows),train_base_scenes=len(base)))
    trace_ix=np.asarray([i for i,r in enumerate(held['records']) if r['group'][1:]==(0,0,0)],dtype=int)
    assert len(trace_ix)==8
    metrics=[];losses=[];started=time.time()
    for step in range(LOCK['updates']+1):
        if step in LOCK['checkpoints']:
            net.eval();hs=score(net,hid,hc);fs=score(net,fid,fc)
            tr=capture(net,hid[trace_ix],hc[trace_ix])
            np.savez_compressed(run/f'scores_{step:05}.npz',held=hs,familiar=fs)
            np.savez_compressed(run/f'traces_{step:05}.npz',indices=trace_ix,**tr)
            metric=dict(step=step,held=summarize_held(hs,held),familiar_accuracy=float(np.mean(fs.argmax(1)==fam['labels'])))
            if step==LOCK['updates']:
                ns=score(net,nid,nc);correct=ns.argmax(1)==fresh['labels']
                np.savez_compressed(run/'fresh_final_scores.npz',scores=ns)
                metric['fresh_accuracy']=float(np.mean(correct))
                metric['required_accuracy']={role:float(np.mean(correct[fresh[role+'_required']])) if np.any(fresh[role+'_required']) else None for role in ['source','target']}
                metric['required_n']={role:int(fresh[role+'_required'].sum()) for role in ['source','target']}
                ident=score(net,hid[:64].clone(),hc[:64].clone())
                metric['identity_max_error']=float(np.max(np.abs(ident-hs[:64])))
                reversed_objects=score(net,fid[:128],fc[:128,:,torch.tensor([2,1,0]),:])
                metric['object_permutation_agreement']=float(np.mean(reversed_objects.argmax(1)==fs[:128].argmax(1)))
            metrics.append(metric)
            torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,step=step,seed=seed,arm=arm,config=data.CFG),run/f'checkpoint_{step:05}.pt')
            guard()
        if step==LOCK['updates']:break
        net.train();ix=torch.from_numpy(rng.integers(0,len(rows),LOCK['batch_size']).astype(np.int64));batch=rows[ix]
        qs=qbank[batch[:,0]];c=base[batch[:,2:]]
        logits=net.listen(messages(qs),c);loss=F.cross_entropy(logits,batch[:,1])
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1);opt.step()
        COUNTS['training_updates']+=1;COUNTS['training_examples']+=len(batch)
        if step%100==0:losses.append(dict(update=step+1,loss=float(loss.detach()),batch_accuracy=float((logits.argmax(1)==batch[:,1]).float().mean())))
        if step%500==0:guard()
    result=dict(status='COMPLETE_FROZEN_RUN',phase=phase,seed=seed,identity_map=imap,arm=arm,initial_state_sha256=initial,seconds=time.time()-started,metrics=metrics,training_trace=losses)
    dump(run/'RESULT.json',result)
    dump(HERE/'COMPUTE_ACCOUNTING.json',dict(**COUNTS,elapsed_seconds=time.time()-START,model_runs_completed=len(list(RUNS.rglob('RESULT.json')))))
    # Progress does not expose held-out metrics; the fixed rule controls continuation.
    print('RUN_COMPLETE',phase,seed,arm,'seconds',round(result['seconds'],1),flush=True)
    del net,opt,rows,base,ds,hid,hc,fid,fc,nid,nc
    return result,initial

def development_gate(results):
    final={r['arm']:r['metrics'][-1] for r in results}
    checks=dict(familiar_all_pass=all(r['familiar_accuracy']>=.95 for r in final.values()),required_comparisons_learned=all(v is None or v>=.95 for r in final.values() for v in r['required_accuracy'].values()),identity_all_pass=all(r['identity_max_error']<=1e-5 for r in final.values()))
    avg=lambda r:(r['held']['source']['paired_rate']+r['held']['target']['paired_rate'])/2
    baseline=avg(final['neither']);checks['baseline_headroom']=baseline<=.80
    checks['some_treatment_effect']=max(avg(final[a])-baseline for a in data.ARMS if a!='neither')>=.10
    out=dict(status='PASS_CONTINUE_CONFIRMATION' if all(checks.values()) else 'STOP_DEVELOPMENT_GATE',checks=checks,neither_paired_average=baseline,all_development_final=final)
    dump(HERE/'DEVELOPMENT_GATE.json',out);return out

def main():
    global START,LOCK
    START=time.time();LOCK=verify();assert not RUNS.exists(),'Refuse reuse/overwrite existing study runs'
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    RUNS.mkdir();guard();technical_gate()
    allruns=[]
    for phase,seeds in [('development',[LOCK['development_seed']]),('confirmation',LOCK['confirmation_seeds'])]:
        for seed in seeds:
            imap=np.random.default_rng(data.stable_seed(seed,'identity-map')).permutation(6).tolist()
            initial=None;results=[]
            # A predeclared Latin arm order reduces a systematic time/order confound.
            offset=seed%len(data.ARMS);arms=data.ARMS[offset:]+data.ARMS[:offset]
            for arm in arms:
                result,initial=train_one(phase,seed,imap,arm,initial);results.append(result);allruns.append(result)
            if phase=='development':
                gate=development_gate(results);print('DEVELOPMENT_DECISION',gate['status'],flush=True)
                if gate['status']!='PASS_CONTINUE_CONFIRMATION':
                    dump(HERE/'COLLECTION.json',dict(status=gate['status'],runs_completed=len(allruns),confirmation_runs=0,counts=COUNTS,elapsed_seconds=time.time()-START));return
    dump(HERE/'COLLECTION.json',dict(status='COMPLETE_FROZEN_COLLECTION',runs_completed=len(allruns),confirmation_runs=48,counts=COUNTS,elapsed_seconds=time.time()-START))
    print('COLLECTION_COMPLETE',len(allruns),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump(HERE/'INTERRUPTION.json',dict(status='STOP_RESOURCE_OR_TECHNICAL',error=repr(exc),traceback=traceback.format_exc(),counts=COUNTS,elapsed_seconds=time.time()-START if START else None))
        raise
