"""Bounded reference-only Transformer comparison; no trained-student reproduction."""
import ast,csv,hashlib,itertools,json,random,time,warnings
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from reference import Reference
P=Path(__file__).resolve().parent
UPSTREAM=P.parent/'transformer_comparison_feasibility_2026-10-18/upstream/encoding.py'
DEV_SEEDS=[101,103,107,109];HELD_SEEDS=[113,127,131,137]
DEV_DOSES=[0.,.25,.75,1.];HELD_DOSES=[0.,.125,.375,.625,.875,1.]


def dump(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def inputs():
    allseq=list(itertools.product(range(4),repeat=4));rng=np.random.default_rng(4819)
    good=[s for s in allseq if len(set(s))==4];bad=[s for s in allseq if len(set(s))<4]
    rng.shuffle(good);rng.shuffle(bad)
    dev=good[:12]+bad[:36];held=good[12:24]+bad[36:72]
    rng.shuffle(dev);rng.shuffle(held)
    return allseq,dev,held


class Implementation:
    """Native Transformer plus an independent output-irrelevant Transformer branch.
    Both branches use attention, residual addition and pointwise ReLU MLPs.
    The resulting residual coordinates are permuted, preserving the native circuit.
    """
    def __init__(self,core,seed):
        self.core,self.seed=core,seed;g=np.random.default_rng(seed);self.aux=8
        tensor=lambda shape,scale:torch.tensor(g.normal(size=shape)*scale,dtype=torch.float64)
        self.emb=tensor((core.n+1,8),.5);self.pos=tensor((core.n+1,8),.2)
        self.weights=[]
        for k in range(2*core.config['layers']):
            if k%2==0:self.weights.append([tensor((8,8),.15) for _ in range(4)])
            else:self.weights.append([tensor((8,16),.15),tensor((16,),.03),tensor((16,8),.15),tensor((8,),.03)])
        self.perm=torch.tensor(g.permutation(core.width+8));self.inverse=torch.argsort(self.perm)

    def encode(self,c,a):return torch.cat([c,a],dim=-1)[:,:,self.perm]
    def decode(self,h):
        z=h[:,:,self.inverse];return z[:,:,:self.core.width],z[:,:,self.core.width:]

    def auxiliary_op(self,a,k):
        w=self.weights[k]
        if k%2==0:
            q,kk,v=a@w[0],a@w[1],a@w[2]
            return a+torch.softmax(q@kk.transpose(-1,-2)/np.sqrt(8),dim=-1)@v@w[3]
        return a+torch.relu(a@w[0]+w[1])@w[2]+w[3]

    def trace(self,seqs):
        c=self.core.encode(seqs);ids=torch.tensor([[self.core.n]+list(x) for x in seqs]);a=self.emb[ids]+self.pos[torch.arange(self.core.n+1)]
        h=self.encode(c,a);out=[h]
        for k in range(len(self.weights)):
            c=self.core.op(c,k);a=self.auxiliary_op(a,k);out.append(self.encode(c,a))
        return out

    def resume(self,h,stage):
        c,a=self.decode(h)
        for k in range(stage,len(self.weights)):
            c=self.core.op(c,k);a=self.auxiliary_op(a,k)
        return self.core.margin(c)

    def clamp(self,h,dose):
        c,a=self.decode(h);target=self.core.clamp_tokens_zero(c)
        return self.encode(c+dose*(target-c),a)

    def state_arrays(self):
        d={'aux_emb':self.emb.numpy(),'aux_pos':self.pos.numpy(),'permutation':self.perm.numpy()}
        for k,w in enumerate(self.weights):
            for j,v in enumerate(w):d[f'aux_op{k}_w{j}']=v.numpy()
        return d


def dtw(cost):
    n,m=cost.shape;v=np.full((n,m),np.inf);parent={}
    for i in range(n):
        for j in range(m):
            if i==j==0:v[i,j]=cost[i,j];continue
            choices=[]
            for di,dj,p in [(1,1,0.),(1,0,.01),(0,1,.01)]:
                if i>=di and j>=dj:choices.append((v[i-di,j-dj]+p,(i-di,j-dj)))
            val,prev=min(choices,key=lambda q:q[0]);v[i,j]=cost[i,j]+val;parent[i,j]=prev
    path=[(n-1,m-1)]
    while path[-1]!=(0,0):path.append(parent[path[-1]])
    return list(reversed(path))


def published_functions():
    s=UPSTREAM.read_text();b=s.encode();gitsha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    assert gitsha=='abd3db7367317c51974a265026efb1649a4ec76a',gitsha
    module=ast.parse(s);nodes=[a for a in module.body if isinstance(a,ast.FunctionDef) and a.name in ['encoder_fit','point_test']]
    env={'RidgeCV':RidgeCV,'train_test_split':train_test_split,'np':np,'random':random}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(UPSTREAM),'exec'),env)
    # Exact released function bodies. Wrapper logs scores without changing RNG/calls.
    original=env['encoder_fit'];log=[]
    def tracked(x,y):
        score=float(original(x,y));log.append({'source_id':id(x),'target_id':id(y),'r2':score});return score
    env['encoder_fit']=tracked
    return env['point_test'],log


@torch.no_grad()
def execute():
    assert (P/'FREEZE_RECEIPT.json').exists();D=P/'results';D.mkdir(exist_ok=False);start=time.time()
    allseq,dev,held=inputs();dump(D/'INPUTS.json',{'N':4,'dev':dev,'held':held,'normal_gate_all':allseq,'split_seed':4819})
    models={kind:Reference(kind,4) for kind in ['sorting','counting']};raw={};weights={};gates=[];meta={}
    for kind,m in models.items():
        meta[kind]={'width':m.width,'config':m.config,'summary_cut':m.cut,'labels':m.labels,'token_columns':m.token_columns,'zero_column':m.zero_column,'positive':m.positive,'negative':m.negative,'token_encoding':{str(a):b for a,b in m.compiled.input_encoder.encoding_map.items()}}
        for key,obj in m.params.items():
            for name,value in obj.items():weights[kind+'__'+key+'__'+name]=value.numpy()
        native=m.trace(allseq);y=m.margin(native[-1]);truth=torch.tensor([1 if len(set(a))==4 else -1 for a in allseq])
        error=max(m.compare_upstream(s) for s in [[0,1,2,3],[0,0,2,3]])
        gates.append({'program':kind,'all_normal_correct':int((y*truth>0).sum()),'n':len(allseq),'upstream_layer_max_error':error,'all_normal_margin_deviation':float((y-truth).abs().max())})
    dump(D/'MODEL_METADATA.json',meta);dump(D/'COMPETENCE.json',gates)
    if any(g['all_normal_correct']!=g['n'] or g['upstream_layer_max_error']>1e-8 for g in gates):
        dump(D/'SUMMARY.json',{'status':'NO-GO: reference competence/adapter gate','gates':gates});return
    reprs={};responses={};controls=[]
    for kind,m in models.items():
        for split,seqs,doses,seeds in [('dev',dev,DEV_DOSES,DEV_SEEDS),('held',held,HELD_DOSES,HELD_SEEDS)]:
            # Source101 also measured on held inputs as the prediction reference.
            eval_seeds=seeds if split=='dev' else HELD_SEEDS+[101]
            native=m.trace(seqs);clean=m.margin(native[-1]).numpy()
            direct=np.stack([np.stack([m.resume(h+q*(m.clamp_tokens_zero(h)-h),k).numpy() for q in doses],axis=1) for k,h in enumerate(native)],axis=1)
            raw[f'{kind}__{split}__native_logits']=direct;raw[f'{kind}__{split}__native_clean']=clean
            reprs[kind,split]=[]
            for seed in eval_seeds:
                impl=Implementation(m,seed);states=impl.trace(seqs)
                baseline=impl.resume(states[-1],len(states)-1).numpy();error=float(np.max(abs(baseline-clean)))
                logits=np.stack([np.stack([impl.resume(impl.clamp(h,dose),k).numpy() for dose in doses],axis=1) for k,h in enumerate(states)],axis=1)
                response=(baseline[:,None,None]-logits)/2
                key=f'{kind}__{split}__s{seed}'
                raw[key+'__logits']=logits;raw[key+'__effects']=response
                raw[key+'__states']=np.stack([h.numpy() for h in states],axis=1)
                responses[kind,split,seed]=response
                if seed in seeds:reprs[kind,split].append(np.concatenate([h.mean(1).numpy() for h in states],axis=1))
                controls.append({'program':kind,'split':split,'seed':seed,'normal_max_error':error,'native_intervention_max_error':float(np.max(abs(logits-direct))),'sham_max_error':float(np.max(abs(logits[:,:,0]-baseline[:,None])))})
                for name,value in impl.state_arrays().items():weights[f'{kind}__s{seed}__'+name]=value
            raw[f'{kind}__{split}__representations']=np.stack(reprs[kind,split])
            # Independent semantic diagnostic, all true permutations, frozen static cut.
            take=np.array([len(set(a))==4 for a in seqs]);clamped=direct[take,m.cut,-1]
            raw[f'{kind}__{split}__summary_clamp_margins']=clamped
    dump(D/'CONTROLS.json',controls)
    np.savez_compressed(D/'weights.npz',**weights);np.savez_compressed(D/'raw_outputs.npz',**raw)
    if any(max(c['normal_max_error'],c['native_intervention_max_error'],c['sham_max_error'])>1e-8 for c in controls):
        dump(D/'SUMMARY.json',{'status':'NO-GO: implementation sampler control','controls':controls});return
    semantic={kind:{split:bool(np.all(raw[f'{kind}__{split}__summary_clamp_margins']*(-1 if kind=='sorting' else 1)>.5)) for split in ['dev','held']} for kind in models}
    if not all(all(d.values()) for d in semantic.values()):
        dump(D/'SUMMARY.json',{'status':'NO-GO: declared semantic predictions failed','semantic':semantic});return
    pairs=[]
    for a,b in itertools.product(models,repeat=2):
        x=responses[a,'dev',101].mean(0)[:,1:];y=responses[b,'dev',101].mean(0)[:,1:]
        cost=np.mean((x[:,None]-y[None,:])**2,axis=-1);path=dtw(cost)
        distance=float(np.sqrt(np.mean([cost[i,j] for i,j in path])))
        src=responses[a,'held',101][:,:,1:5]
        for seed in HELD_SEEDS:
            dst=responses[b,'held',seed][:,:,1:5]
            pred=np.stack([src[:,[i for i,jj in path if jj==j]].mean(1) for j in range(dst.shape[1])],axis=1)
            raw[f'prediction_{a}_{b}_s{seed}']=pred
            pairs.append({'source':a,'target':b,'target_seed':seed,'same_algorithm':a==b,'development_distance':distance,'path':path,'held_individual_rmse':float(np.sqrt(np.mean((pred-dst)**2)))})
    dump(D/'PROFILE_COMPARISON.json',pairs)
    positives=[a['development_distance'] for a in pairs if a['same_algorithm']];negatives=[a['development_distance'] for a in pairs if not a['same_algorithm']]
    maxpos,minneg=max(positives),min(negatives);separates=maxpos+1e-7<minneg;threshold=(maxpos+minneg)/2
    # Released code's practical classifier, not a claim to reproduce theoretical G.1.
    point_test,log=published_functions();baseline=[];identities={id(v):f'{kind}/{split}/impl{i}' for (kind,split),values in reprs.items() for i,v in enumerate(values)}
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter('always')
        for split in ['dev','held']:
            for ai,a in enumerate(models):
                for bi,b in enumerate(models):
                    for strap in range(4):
                        seed=7301+1000*(split=='held')+100*ai+10*bi+strap
                        np.random.seed(seed);random.seed(seed);old=len(log)
                        reject=bool(point_test(reprs[a,split],reprs[b,split],crit=1.96,n=20))
                        for item in log[old:]:
                            item['source_id']=identities[item['source_id']];item['target_id']=identities[item['target_id']]
                        baseline.append({'split':split,'source':a,'target':b,'strap':strap,'seed':seed,'reject_same':reject,'correct':reject==(a!=b),'ridge_log_start':old,'ridge_log_end':len(log)})
                        print('baseline',split,a,b,strap,'reject',reject,flush=True)
    dump(D/'BASELINE_WARNINGS.json',[{'type':type(w.message).__name__,'message':str(w.message)} for w in warning_records]);dump(D/'CONGRUITY_CODE_BASELINE.json',baseline);dump(D/'RIDGE_CALLS.json',log)
    np.savez_compressed(D/'raw_outputs.npz',**raw)
    summary={'status':'NARROW: reference-only benchmark','normal_gate':gates,'sampler_controls_all_pass':True,'semantic_predictions':semantic,
       'profile':{'calibrates':separates,'max_positive':maxpos,'min_negative':minneg,'threshold':threshold,'held_classification_correct':sum((p['development_distance']<=threshold)==p['same_algorithm'] for p in pairs) if separates else None,'held_pairs':len(pairs),'same_algorithm_held_rmse_max':max(p['held_individual_rmse'] for p in pairs if p['same_algorithm'])},
       'released_code_held_correct':sum(p['correct'] for p in baseline if p['split']=='held'),'released_code_held_n':16,
       'baseline_scope':'Unchanged encoder_fit and point_test from released encoding.py; restricted custom nuisance-bank sampler, N4 compiled references, four straps. NOT published trained-model-bank reproduction or formal Algorithm1/G.1 theorem implementation.',
       'elapsed_seconds':time.time()-start}
    dump(D/'SUMMARY.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':execute()
