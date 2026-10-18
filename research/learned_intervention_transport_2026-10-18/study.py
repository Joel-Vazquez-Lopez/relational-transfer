"""Single frozen CPU study. Explicit grammar-feature MLPs, not text encoders/LLMs."""
import argparse, hashlib, itertools, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
SEEDS=[11,23,37]
DEV_DOSES=[0.,.25,.5,1.]
HELD_DOSES=[0.,.125,.375,.625,.875,1.]
TRAIN_STEPS=1800
WIDTH=16
DTYPE=torch.float64
torch.set_num_threads(2)
NAMES={'train':['Alice','Bob','Clara','David','Emma','Frank','Grace','Henry'],
       'dev':['Iris','Jack','Kira','Leo','Maya','Noah','Olivia','Peter'],
       'held':['Quinn','Rosa','Sam','Tara','Uma','Victor','Wendy','Yusuf']}


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def name_feature(name):
    return (int(hashlib.sha256(name.encode()).hexdigest()[:8],16)/0xffffffff-.5)*.5


def dataset(split):
    rows=[]; count=8 if split=='train' else 4
    for r,v,o,q,l in itertools.product([-1.,1.],repeat=5):
        f=r*v*o
        for context in range(count):
            names=NAMES[split]; a=names[context%len(names)]; b=names[(context+3)%len(names)]
            n1,n2=name_feature(a),name_feature(b)
            t=(-1. if context%2 else 1.)*(.5 if split=='held' else 1.)
            agent,patient=(a,b) if r>0 else (b,a)
            verb='thanked' if split=='held' else 'praised'
            if v>0 and f>0: sentence=f'{agent} {verb} {patient}.'
            elif v>0: sentence=f'{patient} was the person whom {agent} {verb}.'
            elif f>0: sentence=f'{patient} was {verb} by {agent}.'
            else: sentence=f'By {agent}, {patient} was {verb}.'
            if split=='held': sentence=('At noon, '+sentence[0].lower()+sentence[1:]) if t>0 else sentence[:-1]+' at noon.'
            elif t>0: sentence='Today, '+sentence[0].lower()+sentence[1:]
            else: sentence=sentence[:-1]+' yesterday.'
            # Names are proper nouns: retain their capitalization at all positions.
            for name in [a,b]: sentence=sentence.replace(name.lower(),name)
            options=[a,b] if l>0 else [b,a]
            answer=agent if q>0 else patient
            rows.append({'id':f'{split}_{len(rows):04d}','sentence':sentence,
                'question':'Who performed the action?' if q>0 else 'Who received the action?',
                'options':options,'answer':answer,'agent':agent,'patient':patient,
                'features':[o,v,f,n1,n2,t],'R':r,'Q':q,'L':l,'Y':r*q*l,
                'annotation':'Deterministic AI-authored template grammar; no human annotation or external LLM call.'})
    X=torch.tensor([r['features'] for r in rows],dtype=DTYPE)
    R=torch.tensor([r['R'] for r in rows],dtype=DTYPE)
    return rows,X,R


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.ModuleList([nn.Linear(6,WIDTH),nn.Linear(WIDTH,WIDTH),nn.Linear(WIDTH,WIDTH)])
        self.head=nn.Sequential(nn.Linear(WIDTH,WIDTH),nn.Tanh(),nn.Linear(WIDTH,1))
        self.bypass=nn.Sequential(nn.Linear(3,WIDTH),nn.Tanh(),nn.Linear(WIDTH,WIDTH),nn.Tanh(),nn.Linear(WIDTH,1))
        self.double()

    def states(self,x):
        states=[]
        for layer in self.enc:
            x=torch.tanh(layer(x));states.append(x)
        return states

    def answer(self,x,route='cached'):
        return self.head(self.states(x)[-1]).squeeze(-1) if route=='cached' else self.bypass(x[:,:3]).squeeze(-1)

    def resume(self,h,k,x,route):
        if route=='bypass': return self.bypass(x[:,:3]).squeeze(-1)
        for j in range(k+1,3): h=torch.tanh(self.enc[j](h))
        return self.head(h).squeeze(-1)


class Coordinates:
    def __init__(self,kind,seed,stage):
        self.kind=kind
        rng=np.random.default_rng(seed*100+stage)
        self.Q=torch.tensor(np.linalg.qr(rng.normal(size=(WIDTH,WIDTH)))[0],dtype=DTYPE)

    def forward(self,h):
        if self.kind=='identity': return h
        if self.kind=='nonlinear': h=torch.sinh(2*h)/2
        return h@self.Q

    def inverse(self,u):
        if self.kind=='identity': return u
        h=u@self.Q.T
        return torch.asinh(2*h)/2 if self.kind=='nonlinear' else h


def contrast(h,r): return h[r>0].mean(0)-h[r<0].mean(0)


def train(model,x,r,route):
    params=list(model.enc.parameters())+list(model.head.parameters()) if route=='cached' else list(model.bypass.parameters())
    optim=torch.optim.Adam(params,lr=.01,weight_decay=1e-5)
    y=(r+1)/2; trace=[]
    for step in range(TRAIN_STEPS):
        optim.zero_grad(); logits=model.answer(x,route)
        loss=nn.functional.binary_cross_entropy_with_logits(logits,y)
        loss.backward();optim.step()
        if step%100==0 or step==TRAIN_STEPS-1:
            trace.append({'step':step+1,'loss':float(loss.detach()),'correct':int(((logits.detach()>0)==(r>0)).sum())})
    return trace


def preflight():
    torch.manual_seed(9001); model=Model(); rng=torch.Generator().manual_seed(9002)
    x=torch.randn((7,6),generator=rng,dtype=DTYPE)*.2
    errors=[]
    with torch.no_grad():
        for k,h in enumerate(model.states(x)):
            delta=torch.randn(h.shape,generator=rng,dtype=DTYPE)*.1
            for kind in ['identity','orthogonal','nonlinear']:
                t=Coordinates(kind,9001,k)
                roundtrip=float((t.inverse(t.forward(h))-h).abs().max())
                native=model.resume(h+delta,k,x,'cached')
                transported=model.resume(t.inverse(t.forward(h+delta)),k,x,'cached')
                err=float((native-transported).abs().max());assert max(roundtrip,err)<1e-10
                errors.append({'stage':k,'coordinates':kind,'roundtrip_max':roundtrip,'transport_output_max':err})
    dump(ROOT/'PREFLIGHT.json',{'untrained_infrastructure_only':True,'checks':errors})
    print('Untrained transport preflight passed',flush=True)


def cka(h,g):
    h=h-h.mean(0);g=g-g.mean(0)
    den=torch.linalg.norm(h.T@h)*torch.linalg.norm(g.T@g)
    if float(den)<=1e-20:return 0.
    return float(torch.clamp(torch.linalg.norm(h.T@g)**2/den,0.,1.))


def dtw(cost):
    n,m=cost.shape;d=np.full((n,m),np.inf);parents={}
    for i in range(n):
        for j in range(m):
            if i==j==0:d[i,j]=cost[i,j];continue
            opts=[]
            for di,dj,p in [(1,1,0.),(1,0,.01),(0,1,.01)]:
                if i>=di and j>=dj:opts.append((d[i-di,j-dj]+p,(i-di,j-dj)))
            val,prev=min(opts,key=lambda z:z[0]);d[i,j]=cost[i,j]+val;parents[i,j]=prev
    path=[(n-1,m-1)]
    while path[-1]!=(0,0):path.append(parents[path[-1]])
    return list(reversed(path))


@torch.no_grad()
def evaluate(model,x,r,route,xd,rd,doses):
    states=model.states(x);devstates=model.states(xd)
    base=model.answer(x,route)
    cf=x.clone();cf[:,0]*=-1
    opposite=model.answer(cf,route)
    den=r*(base-opposite)
    # A gate checked separately ensures each denominator is positive and well defined.
    assert bool((den>1e-8).all())
    raw={};stats=[]
    for kind in ['identity','orthogonal','nonlinear']:
        outputs=[];directions=[];nat=[];rt=[]
        for k,h in enumerate(states):
            t=Coordinates(kind,model.seed,k)
            u=t.forward(h);hd=devstates[k]
            d=contrast(hd,rd);du=contrast(t.forward(hd),rd)
            directions.append(du.numpy());nat.append(u.numpy())
            rt.append(float((model.resume(t.inverse(u),k,x,route)-base).abs().max()))
            stage=[]
            for dose in doses:
                native_h=h-dose*r[:,None]*d
                oracle_u=t.forward(native_h)
                blind_u=u-dose*r[:,None]*du
                a=model.resume(native_h,k,x,route)
                b=model.resume(t.inverse(oracle_u),k,x,route)
                c=model.resume(t.inverse(blind_u),k,x,route)
                # All unnormalized scalar logits are saved, not only ratios.
                stage.append(torch.stack([a,b,c],dim=-1).numpy())
            outputs.append(np.stack(stage,axis=1))
        arr=np.stack(outputs,axis=1) # case,stage,dose,[native,oracle,blind]
        eff=r.numpy()[:,None,None,None]*(base.numpy()[:,None,None,None]-arr)/den.numpy()[:,None,None,None]
        raw[kind+'__logits']=arr;raw[kind+'__effects']=eff
        raw[kind+'__directions']=np.stack(directions);raw[kind+'__natural']=np.stack(nat,axis=1)
        stats.append({'coordinates':kind,'native_oracle_max':float(np.max(abs(arr[:,:,:,0]-arr[:,:,:,1]))),
            'native_blind_effect_rmse':float(np.sqrt(np.mean((eff[:,:,:,0]-eff[:,:,:,2])**2))),
            'native_blind_effect_max':float(np.max(abs(eff[:,:,:,0]-eff[:,:,:,2]))),
            'normal_roundtrip_output_max':max(rt),'sham_logit_max':float(np.max(abs(arr[:,:,0,:]-base.numpy()[:,None,None]))),
            'direction_norms':[float(np.linalg.norm(a)) for a in directions]})
    raw['clean_role_logits']=base.numpy();raw['counterfactual_role_logits']=opposite.numpy();raw['normalizer']=den.numpy()
    # Natural coherent donor controls: reverse role, or preserve role while changing voice/order.
    same=x.clone();same[:,0]*=-1;same[:,1]*=-1
    donors=[model.states(cf),model.states(same)]
    control=np.stack([np.stack([model.resume(d[k],k,x,route).numpy() for k in range(3)],axis=1) for d in donors],axis=-1)
    raw['donor_control_role_logits']=control
    return raw,stats


def execute():
    assert (ROOT/'FREEZE_RECEIPT.json').exists(),'Freeze before any training or main outcomes'
    out=ROOT/'results';out.mkdir(exist_ok=False)
    started=time.time();datasets={s:dataset(s) for s in ['train','dev','held']}
    for s,(rows,x,r) in datasets.items():dump(out/('cases_'+s+'.json'),rows)
    models={};logs={};checkpoints={};gates=[]
    for seed in SEEDS:
        torch.manual_seed(seed);model=Model();model.seed=seed
        logs[str(seed)]={}
        for route in ['cached','bypass']:
            logs[str(seed)][route]=train(model,datasets['train'][1],datasets['train'][2],route)
            print(f'seed={seed} route={route} training complete',flush=True)
        models[seed]=model
        for name,value in model.state_dict().items():checkpoints[f'seed{seed}__{name}']=value.detach().numpy()
        with torch.no_grad():
            for route in ['cached','bypass']:
                for split,(_,x,r) in datasets.items():
                    z=model.answer(x,route);cf=x.clone();cf[:,0]*=-1;zz=model.answer(cf,route)
                    gates.append({'seed':seed,'route':route,'split':split,'n':len(x),
                        'correct':int((z*r>0).sum()),'counterfactual_correct':int((zz*r<0).sum()),
                        'min_signed_margin':float((z*r).min()),'min_normalizer':float((r*(z-zz)).min())})
    np.savez_compressed(out/'checkpoints.npz',**checkpoints);dump(out/'TRAINING.json',logs);dump(out/'COMPETENCE.json',gates)
    # Mandatory competence screen, before all causal intervention outcomes.
    failures=[g for g in gates if g['correct']!=g['n'] or g['counterfactual_correct']!=g['n'] or g['min_normalizer']<=1e-8]
    if failures:
        dump(out/'SUMMARY.json',{'decision':'NO-GO: competence gate','failed_gates':failures,'elapsed_seconds':time.time()-started,'interventions_run':False});print('Stopped: competence gate',flush=True);return
    raw={};metrics=[]
    for seed,model in models.items():
        for route in ['cached','bypass']:
            for split,doses in [('dev',DEV_DOSES),('held',HELD_DOSES)]:
                rows,x,r=datasets[split];_,xd,rd=datasets['dev']
                rr,ss=evaluate(model,x,r,route,xd,rd,doses)
                key=f's{seed}__{route}__{split}__'
                for name,value in rr.items():raw[key+name]=value
                for s in ss:metrics.append({'seed':seed,'route':route,'split':split,**s})
    np.savez_compressed(out/'raw_outputs.npz',**raw);dump(out/'TRANSPORT.json',metrics)
    # Fixed mapping by depth, normal-state CKA, or development intervention responses.
    pairs=[];heldrows=datasets['held'][0];patient=np.array([a['Q']<0 for a in heldrows])
    agent=np.array([a['Q']>0 for a in datasets['dev'][0]])
    for source in SEEDS:
        for target in SEEDS:
            for route in ['cached','bypass']:
                src=raw[f's{source}__cached__dev__identity__effects'][agent,:,:,0].mean(0)[:,1:]
                dst=raw[f's{target}__{route}__dev__identity__effects'][agent,:,:,0].mean(0)[:,1:]
                hs=raw[f's{source}__cached__dev__identity__natural'];ht=raw[f's{target}__{route}__dev__identity__natural']
                causal_cost=np.mean((src[:,None,:]-dst[None,:,:])**2,axis=-1)
                for method in ['depth','cka','causal']:
                    if method=='depth':cost=(np.arange(3)[:,None]-np.arange(3)[None,:])**2
                    elif method=='cka':cost=np.array([[1-cka(torch.tensor(hs[:,i]),torch.tensor(ht[:,j])) for j in range(3)] for i in range(3)])
                    else:cost=causal_cost
                    path=dtw(cost)
                    # Report causal disagreement for every alignment, even geometric ones.
                    distance=float(np.sqrt(np.mean([causal_cost[i,j] for i,j in path])))
                    source_e=raw[f's{source}__cached__held__identity__effects'][patient,:,1:5,0]
                    target_e=raw[f's{target}__{route}__held__identity__effects'][patient,:,1:5,0]
                    pred=np.stack([source_e[:,[i for i,jj in path if jj==j],:].mean(1) for j in range(3)],axis=1)
                    error=pred-target_e
                    pairs.append({'source_seed':source,'target_seed':target,'target_route':route,'method':method,'path':path,
                        'development_causal_distance':distance,'held_individual_rmse':float(np.sqrt(np.mean(error**2))),
                        'held_case_max_rmse':float(np.max(np.sqrt(np.mean(error**2,axis=(1,2))))),
                        'held_stage_rmse':np.sqrt(np.mean(error**2,axis=(0,2))).tolist(),
                        'calibration':source==11 and target==23,'primary_held':source in [11,23] and target==37,
                        'same_seed_self':source==target and route=='cached'})
                    raw[f'prediction_s{source}_t{target}_{route}_{method}']=pred
    # Append predictions without any new forward pass.
    np.savez_compressed(out/'raw_outputs.npz',**raw);dump(out/'PAIR_RESULTS.json',pairs)
    thresholds={}
    for method in ['depth','cka','causal']:
        cal=[p for p in pairs if p['method']==method and p['calibration']]
        positive=next(p['development_causal_distance'] for p in cal if p['target_route']=='cached')
        negative=next(p['development_causal_distance'] for p in cal if p['target_route']=='bypass')
        valid=positive+1e-7<negative;threshold=(positive+negative)/2
        held=[p for p in pairs if p['method']==method and p['primary_held']]
        thresholds[method]={'calibration_positive':positive,'calibration_negative':negative,'separable':valid,'threshold':threshold,
            'held_classification_correct':sum((p['development_causal_distance']<=threshold)==(p['target_route']=='cached') for p in held) if valid else None,
            'held_n':len(held),'same_route_prediction_passes':sum(p['held_individual_rmse']<=.1 for p in held if p['target_route']=='cached'),
            'same_route_prediction_n':2}
    numeric=all(s['native_oracle_max']<1e-8 and s['normal_roundtrip_output_max']<1e-8 and s['sham_logit_max']<1e-8 for s in metrics)
    linear=all(s['native_blind_effect_max']<1e-8 for s in metrics if s['coordinates']!='nonlinear')
    nonlinear=[s for s in metrics if s['coordinates']=='nonlinear' and s['route']=='cached' and s['split']=='held']
    summary={'decision':'NARROW' if numeric and linear else 'NO-GO: numerical controls',
        'numerical_controls_pass':numeric,'linear_blind_controls_pass':linear,'nonlinear_cached_held':nonlinear,
        'methods':thresholds,'elapsed_seconds':time.time()-started,'models':6,'seed_pairs':3,'interventions_run':True,
        'qualification':'Same-route models share a constrained information route, not a proven identical algorithm. Grammar features are supplied, questions/labels routed algebraically.'}
    dump(out/'SUMMARY.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--execute',action='store_true');a=parser.parse_args()
    if a.preflight:preflight()
    elif a.execute:execute()
    else:parser.error('Choose --preflight or --execute')
