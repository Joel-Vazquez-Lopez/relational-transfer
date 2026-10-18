"""Bounded known-program benchmark. No pretrained-model inference.
Run --preflight before freezing; run --execute only after a freeze receipt exists.
"""
import argparse, hashlib, itertools, json, sys, time, types
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
SEEDS = [11, 23, 37, 41, 53, 67, 79, 97]
METHODS = ['depth', 'cka', 'whole', 'targeted_full', 'graded']
CHANNELS = ['whole', 'role_full', 'order_full', 'role_quarter', 'role_threequarter', 'sham', 'nuisance']
VARIANTS = ['rotation', 'width', 'affine', 'idle', 'nuisance', 'nonlinear']


def cases(held=False):
    amplitudes = (.37, 1.31) if held else (1., 1.)
    rows = []
    for o,v,q,l,c,d in itertools.product([-1.,1.], repeat=6):
        # Held contexts alternate magnitudes independently of the four task factors.
        c1 = c * (amplitudes[0] if d < 0 else amplitudes[1])
        rows.append([o,v,o*v,q,l,o*v*q*l,c1,d])
    return np.asarray(rows)


def features(x):
    return np.column_stack([np.ones(len(x)),x])


class Program:
    def __init__(self, kind='cached', variant='base', seed=0):
        self.kind, self.variant, self.seed = kind, variant, seed
        self.ops = ['initial','cache','query','answer','readout']
        if variant == 'idle':
            self.ops = ['initial','noop','cache','noop','noop','query','noop','answer','readout']
        rng = np.random.default_rng(seed)
        self.A = np.eye(10)
        if variant in ['rotation','affine','nonlinear']:
            self.A = np.linalg.qr(rng.normal(size=(10,10)))[0]
        if variant == 'affine':
            self.A = self.A @ np.diag(np.linspace(.5,2.,10))
        if variant == 'width':
            self.A = np.linalg.qr(rng.normal(size=(14,10)))[0].T
        self.inv = np.linalg.pinv(self.A)
        self.alpha = .5 if variant == 'nonlinear' else 0.
        self.clock = 200. if variant == 'nuisance' else 20.

    def encode(self, s):
        t = np.array(s,copy=True)
        t[...,2] += self.alpha*t[...,0]*t[...,1]
        return t @ self.A

    def decode(self,h):
        t = np.asarray(h) @ self.inv
        t[...,2] -= self.alpha*t[...,0]*t[...,1]
        return t

    def advance(self,s,op,x,k):
        s = s.copy()
        if op == 'cache': s[2] = s[0]*s[1]
        if op == 'query': s[3] = x[3]
        if op == 'answer':
            route = self.kind
            if route == 'context': route = 'cached' if abs(s[6]) > .75 else 'recompute'
            z = s[0]*s[1] if route == 'recompute' else s[2]
            if route == 'threshold': z = 1. if z >= 0 else -1.
            s[5] = z*s[3]*s[4]
        s[8] = self.clock*k/(len(self.ops)-1)
        return s

    def trace(self,x):
        s = np.array([x[0],x[1],0.,0.,x[4],0.,x[6],x[7],0.,1.])
        states=[]
        for k,op in enumerate(self.ops):
            s=self.advance(s,op,x,k)
            states.append(self.encode(s))
        return np.asarray(states)

    def resume(self,h,k,x):
        s=self.decode(h)
        for j in range(k+1,len(self.ops)):
            s=self.advance(s,self.ops[j],x,j)
        return float(s[5])


def fit_components(traces,x):
    X=features(x)
    result=[]
    for k in range(traces.shape[1]):
        H=traces[:,k]
        B=np.linalg.lstsq(X,H,rcond=None)[0]
        H1=np.column_stack([np.ones(len(H)),H])
        row={}
        for name,col in [('role',3),('order',1),('nuisance',7)]:
            decoder=np.linalg.lstsq(H1,X[:,col],rcond=None)[0]
            residual=float(np.max(np.abs(H1@decoder-X[:,col])))
            row[name]=(B[col],decoder,residual)
        result.append(row)
    return result


def causal_data(p,x,fit):
    T=len(p.ops)
    out=np.empty((len(x),T,len(CHANNELS)))
    avail=np.empty((T,3),bool)
    traces=np.asarray([p.trace(a) for a in x])
    for k in range(T):
        for a,name in enumerate(['role','order','nuisance']):
            direction,decoder,resid=fit[k][name]
            avail[k,a] = np.linalg.norm(direction)>1e-8 and resid<1e-8
        for i,a in enumerate(x):
            h=traces[i,k]
            y=p.resume(h,k,a)
            donor=a.copy(); donor[0]*=-1; donor[2]*=-1; donor[5]*=-1
            whole=p.resume(p.trace(donor)[k],k,a)
            values=[whole]
            for name,dose in [('role',1.),('order',1.),('role',.25),('role',.75),('role',0.),('nuisance',1.)]:
                direction,decoder,resid=fit[k][name]
                column={'role':2,'order':0,'nuisance':6}[name]
                target=-a[column]
                estimate=np.r_[1.,h]@decoder
                patched=h+dose*(target-estimate)*direction if resid<1e-8 else h.copy()
                values.append(p.resume(patched,k,a))
            out[i,k]=(np.asarray(values)-y)/(-2*y)
    return traces,out,avail


def cka(a,b):
    a=a-a.mean(0); b=b-b.mean(0)
    den=np.linalg.norm(a.T@a)*np.linalg.norm(b.T@b)
    return float(np.linalg.norm(a.T@b)**2/den) if den else 0.


def align(cost):
    n,m=cost.shape
    D=np.full((n,m),np.inf); parent={}
    for i in range(n):
        for j in range(m):
            if i==j==0: D[i,j]=cost[i,j]; continue
            candidates=[]
            for di,dj,penalty in [(1,1,0.),(1,0,.01),(0,1,.01)]:
                if i>=di and j>=dj:
                    candidates.append((D[i-di,j-dj]+penalty,(i-di,j-dj)))
            value,prev=min(candidates,key=lambda z:z[0])
            D[i,j]=cost[i,j]+value; parent[i,j]=prev
    path=[(n-1,m-1)]
    while path[-1]!=(0,0): path.append(parent[path[-1]])
    return list(reversed(path))


def profile_path(a,b,channels):
    cost=np.mean((a[:,None,channels]-b[None,:,channels])**2,axis=-1)
    return align(cost),cost


def dsa_imports():
    import torch
    torch.set_num_threads(2)
    try: import ot
    except ImportError:
        guard=types.ModuleType('ot')
        def unavailable(*args,**kwargs): raise RuntimeError('POT unavailable: OT branch forbidden in this benchmark')
        for name in ['dist','emd','emd2','sinkhorn2']: setattr(guard,name,unavailable)
        sys.modules['ot']=guard
    sys.path.insert(0,str(ROOT/'vendor'))
    from DSA.dmd import DMD
    from DSA.simdist import SimilarityTransformDist
    return torch,DMD,SimilarityTransformDist


def dsa_score(a,b,seed):
    torch,DMD,Sim=dsa_imports()
    torch.manual_seed(seed)
    aa=DMD(a,n_delays=2,rank=8,device='cpu',verbose=False); aa.fit()
    bb=DMD(b,n_delays=2,rank=8,device='cpu',verbose=False); bb.fit()
    score=Sim(iters=200,lr=.01,score_method='angular',device='cpu',verbose=False).fit_score(aa.A_v,bb.A_v)
    return float(score.detach().cpu()) if hasattr(score,'detach') else float(score)


def preflight():
    # Infrastructure/unit checks only; no benchmark matching outcomes inspected.
    rng=np.random.default_rng(881)
    e=rng.normal(size=(7,10))
    errors={}
    for variant in ['base']+VARIANTS:
        p=Program(variant=variant,seed=881)
        err=float(np.max(abs(p.decode(p.encode(e))-e)))
        assert err<1e-10
        errors[variant]=err
    x=cases(); assert np.linalg.matrix_rank(features(x))==9
    # Independent synthetic linear dynamics: API smoke, no acceptance threshold fitting.
    a=rng.normal(size=(16,12,10))
    for k in range(1,12): a[:,k]=.8*a[:,k-1]+.05*a[:,k]
    score=dsa_score(a,a,881)
    result={'encoding_roundtrip_max':errors,'design_rank':9,'DSA_self_score':score,
            'purpose':'Infrastructure smoke only; benchmark results not evaluated.'}
    (ROOT/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


def execute():
    assert (ROOT/'FREEZE_RECEIPT.json').exists(),'Freeze protocol and source remotely before execution'
    t=time.time(); output=ROOT/'results'; output.mkdir(exist_ok=False)
    xdev,xheld=cases(),cases(True)
    pair_rows=[]; dsa_rows=[]; raw={}; program_rows=[]
    for seed in SEEDS:
        specs=[('cached','base')]+[('cached',v) for v in VARIANTS]+[('recompute','base'),('threshold','base'),('context','base')]
        data={}
        for kind,variant in specs:
            name=kind+'__'+variant
            p=Program(kind,variant,seed)
            natural=np.asarray([p.trace(a) for a in xdev])
            fit=fit_components(natural,xdev)
            hd,cd,ad=causal_data(p,xdev,fit)
            hh,ch,ah=causal_data(p,xheld,fit)
            data[name]=(hd,cd,hh,ch,ad)
            raw[f's{seed}__{name}__natural_dev']=hd
            raw[f's{seed}__{name}__natural_held']=hh
            raw[f's{seed}__{name}__effects_dev']=cd
            raw[f's{seed}__{name}__effects_held']=ch
            raw[f's{seed}__{name}__availability']=ad
            final=hh[:,-1] @ p.inv
            # Use actual resume readout, not raw encoded coordinate under nonlinear map.
            outputs=np.asarray([p.resume(hh[i,-1],len(p.ops)-1,a) for i,a in enumerate(xheld)])
            program_rows.append({'seed':seed,'program':name,'baseline_correct':int(np.sum(abs(outputs-xheld[:,5])<1e-8)),
                'n':len(xheld),'sham_max':float(np.max(abs(ch[:,:,5]))),'nuisance_max':float(np.max(abs(ch[:,:,6])))})
        source=data['cached__base']
        for name,target in data.items():
            if name=='cached__base': continue
            hd,cd,hh,ch,avail=target
            A=source[1][xdev[:,3]>0].mean(0); B=cd[xdev[:,3]>0].mean(0)
            At=source[3][xheld[:,3]<0].mean(0); Bt=ch[xheld[:,3]<0].mean(0)
            ckacost=np.asarray([[1-cka(source[0][:,i],hd[:,j]) for j in range(hd.shape[1])] for i in range(5)])
            depthcost=np.asarray([[(i/4-j/(hd.shape[1]-1))**2 for j in range(hd.shape[1])] for i in range(5)])
            for method in METHODS:
                if method=='depth': path=align(depthcost); cost=depthcost
                elif method=='cka': path=align(ckacost); cost=ckacost
                else:
                    channels={'whole':[0],'targeted_full':[0,1,2],'graded':[0,1,2,3]}[method]
                    path,cost=profile_path(A,B,channels)
                prediction=[]; truth=[]
                for j in range(Bt.shape[0]):
                    src=[i for i,jj in path if jj==j]
                    prediction.append(float(np.mean(At[src,4])))
                    truth.append(float(Bt[j,4]))
                distance=float(np.sqrt(np.mean([cost[i,j] for i,j in path])))
                # Same scalar benchmark for all alignments: predict held-out dose .75 effects.
                rmse=float(np.sqrt(np.mean((np.array(prediction)-truth)**2)))
                pair_rows.append({'seed':seed,'split':'development' if seed in SEEDS[:2] else 'heldout',
                    'target':name,'method':method,'distance':distance,'heldout_dose_rmse':rmse,
                    'path':path,'prediction':prediction,'truth':truth,
                    'scope':'stress' if name in ['cached__nonlinear','context__base'] else 'core',
                    'expected_same':name.startswith('cached__')})
            # Fixed operational DSA baseline; own self check already in preflight.
            if name=='context__base': continue  # cap: 64 descriptive DSA comparisons
            try:
                score=dsa_score(source[0],hd,seed)
                dsa_rows.append({'seed':seed,'target':name,'score':score,'error':None})
            except Exception as exc:
                dsa_rows.append({'seed':seed,'target':name,'score':None,'error':repr(exc)})
        print(f'seed {seed} complete ({time.time()-t:.1f}s)',flush=True)
    raw['cases_dev']=xdev; raw['cases_held']=xheld
    np.savez_compressed(output/'raw_arrays.npz',**raw)
    for name,rows in [('pair_results',pair_rows),('program_controls',program_rows),('dsa_results',dsa_rows)]:
        (output/(name+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
    # Thresholds learned from development seeds only, with no adjustment after overlap.
    decisions={}
    for method in METHODS:
        dev=[r for r in pair_rows if r['method']==method and r['split']=='development' and r['scope']=='core']
        maxsame=max(r['distance'] for r in dev if r['expected_same'])
        mindiff=min(r['distance'] for r in dev if not r['expected_same'])
        threshold=(maxsame+mindiff)/2
        valid=maxsame+1e-7<mindiff
        held=[r for r in pair_rows if r['method']==method and r['split']=='heldout' and r['scope']=='core']
        decisions[method]={'dev_max_same':maxsame,'dev_min_different':mindiff,'threshold':threshold,
            'separable':valid,'held_correct':sum((r['distance']<=threshold)==r['expected_same'] for r in held) if valid else None,
            'held_n':len(held),'held_dose_rmse_mean':float(np.mean([r['heldout_dose_rmse'] for r in held])),
            'held_same_dose_rmse_max':max(r['heldout_dose_rmse'] for r in held if r['expected_same'])}
    summary={'methods':decisions,'elapsed_seconds':time.time()-t,'seeds':SEEDS,
        'program_instances':len(program_rows),'cases_per_instance_split':len(xheld),
        'normal_outputs_correct':sum(r['baseline_correct'] for r in program_rows),
        'normal_outputs_n':sum(r['n'] for r in program_rows),
        'DSA_error_count':sum(r['error'] is not None for r in dsa_rows),
        'DSA_comparisons':len(dsa_rows),'all_sham_max':max(r['sham_max'] for r in program_rows),
        'all_nuisance_max':max(r['nuisance_max'] for r in program_rows)}
    (output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--preflight',action='store_true'); parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.preflight: preflight()
    elif args.execute: execute()
    else: parser.error('Choose --preflight or --execute')
