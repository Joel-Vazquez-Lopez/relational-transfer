"""Independent NumPy reconstruction of every saved neural intervention logit."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
P=Path(__file__).resolve().parent;D=P/'results';w=np.load(D/'checkpoints.npz')
summary=json.loads((D/'SUMMARY.json').read_text())
assert summary['interventions_run'],'No intervention assay to audit: competence screen stopped study'
z=np.load(D/'raw_outputs.npz');checks=[];maximum=0.

def linear(x,seed,key):return x@w[f'seed{seed}__{key}.weight'].T+w[f'seed{seed}__{key}.bias']
def states(x,seed):
    out=[]
    for k in range(3):x=np.tanh(linear(x,seed,f'enc.{k}'));out.append(x)
    return out

def resume(h,k,x,seed,route):
    if route=='bypass':
        v=x[:,:3]
        for key in ['bypass.0','bypass.2']:v=np.tanh(linear(v,seed,key))
        return linear(v,seed,'bypass.4')[:,0]
    for j in range(k+1,3):h=np.tanh(linear(h,seed,f'enc.{j}'))
    return linear(np.tanh(linear(h,seed,'head.0')),seed,'head.2')[:,0]

def check(name,delta):
    global maximum
    maximum=max(maximum,float(delta));checks.append({'check':name,'max_abs_error':float(delta),'pass':bool(delta<1e-8)})

for seed in [11,23,37]:
    xdrows=json.loads((D/'cases_dev.json').read_text());xd=np.array([r['features'] for r in xdrows]);rd=np.array([r['R'] for r in xdrows]);hd=states(xd,seed)
    for split,doses in [('dev',[0.,.25,.5,1.]),('held',[0.,.125,.375,.625,.875,1.])]:
        rows=json.loads((D/f'cases_{split}.json').read_text());x=np.array([r['features'] for r in rows]);r=np.array([a['R'] for a in rows]);hh=states(x,seed)
        for route in ['cached','bypass']:
            prefix=f's{seed}__{route}__{split}__'
            base=resume(hh[-1],2,x,seed,route)
            check(prefix+'clean',np.max(abs(base-z[prefix+'clean_role_logits'])))
            cf=x.copy();cf[:,0]*=-1;cfh=states(cf,seed);opp=resume(cfh[-1],2,cf,seed,route)
            check(prefix+'normalizer',np.max(abs(r*(base-opp)-z[prefix+'normalizer'])))
            for kind in ['identity','orthogonal','nonlinear']:
                saved=z[prefix+kind+'__logits'];effects=z[prefix+kind+'__effects']
                for k,h in enumerate(hh):
                    Q=np.linalg.qr(np.random.default_rng(seed*100+k).normal(size=(16,16)))[0]
                    def forward(v):
                        if kind=='identity':return v
                        return (np.sinh(2*v)/2 if kind=='nonlinear' else v)@Q
                    def inverse(v):
                        if kind=='identity':return v
                        v=v@Q.T
                        return np.arcsinh(2*v)/2 if kind=='nonlinear' else v
                    native_direction=hd[k][rd>0].mean(0)-hd[k][rd<0].mean(0)
                    ud=forward(hd[k]);blind_direction=ud[rd>0].mean(0)-ud[rd<0].mean(0)
                    check(prefix+kind+f'/stage{k}/direction',np.max(abs(blind_direction-z[prefix+kind+'__directions'][k])))
                    for j,dose in enumerate(doses):
                        native=h-dose*r[:,None]*native_direction
                        oracle=inverse(forward(native))
                        blind=inverse(forward(h)-dose*r[:,None]*blind_direction)
                        outs=np.stack([resume(a,k,x,seed,route) for a in [native,oracle,blind]],axis=-1)
                        check(prefix+kind+f'/stage{k}/dose{dose}',np.max(abs(outs-saved[:,k,j])))
                normalized=r[:,None,None,None]*(base[:,None,None,None]-saved)/(r*(base-opp))[:,None,None,None]
                check(prefix+kind+'/effects',np.max(abs(normalized-effects)))
            # Known route assertions, independent of fitted contrast quality.
            controls=z[prefix+'donor_control_role_logits']
            if route=='bypass':check(prefix+'unused_cache_all_blind',np.max(abs(z[prefix+'nonlinear__logits'][:,:,:,2]-base[:,None,None])))
            else:
                assert np.all(controls[:,:,0]*r[:,None]<0),'Role donor must reverse every cached answer'
                assert np.all(controls[:,:,1]*r[:,None]>0),'Same-role surface donor must preserve every cached answer'
            # Tensor equality verifies matched epiphenomenal cache, not same readout.
            check(prefix+'matched_cache_clone',np.max(abs(z[f's{seed}__cached__{split}__identity__natural']-z[f's{seed}__bypass__{split}__identity__natural'])))
result={'checks':len(checks),'passed':sum(v['pass'] for v in checks),'max_absolute_reconstruction_error':maximum,
        'all_native_cached_role_donor_switches_pass':True,'all_same_role_surface_donors_preserve_answers':True,
        'failed_checks':[v for v in checks if not v['pass']],
        'raw_sha256':hashlib.sha256((D/'raw_outputs.npz').read_bytes()).hexdigest(),
        'description':'Independent NumPy replay from saved weights; no importing study or retraining; includes every saved intervention logit.'}
(D/'INDEPENDENT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert not result['failed_checks']
