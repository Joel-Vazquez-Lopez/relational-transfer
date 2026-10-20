"""D45: frozen readout after releasing or retaining E identity ties."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=Path('/Users/joelvazquezlopez/Documents/Codex/2026-09-08/files-pasted-by-the-user-you')
sys.path.insert(0,str(ROOT/'src'))
import d5_topology as d5
from pilot import Agents
MODELS={name:HERE/'exploratory_E_tie_release'/name for name in ('release','retain')}
STEPS=(0,10,50,100,250,500,1000,2000)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

scenes=[]
for factor in ('source','target'):
    for third in (4,5):
        for event in (0,1):
            for repeat in range(32):
                meanings=(d5.qset('S',1,third,2,event) if factor=='source' else
                          d5.qset('S',third,1,2,event))
                alternative=1 if factor=='source' else 2
                order=[(k+repeat)%4 for k in range(4)]
                candidates=[]
                for position,member in enumerate(order):
                    meaning=meanings[member]
                    distractor=d5.allowed_distractors(meaning)[(repeat+position)%4]
                    tensor,_=d5.render_fixed(meaning,[991,repeat,third,event,position,
                                                       0 if factor=='source' else 1],distractor)
                    candidates.append(tensor)
                scenes.append({'factor':factor,'base':meanings[0],
                               'after':meanings[alternative],
                               'base_pos':order.index(0),'after_pos':order.index(alternative),
                               'candidates':np.stack(candidates)})
assert len(scenes)==256
tensor=torch.tensor(np.stack([s['candidates'] for s in scenes]))
base=[s['base'] for s in scenes];after=[s['after'] for s in scenes]
hashes={'src/d5_topology.py':sha(ROOT/'src/d5_topology.py'),
        'src/pilot.py':sha(ROOT/'src/pilot.py'),
        'd15_scene_rows.csv':sha(HERE/'d15_scene_rows.csv')}
rows=[]
for name,run in MODELS.items():
    for step in STEPS:
        ck=run/f'checkpoint_step_{step:05d}.pt'
        hashes[str(ck.relative_to(HERE))]=sha(ck)
        package=torch.load(ck,map_location='cpu',weights_only=True)
        model=Agents(package['config']);model.load_state_dict(package['model'],strict=True);model.eval()
        state=model.state_dict();a=state['sym'][1]+state['mpos'][0,1];b=state['sym'][2]+state['mpos'][0,1]
        symbol_delta=float(torch.linalg.vector_norm(state['sym'][1]-state['sym'][2]))
        candidate_input_delta=float(torch.linalg.vector_norm(
            state['candidate.inp.weight'][:,1]-state['candidate.inp.weight'][:,2]))
        norm=state['msg_blocks.0.n1.weight'];bias=state['msg_blocks.0.n1.bias']
        dz=F.layer_norm(b,(48,),norm,bias)-F.layer_norm(a,(48,),norm,bias)
        wv=state['msg_blocks.0.qkv.weight'][96:144]
        gain=float(torch.linalg.vector_norm(F.linear(dz,wv))/torch.linalg.vector_norm(dz))
        with torch.no_grad():
            c=F.normalize(model.candidate(tensor.reshape(-1,3,14)),dim=-1).reshape(256,4,48)
            q0=d5.extract_queries(model,base)[1]
            q1=d5.extract_queries(model,after)[1]
            msg0=d5.oracle_messages(base)[0];msg1=d5.oracle_messages(after)[0]
            def block0_cls(msg):
                x=torch.cat([model.mcls.expand(len(msg),-1,-1),msg@model.sym],1)+model.mpos
                return F.normalize(model.msg_blocks[0](x)[:,0],dim=-1)
            q0b=block0_cls(msg0);q1b=block0_cls(msg1)
            scores=torch.einsum('bd,bkd->bk',q1,c)*10
        for factor in ('source','target'):
            indices=[i for i,s in enumerate(scenes) if s['factor']==factor]
            data=[]
            for i in indices:
                s=scenes[i];dq=(q1[i]-q0[i]).numpy();dc=(c[i,s['after_pos']]-c[i,s['base_pos']]).numpy()
                nq=float(np.linalg.norm(dq));nc=float(np.linalg.norm(dc))
                data.append({'q':nq,'q_block0':float(torch.linalg.vector_norm(q1b[i]-q0b[i])),
                             'c':nc,'cos':float(np.dot(dq,dc)/(nq*nc)) if nq*nc else 0,
                             'redirect':int(int(scores[i].argmax())==s['after_pos']),
                             'margin':float(scores[i,s['after_pos']]-scores[i,s['base_pos']])})
            rows.append({'model':name,'seed':1701,'update':step,'factor':factor,
                         'symbol_delta':symbol_delta,'candidate_input_delta':candidate_input_delta,
                         'value_projection_gain':gain,
                         'query_delta':float(np.mean([r['q'] for r in data])),
                         'block0_cls_delta':float(np.mean([r['q_block0'] for r in data])),
                         'candidate_contrast':float(np.mean([r['c'] for r in data])),
                         'alignment_cos':float(np.mean([r['cos'] for r in data])),
                         'redirected':float(np.mean([r['redirect'] for r in data])),
                         'after_margin':float(np.mean([r['margin'] for r in data]))})
outfile=HERE/'d45_release_developmental.csv'
with outfile.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
manifest={'diagnostic':'D45','status':'discovery_only','selected_before':'D44 release outcome',
          'input_hashes_sha256':hashes,'script_sha256':sha(Path(__file__)),
          'output_hashes_sha256':{'d45_release_developmental.csv':sha(outfile)},
          'protected_seeds_accessed':False}
(HERE/'d45_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for r in rows:
    print(r['model'],r['update'],r['factor'],'q',round(r['query_delta'],3),
          'c',round(r['candidate_contrast'],3),'redirect',round(r['redirected'],3))
