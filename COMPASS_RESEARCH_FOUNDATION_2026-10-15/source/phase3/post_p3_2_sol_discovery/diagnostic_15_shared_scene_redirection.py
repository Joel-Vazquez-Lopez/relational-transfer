"""D15: exact shared-scene source/target redirection for still-withheld pair 1-2."""
import csv,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=Path('/Users/joelvazquezlopez/Documents/Codex/2026-09-08/files-pasted-by-the-user-you')
sys.path.insert(0,str(ROOT/'src'))
import d5_topology as d5
from pilot import Agents
E='E_path3_missing_exploratory'
MODELS=[('A_original_clique_hole',s) for s in (1701,1702,1703)]+[(E,s) for s in (2701,2702,2703)]
assert not any(seed in (1704,1705,1706) for _,seed in MODELS)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,rows):
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

scenes=[]
for factor in ('source','target'):
    for third in (4,5):
        for event in (0,1):
            for repeat in range(32):
                meanings=(d5.qset('S',1,third,2,event) if factor=='source' else
                          d5.qset('S',third,1,2,event))
                original,alternative=0,(1 if factor=='source' else 2)
                order=[(k+repeat)%4 for k in range(4)]
                scene=[]
                for position,member in enumerate(order):
                    meaning=meanings[member]
                    allowed=d5.allowed_distractors(meaning)
                    distractor=allowed[(repeat+position)%4]
                    tensor,_=d5.render_fixed(meaning,[991,repeat,third,event,position,
                                                         0 if factor=='source' else 1],distractor)
                    scene.append(tensor)
                scenes.append({'factor':factor,'third':third,'event':event,'repeat':repeat,
                               'base_meaning':meanings[original],'after_meaning':meanings[alternative],
                               'base_pos':order.index(original),'after_pos':order.index(alternative),
                               'candidate_tensor':np.stack(scene)})
assert len(scenes)==256
candidate_tensors=torch.tensor(np.stack([r['candidate_tensor'] for r in scenes]))
rows=[];hashes={'src/d5_topology.py':sha(ROOT/'src/d5_topology.py'),
                'src/pilot.py':sha(ROOT/'src/pilot.py'),
                'configs/pilot.json':sha(ROOT/'configs/pilot.json')}
for cond,seed in MODELS:
    ck=(HERE/'exploratory_E/runs'/E/f'seed_{seed}'/'checkpoint.pt' if cond==E else
        ROOT/'results/d5_topology/runs'/cond/f'seed_{seed}'/'checkpoint.pt')
    hashes[str(ck)]=sha(ck)
    package=torch.load(ck,map_location='cpu',weights_only=True)
    model=Agents(package['config']);model.load_state_dict(package['model'],strict=True);model.eval()
    with torch.no_grad():
        c=F.normalize(model.candidate(candidate_tensors.reshape(-1,3,14)),dim=-1).reshape(len(scenes),4,48)
        qbase=d5.extract_queries(model,[r['base_meaning'] for r in scenes])[1]
        qafter=d5.extract_queries(model,[r['after_meaning'] for r in scenes])[1]
        sbase=torch.einsum('bd,bkd->bk',qbase,c)*10
        safter=torch.einsum('bd,bkd->bk',qafter,c)*10
        # Independent direct model scores validate the path at a pilot batch.
        msg=d5.oracle_messages([r['after_meaning'] for r in scenes[:8]])[0]
        direct=model.listen(msg,candidate_tensors[:8])
        assert torch.max(torch.abs(direct-safter[:8])).item()<1e-5
    for i,r in enumerate(scenes):
        p0,p1=r['base_pos'],r['after_pos']
        dq=(qafter[i]-qbase[i]).numpy();dc=(c[i,p1]-c[i,p0]).numpy()
        n1=float(np.linalg.norm(dq));n2=float(np.linalg.norm(dc))
        rows.append(dict(condition=cond,seed=seed,factor=r['factor'],third=r['third'],
                         event=r['event'],repeat=r['repeat'],base_pos=p0,after_pos=p1,
                         query_delta=n1,candidate_contrast=n2,
                         alignment_cos=float(np.dot(dq,dc)/(n1*n2)) if n1*n2 else 0,
                         base_margin=float(sbase[i,p0]-sbase[i,p1]),
                         after_margin=float(safter[i,p1]-safter[i,p0]),
                         base_correct=int(int(sbase[i].argmax())==p0),
                         redirected=int(int(safter[i].argmax())==p1)))
rowfile=HERE/'d15_scene_rows.csv';write(rowfile,rows)
groups=defaultdict(list)
for r in rows:groups[(r['condition'],r['seed'],r['factor'])].append(r)
fields=['query_delta','candidate_contrast','alignment_cos','base_margin','after_margin','base_correct','redirected']
summary=[]
for key,g in sorted(groups.items()):
    d=dict(zip(['condition','seed','factor'],key));d['scenes']=len(g)
    d.update({f:float(np.mean([r[f] for r in g])) for f in fields});summary.append(d)
sumfile=HERE/'d15_model_summary.csv';write(sumfile,summary)
manifest={'diagnostic':'D15','status':'discovery_only','selected_after':'D09_before_D12_results',
          'input_hashes_sha256':hashes,'script_sha256':sha(Path(__file__)),
          'output_hashes_sha256':{f:sha(HERE/f) for f in ('d15_scene_rows.csv','d15_model_summary.csv')},
          'scene_count':len(scenes),'model_count':len(MODELS),'protected_seeds_accessed':False}
(HERE/'d15_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for cond in ('A_original_clique_hole',E):
    for factor in ('source','target'):
        g=[r for r in summary if r['condition']==cond and r['factor']==factor]
        print(cond[:1],factor,'models',len(g),
              'base_correct',round(np.mean([r['base_correct'] for r in g]),3),
              'redirected',round(np.mean([r['redirected'] for r in g]),3),
              'q_delta',round(np.mean([r['query_delta'] for r in g]),3),
              'c_delta',round(np.mean([r['candidate_contrast'] for r in g]),3),
              'alignment',round(np.mean([r['alignment_cos'] for r in g]),3),
              'after_margin',round(np.mean([r['after_margin'] for r in g]),3))
