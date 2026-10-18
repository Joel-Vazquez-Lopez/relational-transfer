"""Complete saved-probe curves, not layer selection."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
D=json.loads((P/'RESULTS.json').read_text());rr=D['all']
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
for i,m in enumerate(['qwen7b','llama3b']):
 for j,site in enumerate(['period','answer']):
  ax=axes[i,j]
  for train,col in [('a01','#2563eb'),('a02','#b91c1c')]:
   r=[x for x in rr if x['study']=='atlas' and x['model']==m and x['site']==site and x['training']==train and x['decoder_fit']=='outgoing_shared' and x['boundary']=='outgoing']
   ax.plot([x['layer']+1 for x in r],[x['gain'] for x in r],color=col,label=train+' → other family')
   perm=[x for x in rr if x['study']=='atlas_permutation' and x['model']==m and x['site']==site and x['training']==train]
   ax.plot([x['layer']+1 for x in perm],[np.log(2)*1.5-np.mean([t['ce'] for t in x['all']]) for x in perm],color=col,ls=':',alpha=.5,label=train+' all 12 permutations mean')
  ax.axhline(0,c='black',lw=.7);ax.grid(alpha=.15);ax.set_title(m+' · '+site);ax.set_ylabel('Held-family log-score gain over prior (nats)');ax.set_xlabel('Residual block boundary');ax.legend(fontsize=7)
fig.suptitle('Unqueried Theme/Goal accessibility — two-family exploratory diagnostic');fig.tight_layout()
for ext in ['png','svg']:fig.savefig(P/('ACCESSIBILITY.'+ext),dpi=160)
summary=[]
for m in ['qwen7b','llama3b']:
 for site in ['period','answer']:
  for fold in ['a01','a02']:
   x=next(x for x in rr if x['study']=='atlas' and x['model']==m and x['site']==site and x['training']==fold and x['layer']==27 and x['boundary']=='outgoing' and x['decoder_fit']=='outgoing_shared')
   summary.append({k:x[k] for k in ['model','site','training','test','gain','ce','correct','n']})
# Replay every recorded probability's loss and classification.
checks=0
for r in rr:
 if 'probabilities' not in r:continue
 p=np.asarray(r['probabilities']);y=np.asarray(r['labels']);ce=-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()
 assert abs(ce-r['ce'])<1e-10 and int(np.sum(p.argmax(1)==y))==r['correct'];assert np.max(abs(p.sum(1)-1))<1e-10;checks+=1
root=P.parent.parent;graded=root/'research/graded_relational_routing_2026-10-18';payload_checks=0
for model in ['qwen7b','llama3b']:
 for r in map(json.loads,(graded/model/'RAW.jsonl').read_text().splitlines()):
  if r['kind']=='scientific' and r['mode']=='baseline':
   assert hashlib.sha256((graded/model/r['payload']).read_bytes()).hexdigest()==r['payload_sha256'];payload_checks+=1
out=dict(final_boundary=summary,probability_replays=checks,graded_baseline_payload_hashes=payload_checks,all_passed=True,no_new_transformer_forwards=True,probe_warnings=D['warnings'])
(P/'AUDIT_SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
