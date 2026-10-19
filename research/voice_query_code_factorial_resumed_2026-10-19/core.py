"""Inference-free definitions for one frozen factorial diagnostic."""
from pathlib import Path
import json,hashlib,os
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
LENGTH=128;BLOCKS=28
FAMILIES=[('j01','development','Bianca','Edmund','detained','detain'),('j02','development','Gemma','Rafael','dismissed','dismiss'),('j03','development','Aisha','Colin','photographed','photograph'),('j04','evaluation','Ingrid','Stefan','vaccinated','vaccinate'),('j05','evaluation','Yara','Rupert','supervised','supervise'),('j06','evaluation','Selma','Damian','fined','fine')]
SYSTEM='Decide whether the proposition in the question is entailed by the sentence. Otherwise classify it as not entailed. Follow the response legend. Output only the required symbol.'
OLD_SYSTEM='Answer the question using only the sentence. Reply with only Yes or No.'
BACKGROUND='Background: The shelf holds a plain folder beside a closed notebook. The wall is white and the room is quiet.\n'
PREFIXES=['','Record: ']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def mask(row,mode,cut,block):
 a=np.tri(LENGTH,dtype=bool)
 if mode=='baseline' or block<cut:return a
 s,p=row['source_start'],row['period']
 if mode=='frozen':a[p:,s:p]=False
 elif mode=='closed':a[p:,s:p]=False;a[p+1:,p]=False
 elif mode=='irrelevant':
  lo,hi=row['neutral_span'];assert hi-lo==p-s+1;a[p:,lo:hi-1]=False;a[p+1:,hi-1]=False
 else:raise ValueError(mode)
 assert a.any(axis=1).all();assert not a[np.triu_indices(LENGTH,1)].any()
 return a

def lse(a):return float(np.logaddexp.reduce(np.asarray(a,dtype=np.float64)))
def score(logits,row,decode):
 aliases=row['aliases'];best=int(np.argmax(logits));cls={k:lse(logits[v]) for k,v in aliases.items()};actual=next((k for k,v in aliases.items() if best in v),None);top=np.argsort(logits)[-12:][::-1]
 return dict(expected=row['expected'],actual=actual,native_correct=actual==row['expected'],format_compliant=actual is not None,argmax_id=best,argmax_token=decode([best]),class_logits=cls,s_entail=cls['entailed']-cls['not_entailed'],top12=[dict(id=int(i),token=decode([int(i)]),logit=float(logits[i])) for i in top])
def unit(rr):
 assert len(rr)==4 and {(r['role'],r['query']) for r in rr}=={(0,0),(0,1),(1,0),(1,1)}
 x={(r['role'],r['query']):r['s_entail'] for r in rr};mu_yes=(x[0,0]+x[1,1])/2;mu_no=(x[0,1]+x[1,0])/2
 ordinal=[(-1)**q*(x[0,q]-x[1,q]) for q in [0,1]]+[(-1)**r*(x[r,0]-x[r,1]) for r in [0,1]]
 return dict(D=(mu_yes-mu_no)/2,positive_mean=mu_yes,negative_mean=mu_no,offset=(mu_yes+mu_no)/2,ordinal_pairs=ordinal,all_ordinal_positive=all(v>0 for v in ordinal),native_correct=sum(r['native_correct'] for r in rr),format_compliant=sum(r['format_compliant'] for r in rr),native_complete=all(r['native_correct'] for r in rr))

def patterns(groups,family):
 def get(voice,qvoice,lex,m,pos,mode='baseline'):
  return next(u for u in groups if (u['family'],u['voice'],u['question_voice'],u['lexicon'],u['mapping'],u['position'],u['panel'],u['mode'])==(family,voice,qvoice,lex,m,pos,'factorial',mode))
 contrasts=[]
 for qv in ['active','passive']:
  for lex in ['YN','AB']:
   for m in [0,1]:
    for comparison in ['same_position0','same_position1','matched_period']:
     av=1 if comparison in ['same_position1','matched_period'] else 0;pv=1 if comparison=='same_position1' else 0
     gap=get('passive',qv,lex,m,pv)['positive_mean']-get('active',qv,lex,m,av)['positive_mean']
     contrasts.append(dict(question_voice=qv,lexicon=lex,mapping=m,comparison=comparison,gap=gap))
 old=[u for u in groups if u['family']==family and u['panel']=='old_reference' and u['mode']=='baseline'];assert len(old)==2;oldgap=next(u['positive_mean'] for u in old if u['voice']=='passive')-next(u['positive_mean'] for u in old if u['voice']=='active')
 compliance=sum(u['format_compliant'] for u in groups if u['family']==family and u['panel']=='factorial' and u['mode']=='baseline')/128
 # Disjoint operational patterns, not exhaustive theories of linguistic processing.
 source=oldgap>=4 and all(c['gap']>=4 for c in contrasts)
 match=oldgap>=4 and all(c['gap']>=4 if c['question_voice']=='active' else c['gap']<=-4 for c in contrasts)
 label=oldgap>=4 and all((c['gap']>=4 if c['mapping']==0 else c['gap']<=-4) if c['lexicon']=='YN' else abs(c['gap'])<=1 for c in contrasts)
 flags={k:bool(v and compliance>=.9) for k,v in [('construction_consistent',source),('voice_match_penalty',match),('literal_label_linked',label)]}
 assert sum(flags.values())<=1
 return dict(family=family,old_positive_gap=oldgap,format_compliance=compliance,contrasts=contrasts,patterns=flags)
def choose(profiles):
 counts={k:sum(p['patterns'][k] for p in profiles) for k in ['construction_consistent','voice_match_penalty','literal_label_linked']}
 candidates=[k for k,v in counts.items() if v>=2];assert len(candidates)<=1
 return dict(candidate=candidates[0] if candidates else None,counts=counts,rule='At least2/3 complete lexical families; no p values or population inference')
