"""Fixed definitions; no model initialization."""
from pathlib import Path
import hashlib,json,os
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
LENGTH=128
SYSTEM='Decide whether the proposition in the question is entailed by the sentence. Otherwise classify it as not entailed. Follow the response legend. Output only the required symbol.'
BACKGROUND='Background: The shelf holds a plain folder beside a closed notebook. The wall is white and the room is quiet.\n'
FAMILIES=[('l01','Clara','Mateo','arrested','arrest'),('l02','Hector','Nadia','thanked','thank')]
PREDICTIONS=('negative_category','last_option','relational_answer')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def score(logits,row,decode):
    ids=row['aliases'];z=np.asarray(logits,dtype=np.float64);best=int(np.argmax(z))
    cls=next((k for k,v in ids.items() if best in v),None)
    margins={k:float(np.logaddexp.reduce(z[v])) for k,v in ids.items()}
    top=np.argsort(z)[-12:][::-1]
    return dict(actual=cls,argmax_id=best,argmax_token=decode([best]),native_correct=cls==row['expected'],format_compliant=cls is not None,s_entail=margins['entailed']-margins['not_entailed'],class_logits=margins,top12=[dict(id=int(i),token=decode([int(i)]),logit=float(z[i])) for i in top])
def predicted(row,hypothesis):
    return {'negative_category':'not_entailed','last_option':row['last_class'],'relational_answer':row['expected']}[hypothesis]
def classify(rows):
    counts={}
    for h in PREDICTIONS:
        total=sum(r['actual']==predicted(r,h) for r in rows)
        families={f:sum(r['actual']==predicted(r,h) for r in rows if r['family']==f) for f,*_ in FAMILIES}
        orders={o:sum(r['actual']==predicted(r,h) for r in rows if r['legend_order']==o) for o in ['entailed_first','nonentailed_first']}
        passes=total>=28 and min(families.values())>=14 and min(orders.values())>=14
        counts[h]=dict(matches=total,n=len(rows),by_family=families,by_order=orders,qualifies=passes)
    qualified=[h for h,c in counts.items() if c['qualifies']]
    assert len(qualified)<=1
    return dict(candidate=qualified[0] if qualified else None,counts=counts,causal_task_ready=all(r['native_correct'] for r in rows),no_p_values=True)
