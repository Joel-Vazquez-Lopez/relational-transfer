"""Post hoc routing controls on cached outputs. No language-model inference.

Uses the pre-existing question-classifier recipe on the same 160 training events.
This is an exploratory secondary check, not part of the frozen primary test.
"""
import json,hashlib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from analyze import exact
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
train=json.loads((PROJECT/'research/candidate_free_role_binding_2026-10-14/SPLITS.json').read_text())['train']
items=json.loads((ROOT/'ITEMS.json').read_text())['items']
assert not {i['sentence'] for i in items}&{i['sentence'] for i in train}
qs=[];labels=[]
for x in train:
 qs.extend([x['actor_question'],x['patient_question']]);labels.extend(['subject','object'])
classifier=make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(2,5),lowercase=True),LogisticRegression(C=1,max_iter=2000,random_state=910127));classifier.fit(qs,labels)
pred={(i['id'],role):str(classifier.predict([i[role+'_question']])[0]) for i in items for role in ['subject','object']}
result={'status':'post_hoc_secondary_development','recipe':'Same TFIDF char2-5 logistic C1 seed910127 recipe as earlier typed-goal study; 160 training events only; no test fitting','source_split_sha256':sha(PROJECT/'research/candidate_free_role_binding_2026-10-14/SPLITS.json'),'train_events':len(train),'selector_correct':sum(p==role for (ident,role),p in pred.items()),'selector_total':48,'models':{}}
for model in ['0.5b','1.5b']:
 path=ROOT/f'PREDICTIONS_{model}.jsonl';records=[json.loads(l) for l in path.read_text().splitlines()];by={(r['id'],r['arm'],r['role']):r for r in records};metrics={}
 for arm in ['query_visible','query_blind']:
  for selector in ['gold','predicted']:
   correct=both=0
   for i in items:
    oks=[]
    for role in ['subject','object']:
     r=by[i['id'],arm,role if arm=='query_visible' else 'both'];key=role if selector=='gold' else pred[i['id'],role];ok=exact(r['fields'].get(key),i[role+'_accepted']);correct+=ok;oks.append(ok)
    both+=all(oks)
   metrics[arm+'_'+selector]={'exact':correct,'answer_total':48,'both':both,'event_total':24}
 result['models'][model]={'predictions_sha256':sha(path),'routing':metrics}
(ROOT/'SECONDARY_ROUTING.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
