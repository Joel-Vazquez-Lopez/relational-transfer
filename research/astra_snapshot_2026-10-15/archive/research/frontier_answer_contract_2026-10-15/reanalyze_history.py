"""Exploratory read-only reanalysis; only explicitly named examined files."""
from pathlib import Path
import json,re,hashlib,collections
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1];hashes={}
def load(rel):
 p=PROJECT/rel;hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def rows(rel):
 p=PROJECT/rel;hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return [json.loads(x) for x in p.read_text().splitlines()]
def n(s):return re.sub(r'\s+',' ',re.sub(r'[^\w\s]',' ',(s or '').casefold())).strip()
def contained(span,text):return ' '+n(span)+' ' in ' '+n(text)+' '
def fields(x):return (n(x.get('actor')),n(x.get('patient')))
out={'status':'post hoc exploratory reanalysis, no new inference; no independent confirmation','source_hashes':hashes}
r='research/frozen_query_diagnostic_2026-10-15/';items={x['id']:x for x in load(r+'ITEMS.json')['items']};out['direct_error_types']={};out['field_only_by_question']={}
for model in ['0.5b','1.5b']:
 records=rows(r+f'PREDICTIONS_{model}.jsonl');follow=rows(r+f'FIELD_ONLY_{model}.jsonl');audit={}
 for role in ['subject','object']:
  c=collections.Counter();examples=[]
  for x in records:
   if x['arm']!='direct' or x['role']!=role:continue
   i=items[x['id']];other='object' if role=='subject' else 'subject';raw=x['raw'];correct=n(raw) in {n(s) for s in i[role+'_accepted']};has=any(contained(s,raw) for s in i[role+'_accepted']);opp=any(contained(s,raw) for s in i[other+'_accepted'])
   category='exact_correct' if correct else ('contains_correct_only' if has and not opp else 'contains_both' if has and opp else 'contains_opposite_only' if opp else 'contains_neither');c[category]+=1;c['n']+=1
   if not correct and has:examples.append({'id':x['id'],'question':i[role+'_question'],'raw':raw,'accepted':i[role+'_accepted'],'category':category})
  audit[role]={'counts':dict(c),'examples':examples}
 out['direct_error_types'][model]=audit
 out['field_only_by_question'][model]={}
 for qrole in ['subject','object']:
  rr=[x for x in follow if x['role']==qrole];out['field_only_by_question'][model][qrole]={k:sum(n(x['fields'].get(k,'')) in {n(a) for a in items[x['id']][k+'_accepted']} for x in rr) for k in ['subject','object']}
# Old causal panel: rules predict behavior, not semantic correctness.
r=load('llm_role_internal_v2/RESULT.json')['records'];grouped={};out['old_patch_panel']={'n':len(r),'last_mention_by_condition':{},'observed_correct_by_verb':{}}
for w in sorted({x['wording'] for x in r}):
 rr=[x for x in r if x['wording']==w];last=lambda x:x['patient'] if x['voice']=='active' else x['actor'];out['old_patch_panel']['last_mention_by_condition'][w]={'n':len(rr),'last_mention':sum(n(x['first_token'])==n(last(x)) for x in rr)}
 for verb in sorted({x['verb'] for x in rr}):
  aa=[x for x in rr if x['verb']==verb and x['voice']=='active'];out['old_patch_panel']['observed_correct_by_verb'][w+'/'+verb]={'n':len(aa),'correct':sum(x['correct'] for x in aa)}
for unit in ['pair_index','verb']:
 hit=0;total=0
 for held in sorted({x[unit] for x in r}):
  train=[x for x in r if x[unit]!=held];test=[x for x in r if x[unit]==held];rule={}
  for w in sorted({x['wording'] for x in train}):
   for voice in ['active','passive']:
    sub=[x for x in train if x['wording']==w and x['voice']==voice];first=lambda x:x['actor'] if x['voice']=='active' else x['patient'];first_count=sum(n(x['first_token'])==n(first(x)) for x in sub);rule[w,voice]='first' if first_count>len(sub)/2 else 'last'
  for x in test:
   first=x['actor'] if x['voice']=='active' else x['patient'];last=x['patient'] if x['voice']=='active' else x['actor'];pred=first if rule[x['wording'],x['voice']]=='first' else last;hit+=n(pred)==n(x['first_token']);total+=1
 out['old_patch_panel']['leave_one_'+unit+'_out_behavior_prediction']={'correct':hit,'n':total,'features':'question type and voice; chooses first/last mentioned participant','limit':'retrospective model class, not independent prospective validation'}
# Natural verifier: interventions on drafts and requested roles, all development only.
b='research/natural_role_verifier_development_2026-10-13/';out['natural_verifier']=[]
for start in ['fresh1','fresh3']:
 for arm in ['verifier_clean','verifier_noisy']:
  normal=load(b+f'EVAL_{start}_{arm}_development_normal.json')['records'];swapped=load(b+f'EVAL_{start}_{arm}_development_swapped.json')['records'];idx={(x['id'],x['role'],x['order']):x for x in swapped};c=collections.Counter(n=len(normal));events=collections.defaultdict(list)
  for x in normal:
   y=idx[x['id'],x['role'],x['order']];same=fields(x)==fields(y);flip=n(x['answer'])!=n(y['answer']);good=same and x['actor_correct'] and x['patient_correct'];c['same_fields']+=same;c['answer_changed']+=flip;c['same_fields_answer_changed']+=same and flip;c['same_correct_fields']+=good;c['same_correct_fields_answer_changed']+=good and flip;events[x['id']].append(bool(same and flip))
  c['distinct_events_with_same_fields_answer_change']=sum(any(v) for v in events.values());query={}
  for stress,rr in [('normal',normal),('swapped',swapped)]:
   idx={(x['id'],x['role'],x['order']):x for x in rr};cc=collections.Counter()
   for x in rr:
    if x['role']!='actor':continue
    y=idx[x['id'],'patient',x['order']];same=fields(x)==fields(y);flip=n(x['answer'])!=n(y['answer']);good=same and x['actor_correct'] and x['patient_correct'];cc['pairs']+=1;cc['same_fields']+=same;cc['same_fields_answer_changed']+=same and flip;cc['same_correct_fields']+=good;cc['same_correct_fields_answer_changed']+=good and flip;cc['both_answers_correct_given_same_correct_fields']+=good and x['correct'] and y['correct']
   query[stress]=dict(cc)
  out['natural_verifier'].append({'start':start,'arm':arm,'draft_swap':dict(c),'requested_role_swap':query})
for rel,h in hashes.items():assert hashlib.sha256((PROJECT/rel).read_bytes()).hexdigest()==h
(ROOT/'HISTORICAL_REANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n');print('Saved historical reanalysis; sources unchanged:',len(hashes))
