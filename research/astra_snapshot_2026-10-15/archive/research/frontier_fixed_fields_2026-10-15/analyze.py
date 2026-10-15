"""Paired development analysis. Never performs inference or reads sealed material."""
from pathlib import Path
import json,re,hashlib,collections,random,csv
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1];PREV=ROOT.parent/'frontier_answer_contract_2026-10-15'
def norm(s):return re.sub(r'\s+',' ',re.sub(r'[^\w\s]',' ',(s or '').casefold())).strip()
def semantic_norm(s):return ' '.join(t for t in norm(s).split() if t not in {'a','an','the'})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# Save exact individual-example gains/losses from earlier 24-event factorial.
olditems={x['id']:x for x in json.loads((PREV/'ITEMS.json').read_text())};oldrows=[json.loads(x) for x in (PREV/'PREDICTIONS.jsonl').read_text().splitlines()];oi={(x['id'],x['instruction'],x['prefix'],x['role']):x for x in oldrows};trans=[];transition_counts={}
for instr,pref in [('original','Answer:'),('explicit_span_contract',''),('explicit_span_contract','Answer:')]:
 name=instr+'/'+(pref or 'none');ct=collections.Counter()
 for i in olditems.values():
  for role in ['subject','object']:
   a=oi[i['id'],'original','',role];b=oi[i['id'],instr,pref,role];ok=lambda x:norm(x['raw']) in {norm(v) for v in i[role+'_accepted']};aa,bb=ok(a),ok(b);status='gain' if bb and not aa else 'loss' if aa and not bb else 'both_correct' if aa else 'both_wrong';ct[status]+=1
   trans.append({'id':i['id'],'verb':i['verb'],'role':role,'condition':name,'old_answer':a['raw'],'new_answer':b['raw'],'status':status,'accepted_spans':json.dumps(i[role+'_accepted'])})
 transition_counts[name]=dict(ct)
with (ROOT/'INTERFACE_TRANSITIONS.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(trans[0]));w.writeheader();w.writerows(trans)
(ROOT/'INTERFACE_TRANSITIONS.json').write_text(json.dumps({'scope':'Earlier 24-event factorial, post hoc per-item audit','counts':transition_counts,'gain_loss_rows':[x for x in trans if x['status'] in ['gain','loss']]},indent=2)+'\n')
# The following analysis requires the bounded run to have completed.
if not (ROOT/'COMPLETE.json').exists():
 print('Saved earlier interface transitions. New experiment not complete.');raise SystemExit(0)
c=json.loads((ROOT/'COMPLETE.json').read_text());assert sha(ROOT/'PREDICTIONS.jsonl')==c['predictions_sha256'];rows=[json.loads(x) for x in (ROOT/'PREDICTIONS.jsonl').read_text().splitlines()];assert len(rows)==540
items=json.loads((ROOT/'ITEMS.json').read_text());ix={x['id']:x for x in items};assert len({i['document_id'] for i in items})==12
key=lambda x:(x['id'],x['kind'],x.get('role'),x.get('sentence'),x.get('history'),x.get('order'),x.get('swap'));by={key(x):x for x in rows};assert len(by)==540
budget=json.loads((ROOT/'BUDGETS.json').read_text())['budgets'];replays={x['id']:x for x in rows if x['kind']=='extraction_replay'}
def metric(x):
 i=ix[x['id']];r=x['role'];raw=x['raw'];value=norm(raw);sem=semantic_norm(raw);sf=x.get('supplied_fields',{});isfail=x['finish_reason']!='eos';ok=value in {norm(t) for t in i[r+'_accepted']};semantic=sem in {semantic_norm(t) for t in i['referent_aliases'][r]};available=[]
 if x.get('sentence') or x['kind'] in ['direct','direct_review']:available.append(i['sentence'])
 available+=list(sf.values());contiguous=bool(value) and any(' '+value+' ' in ' '+norm(s)+' ' for s in available) and not (' '+norm(i['verb'])+' ' in ' '+value+' ');requested=norm(sf.get(r));opposite=norm(sf.get('object' if r=='subject' else 'subject'));nondegenerate=bool(requested and opposite and requested!=opposite);order=['subject','object'] if x.get('order')=='subject_first' else ['object','subject'];cost=x['forward_work']+x.get('prior_work',0)
 assert cost<=budget[x['id']]
 return {'exact':int(ok and not isfail),'semantic_proxy':int(semantic and not isfail),'compliant':int(contiguous and not isfail),'follows_requested':int(value==requested and nondegenerate and not isfail),'follows_opposite':int(value==opposite and nondegenerate and not isfail),'nondegenerate':int(nondegenerate),'follows_first':int(value==norm(sf.get(order[0])) and nondegenerate and not isfail),'follows_last':int(value==norm(sf.get(order[1])) and nondegenerate and not isfail),'budget_failure':int(isfail),'length_termination':int(x['finish_reason']=='budget_or_length'),'forward_work':cost,'budget_fraction':cost/budget[x['id']]}
scored=[{**x,'scores':metric(x)} for x in rows if x['kind']!='extraction_replay']
cell=lambda x:('/'.join([x['kind'],'sentence'+str(int(x['sentence'])),'history'+str(int(x['history'])),'swapped' if x['swap'] else 'intact']) if x['kind'] in ['generated','reference'] else x['kind'])
cells=collections.defaultdict(list)
for x in scored:cells[cell(x)].append(x)
summary={};event={}
for name,rr in cells.items():
 sub={i['id']:[x for x in rr if x['id']==i['id']] for i in items};event[name]={}
 for eid,part in sub.items():
  values={k:sum(x['scores'][k] for x in part)/len(part) for k in ['exact','semantic_proxy','compliant','follows_requested','forward_work','budget_fraction']};orders=sorted({x.get('order','none') for x in part},key=str);values['both_roles']=sum(all(x['scores']['exact'] for x in part if x.get('order','none')==o) for o in orders)/len(orders);event[name][eid]=values
 summary[name]={'n_outputs':len(rr),'n_events':12,**{k:sum(x['scores'][k] for x in rr) for k in ['exact','semantic_proxy','compliant','follows_requested','follows_opposite','nondegenerate','follows_first','follows_last','budget_failure','length_termination']},'mean_work_proxy':sum(x['scores']['forward_work'] for x in rr)/len(rr),'mean_budget_fraction':sum(x['scores']['budget_fraction'] for x in rr)/len(rr),'both_role_success_mean':sum(v['both_roles'] for v in event[name].values())/12,'by_order':{},'by_role':{}}
 for o in ['subject_first','object_first']:
  part=[x for x in rr if x.get('order')==o]
  if part:summary[name]['by_order'][o]={'n':len(part),'exact':sum(x['scores']['exact'] for x in part),'semantic_proxy':sum(x['scores']['semantic_proxy'] for x in part),'compliant':sum(x['scores']['compliant'] for x in part)}
 for role in ['subject','object']:
  part=[x for x in rr if x['role']==role];summary[name]['by_role'][role]={'n':len(part),'exact':sum(x['scores']['exact'] for x in part),'semantic_proxy':sum(x['scores']['semantic_proxy'] for x in part)}
def ci(values):
 vals=list(values.values());rng=random.Random(20261015);boot=sorted(100*sum(rng.choices(vals,k=12))/12 for _ in range(10000));return {'difference_pp':100*sum(vals)/12,'paired_event_bootstrap_95_pp':[boot[250],boot[9750]],'event_gains':sum(v>0 for v in vals),'event_losses':sum(v<0 for v in vals),'event_ties':sum(v==0 for v in vals),'n_events':12}
def diff(a,b,met='exact'):return ci({eid:event[a][eid][met]-event[b][eid][met] for eid in ix})
contrasts={}
for met in ['exact','semantic_proxy','compliant','both_roles']:
 for base in ['direct','direct_review']:
  contrasts[f'generated_sentence_fresh_minus_{base}/{met}']=diff('generated/sentence1/history0/intact',base,met)
 for history in [0,1]:contrasts[f'sentence_effect_history{history}/{met}']=diff(f'generated/sentence1/history{history}/intact',f'generated/sentence0/history{history}/intact',met)
 for sent in [0,1]:contrasts[f'transcript_effect_sentence{sent}/{met}']=diff(f'generated/sentence{sent}/history1/intact',f'generated/sentence{sent}/history0/intact',met)
 contrasts[f'sentence_x_transcript/{met}']=ci({eid:(event['generated/sentence1/history1/intact'][eid][met]-event['generated/sentence1/history0/intact'][eid][met])-(event['generated/sentence0/history1/intact'][eid][met]-event['generated/sentence0/history0/intact'][eid][met]) for eid in ix})
 for sent in [0,1]:
  for hist in [0,1]:contrasts[f'intact_minus_swapped_sentence{sent}_history{hist}/{met}']=diff(f'generated/sentence{sent}/history{hist}/intact',f'generated/sentence{sent}/history{hist}/swapped',met)
contrasts['reference_intact_minus_swapped/exact']=diff('reference/sentence1/history0/intact','reference/sentence1/history0/swapped')
# Swapped fields: raw answer changes and following behavior; exclude identical tuple interventions.
paired={}
for sent in [False,True]:
 for hist in [False,True]:
  rr=[x for x in scored if x['kind']=='generated' and x['sentence']==sent and x['history']==hist and not x['swap']];ct=collections.Counter();effects={}
  for x in rr:
   y=by[(x['id'],'generated',x['role'],sent,hist,x['order'],True)];nondeg=x['scores']['nondegenerate'];ct['pairs']+=1;ct['nondegenerate_pairs']+=nondeg;ct['answer_changed']+=norm(x['raw'])!=norm(y['raw']);ct['changed_given_nondegenerate']+=nondeg and norm(x['raw'])!=norm(y['raw']);ct['follows_requested_in_both']+=nondeg and x['scores']['follows_requested'] and metric(y)['follows_requested']
  paired[f'sentence{int(sent)}_history{int(hist)}']=dict(ct)
# Exact identities across order, normalized text only: first/last position controls.
order_change={}
for name,rr in cells.items():
 if not name.startswith(('generated/','reference/')):continue
 pairs=[]
 for x in rr:
  if x['order']!='subject_first':continue
  y=by[(x['id'],x['kind'],x['role'],x['sentence'],x['history'],'object_first',x['swap'])];pairs.append(norm(x['raw'])!=norm(y['raw']))
 order_change[name]={'answer_changed':sum(pairs),'pairs':len(pairs)}
# Runtime/token counts are empirical; logical forward work omits batch padding and fused-kernel details.
total={'records':len(rows),'actual_generation_calls':sum(x['finish_reason']!='budget_prefill_failure' for x in rows),'actual_input_tokens':sum(x['input_tokens'] for x in rows),'actual_generated_tokens':sum(x['generated_tokens'] for x in rows),'run_wall_seconds_after_model_load':c['elapsed_seconds'],'note':'CPU batches up to4; recorded run wall includes padding, bookkeeping and identity check. Proxy budgets do not certify exact hardware FLOP equality.','all_within_proxy_budget':all(x['scores']['forward_work']<=budget[x['id']] for x in scored)}
result={'scope':'Exploratory 12 previously examined events from12 source documents, one model; all intervals descriptive without multiplicity control','semantic_scoring_limit':'Original QA-SRL exact span and separately provisional referent-alias correctness, not independent human semantic Agent/Patient validation','cells':summary,'contrasts':contrasts,'swap_pairs':paired,'field_order_changes':order_change,'interface_transitions':transition_counts,'runtime':total,'protocol_sha256':sha(ROOT/'PROTOCOL.json')}
(ROOT/'ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
with (ROOT/'SCORED_ROWS.jsonl').open('w') as f:
 for x in scored:f.write(json.dumps({k:v for k,v in x.items() if k not in ['messages','prompt','input_token_ids','generated_token_ids']})+'\n')
print(json.dumps({'cells':{k:{a:v[a] for a in ['n_outputs','exact','semantic_proxy','compliant','follows_requested','nondegenerate','mean_work_proxy','budget_failure']} for k,v in summary.items()},'main_contrasts':{k:v for k,v in contrasts.items() if k.endswith('/exact')},'runtime':total},indent=2))
