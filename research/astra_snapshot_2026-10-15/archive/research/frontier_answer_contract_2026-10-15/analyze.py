"""Exploratory paired factorial analysis; no significance-based next stage."""
from pathlib import Path
import json,re,random,collections,hashlib
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'frozen_query_diagnostic_2026-10-15'
def norm(s):return re.sub(r'\s+',' ',re.sub(r'[^\w\s]',' ',(s or '').casefold())).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads((ROOT/'COMPLETE.json').read_text());assert c['prediction_sha256']==sha(ROOT/'PREDICTIONS.jsonl');assert c['likelihood_sha256']==sha(ROOT/'CANDIDATE_LIKELIHOODS.jsonl')
items=json.loads((ROOT/'ITEMS.json').read_text());idx={x['id']:x for x in items};rows=[json.loads(x) for x in (ROOT/'PREDICTIONS.jsonl').read_text().splitlines()];assert len(rows)==192
scores={};cells={};oldrows=[json.loads(x) for x in (OLD/'PREDICTIONS_1.5b.jsonl').read_text().splitlines()];oldidx={(x['id'],x['arm'],x['role']):x for x in oldrows};replay=[]
for x in rows:
 key=x['id'],x['instruction'],x['prefix'],x['role'];assert key not in scores;good=norm(x['raw']) in {norm(a) for a in idx[x['id']][x['role']+'_accepted']};scores[key]=int(good)
 if x['instruction']=='original' and x['prefix']=='':
  old=oldidx[x['id'],'direct',x['role']];assert old['prompt']==x['prompt'];replay.append({'id':x['id'],'role':x['role'],'raw_identical':x['raw']==old['raw'],'correct_identical':good==(norm(old['raw']) in {norm(a) for a in idx[x['id']][x['role']+'_accepted']})})
for instruction in ['original','explicit_span_contract']:
 for prefix in ['', 'Answer:']:
  cells[instruction+'/'+(prefix or 'none')]={role:sum(scores[i['id'],instruction,prefix,role] for i in items) for role in ['subject','object']};cells[instruction+'/'+(prefix or 'none')]['joint']=sum(all(scores[i['id'],instruction,prefix,role] for role in ['subject','object']) for i in items)
def contrast(i,kind):
 def s(a,b,role):return scores[i['id'],a,b,role]
 if kind=='instruction_x_prefix':return sum(s('explicit_span_contract','Answer:',role)-s('explicit_span_contract','',role)-s('original','Answer:',role)+s('original','',role) for role in ['subject','object'])/2
 if kind=='instruction_x_role':return sum(s('explicit_span_contract',p,'subject')-s('original',p,'subject')-s('explicit_span_contract',p,'object')+s('original',p,'object') for p in ['', 'Answer:'])/2
 if kind=='prefix_x_role':return sum(s(a,'Answer:','subject')-s(a,'','subject')-s(a,'Answer:','object')+s(a,'','object') for a in ['original','explicit_span_contract'])/2
 if kind=='instruction_subject_no_prefix':return s('explicit_span_contract','','subject')-s('original','','subject')
 if kind=='instruction_object_no_prefix':return s('explicit_span_contract','','object')-s('original','','object')
 if kind=='prefix_subject_original':return s('original','Answer:','subject')-s('original','','subject')
 if kind=='prefix_object_original':return s('original','Answer:','object')-s('original','','object')
 raise ValueError(kind)
groups=collections.defaultdict(list)
for i in items:groups[i['sentence_id'].rsplit(':',2)[0]].append(i)
def interval(values):
 rng=random.Random(20261015);keys=sorted(groups);boot=[]
 for _ in range(10000):
  chosen=[i for k in rng.choices(keys,k=len(keys)) for i in groups[k]];boot.append(100*sum(values[i['id']] for i in chosen)/len(chosen))
 boot.sort();return {'difference_pp':100*sum(values.values())/len(values),'paired_document_bootstrap_95_pp':[boot[250],boot[9750]],'source_documents':len(keys),'scope':'exploratory descriptive interval on reused development examples; no multiplicity correction or population confirmation'}
contrasts={k:interval({i['id']:contrast(i,k) for i in items}) for k in ['instruction_x_prefix','instruction_x_role','prefix_x_role','instruction_subject_no_prefix','instruction_object_no_prefix','prefix_subject_original','prefix_object_original']}
lik=[json.loads(x) for x in (ROOT/'CANDIDATE_LIKELIHOODS.jsonl').read_text().splitlines()];lc={}
for metric in ['logprob_sum','logprob_mean','first_token_logprob']:
 lc[metric]={}
 for role in ['subject','object']:
  sub=[x for x in lik if x['role']==role];correct=lambda x:x['scores'][role][metric]>x['scores']['object' if role=='subject' else 'subject'][metric];wronggen=[x for x in sub if not scores[x['id'],'original','',role]];lc[metric][role]={'correct':sum(correct(x) for x in sub),'n':len(sub),'correct_among_generation_wrong':sum(correct(x) for x in wronggen),'generation_wrong':len(wronggen)}
overlap={}
for role in ['subject','object']:
    gain=lambda inst,pr:{i['id'] for i in items if scores[i['id'],inst,pr,role] and not scores[i['id'],'original','',role]}
    aa=gain('original','Answer:');bb=gain('explicit_span_contract','')
    overlap[role]={'prefix_gains':len(aa),'explicit_gains':len(bb),'common_gains':len(aa&bb),'equal_correctness_between_single_cues':sum(scores[i['id'],'original','Answer:',role]==scores[i['id'],'explicit_span_contract','',role] for i in items)}
residual={}
for inst,pr in [('original',''),('original','Answer:'),('explicit_span_contract',''),('explicit_span_contract','Answer:')]:
    values={}
    for i in items:
        b=oldidx[i['id'],'query_blind','both']['fields'];joint=int(all(norm(b.get(k,'')) in {norm(v) for v in i[k+'_accepted']} for k in ['subject','object']));values[i['id']]=joint-int(all(scores[i['id'],inst,pr,k] for k in ['subject','object']))
    residual[inst+'/'+(pr or 'none')]=interval(values)
info={'single_cue_overlap':overlap,'blind_route_minus_direct_joint':residual,'scope':'New untouched 1.5B development-only factorial; no training; same examined active grammatical examples, not verified semantic roles','cells_out_of_24':cells,'contrasts':contrasts,'historical_baseline_replay':{'raw_identical':sum(x['raw_identical'] for x in replay),'correctness_identical':sum(x['correct_identical'] for x in replay),'n':48},'oracle_candidate_scoring':lc,'actual_token_cost':{'input_tokens':sum(x['input_tokens'] for x in rows),'output_tokens':sum(x['generated_tokens'] for x in rows),'calls':len(rows),'generation_wall_seconds':sum(x['batch_seconds'] for x in rows)/4,'teacher_forcing_excluded_from_generation_cost':True,'length_terminations':sum(x['finish_reason']=='length' for x in rows)},'protocol_sha256':sha(ROOT/'PROTOCOL.json')}
(ROOT/'ANALYSIS.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
