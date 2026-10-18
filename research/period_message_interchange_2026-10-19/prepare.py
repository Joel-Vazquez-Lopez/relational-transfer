"""Static/tokenizer preparation. Never loads an LLM or observes logits."""
from pathlib import Path
import json,re,sysconfig,shutil
from transformers import AutoTokenizer
from core import HERE,ROOT,mask,save,sha

def main():
 assert not (HERE/'LOCK.json').exists()
 old=ROOT/'research/graded_relational_routing_2026-10-18';lock=json.loads((old/'LOCK.json').read_text());model=lock['models']['qwen7b']
 for f,d in model['files'].items():assert sha(ROOT/f)==d['sha256'],f
 tok=AutoTokenizer.from_pretrained(ROOT/model['path'],local_files_only=True,trust_remote_code=False)
 inputs=[r for r in json.loads((old/'INPUTS.json').read_text()) if r['model']=='qwen7b' and r['split']=='development'];rows=list(inputs)
 system='Answer the question using only the sentence. Reply with only Yes or No.'
 background='Background: The shelf holds a plain folder beside a closed notebook. The wall is white and the room is quiet.\nSentence: '
 for family,A,B,past,verb in [('h01','Ruby','Simon','pushed','push'),('h02','Mila','Tobias','hugged','hug')]:
  for role in [0,1]:
   agent,patient=(A,B) if role==0 else (B,A)
   for voice in ['active','passive']:
    sentence=f'{agent} {past} {patient}.' if voice=='active' else f'{patient} was {past} by {agent}.'
    for q in [0,1]:
     subject,obj=(A,B) if q==0 else (B,A);question=f'Did {subject} {verb} {obj}?'
     prompt=tok.apply_chat_template([dict(role='system',content=system),dict(role='user',content=background+sentence+'\nQuestion: '+question)],tokenize=False,add_generation_prompt=True)
     enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids,off=enc['input_ids'],enc['offset_mapping'];assert len(ids)<128
     start=prompt.index(background)+len(background);end=start+len(sentence)-1
     s=next(i for i,(lo,hi) in enumerate(off) if lo<=start<hi);p=next(i for i,(lo,hi) in enumerate(off) if lo<=end<hi)
     assert tok.decode([ids[p]]).strip()=='.'
     rows.append(dict(id=f'{family}_r{role}_{voice}_q{q}',model='qwen7b',family=family,split='evaluation',role=role,voice=voice,query=q,names=[A,B],agent=agent,patient=patient,past=past,base=verb,sentence=sentence,question=question,expected='Yes' if subject==agent else 'No',prompt=prompt,token_ids=ids,offsets=off,pad=inputs[0]['pad'],aliases=inputs[0]['aliases'],source_start=s,period=p,neutral_span=[s-(p-s+1),s],prompt_end=len(ids)-1))
 assert len(rows)==32 and len({r['id'] for r in rows})==32
 checks=[]
 for r in rows:
  assert tok.encode(r['prompt'],add_special_tokens=False)==r['token_ids']
  if r['voice']=='active':a,v,b=re.fullmatch(r'(\w+) (\w+) (\w+)\.',r['sentence']).groups()
  else:b,v,a=re.fullmatch(r'(\w+) was (\w+) by (\w+)\.',r['sentence']).groups()
  sub,v,ob=re.fullmatch(r'Did (\w+) (\w+) (\w+)\?',r['question']).groups()
  assert r['expected']==('Yes' if (a,b)==(sub,ob) else 'No')
  other=next(x for x in rows if x['family']==r['family'] and x['voice']==r['voice'] and x['role']!=r['role'] and x['query']==r['query'])
  assert all(r[k]==other[k] for k in ['source_start','period','prompt_end'])
  assert r['token_ids'][r['period']:]==other['token_ids'][other['period']:]
  for k in ['Yes','No']:
   for i in r['aliases'][k]:assert tok.decode([i]).strip()==k
  f=mask(r,'frozen',14,14);n=mask(r,'neighbor',14,14);p=r['period']
  assert f[p+1:,p].all() and not f[p+1:,p-1].any()
  assert n[p+1:,p-1].all() and not n[p+1:,p].any()
  assert not f[p:,r['source_start']:p].any();assert not n[p-1:,r['source_start']:p-1].any()
  checks.append(dict(id=r['id'],grammar_correct=True,opposite_role_rope_indices_equal=True,period_and_neighbor_routes_open_in_own_graph=True))
 save(HERE/'INPUTS.json',rows);save(HERE/'ANNOTATION_TOKEN_AUDIT.json',checks)
 save(HERE/'READOUT_CALIBRATION.json',json.loads((old/'READOUT_CALIBRATION.json').read_text())['qwen7b'])
 sources=dict(prior_lock=sha(old/'LOCK.json'),review_design=sha(ROOT/'research/gallistel_functional_review_2026-10-19/DESIGN.md'),exposed_development_inputs=16,untouched_evaluation_inputs=16,ai_only_annotation=True,large_model_downloads=0)
 save(HERE/'SOURCE_MANIFEST.json',sources)
 save(HERE/'MODEL_MANIFEST.json',model)
 assert shutil.disk_usage(HERE).free>2.5*1024**3
 print('Prepared32 rows, static annotation and route checks passed; no model forwards')
if __name__=='__main__':main()
