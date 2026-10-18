"""Static fresh-panel preparation; no MLX import, weight load or model forward."""
import json, hashlib, importlib.metadata, shutil
from pathlib import Path
from transformers import AutoTokenizer
from core import HERE, ROOT, MODELS, LENGTH, sha, save, mask
import numpy as np

FAMILIES = [
('c01','Daria','Victor','escorted','escort'),('c02','Leila','Quentin','interviewed','interview'),
('c03','Vera','Nolan','rescued','rescue'),('c04','Sonia','Patrick','arrested','arrest'),
('c05','Tessa','Julian','photographed','photograph'),('c06','Cora','Wesley','invited','invite'),
('c07','Lydia','Ethan','warned','warn'),('c08','Naomi','Adrian','comforted','comfort'),
('c09','Freya','Daniel','contacted','contact'),('c10','Hazel','Martin','questioned','question'),
('c11','June','Rowan','accused','accuse'),('c12','Stella','Gavin','congratulated','congratulate'),
('c13','Nina','Cedric','instructed','instruct'),('c14','Laura','Dennis','defended','defend'),
('c15','Phoebe','Vincent','accompanied','accompany'),('c16','Sofia','Tristan','supported','support'),
('c17','Rina','Kieran','distracted','distract'),('c18','Mira','Dorian','hired','hire'),
('c19','Alina','Jasper','reminded','remind'),('c20','Keira','Hugo','startled','startle'),
('c21','Willa','Silas','challenged','challenge'),('c22','Greta','Milo','blamed','blame'),
('c23','Lila','Tobias','recognized','recognize'),('c24','Esme','Conrad','saluted','salute')]
SYSTEM='Answer the question using only the sentence. Reply with only Yes or No.'
BACKGROUND='Background: The shelf holds a plain folder beside a closed notebook. The wall is white and the room is quiet.\nSentence: '

def main():
 assert not (HERE/'LOCK.json').exists()
 old=json.loads((ROOT/'research/graded_relational_routing_2026-10-18/LOCK.json').read_text())
 prior=[]; known_prompts=set();known_sentences=set()
 def visit(x):
  if isinstance(x,dict):
   if isinstance(x.get('prompt'),str):known_prompts.add(x['prompt'])
   if isinstance(x.get('sentence'),str):known_sentences.add(x['sentence'])
   for v in x.values():visit(v)
  elif isinstance(x,list):
   for v in x:visit(v)
 for p in sorted((ROOT/'research').glob('**/INPUTS.json')):
  if HERE in p.parents:continue
  visit(json.loads(p.read_text()));prior.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
 rows=[];models={};audits=[]
 for model,path in MODELS.items():
  tok=AutoTokenizer.from_pretrained(ROOT/path,local_files_only=True,trust_remote_code=False)
  aliases={k:sorted(set(tok.encode(v,add_special_tokens=False)[0] for v in [k,' '+k])) for k in ['Yes','No']}
  for k in aliases:
   for v in [k,' '+k]:assert len(tok.encode(v,add_special_tokens=False))==1
  pad=tok.encode('x',add_special_tokens=False);assert len(pad)==1
  for n,record in old['models'][model]['files'].items():assert sha(ROOT/n)==record['sha256'],n
  models[model]=old['models'][model]
  for fam,a,b,past,base in FAMILIES:
   for r in [0,1]:
    agent,patient=(a,b) if r==0 else (b,a)
    for voice in ['active','passive']:
     sent=f'{agent} {past} {patient}.' if voice=='active' else f'{patient} was {past} by {agent}.'
     for q in [0,1]:
      subject,obj=(a,b) if q==0 else (b,a);question=f'Did {subject} {base} {obj}?'
      prompt=tok.apply_chat_template([dict(role='system',content=SYSTEM),dict(role='user',content=BACKGROUND+sent+'\nQuestion: '+question)],tokenize=False,add_generation_prompt=True)
      enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];off=enc['offset_mapping'];assert len(ids)<LENGTH
      start=prompt.index(BACKGROUND)+len(BACKGROUND);pc=start+len(sent)-1
      s=next(i for i,(lo,hi) in enumerate(off) if lo<=start<hi);p=next(i for i,(lo,hi) in enumerate(off) if lo<=pc<hi)
      assert tok.decode([ids[p]]).strip()=='.';lo=s-(p-s+1);assert off[lo][0]>=prompt.index(BACKGROUND)
      assert prompt not in known_prompts and sent not in known_sentences,('Not fresh',fam,voice)
      row=dict(id=f'{fam}_r{r}_{voice}_q{q}',model=model,family=fam,split='prospective',role=r,query=q,voice=voice,names=[a,b],agent=agent,patient=patient,past=past,base=base,sentence=sent,question=question,expected='Yes' if r==q else 'No',prompt=prompt,token_ids=ids,offsets=off,pad=pad[0],aliases=aliases,source_start=s,period=p,neutral_span=[lo,s],prompt_end=len(ids)-1)
      for mode in ['closed','frozen','irrelevant']:
       assert np.array_equal(mask(row,mode,14,13),np.tri(LENGTH,dtype=bool))
      assert np.count_nonzero(~mask(row,'closed',14,14))==np.count_nonzero(~mask(row,'irrelevant',14,14))
      rows.append(row)
   rr=[x for x in rows if x['model']==model and x['family']==fam]
   for v in ['active','passive']:
    for q in [0,1]:
     x,y=[x for x in rr if x['voice']==v and x['query']==q]
     assert (x['source_start'],x['period'],x['prompt_end'])==(y['source_start'],y['period'],y['prompt_end'])
     assert x['token_ids'][:x['source_start']]==y['token_ids'][:y['source_start']]
     assert x['token_ids'][x['period']:]==y['token_ids'][y['period']:]
   audits.append(dict(model=model,family=fam,balanced=True,role_pair_positions_match=True,new_complete_prompts=True,new_sentences=True))
 save(HERE/'INPUTS.json',rows)
 save(HERE/'SOURCE_MANIFEST.json',dict(annotation='AI-only deterministic template annotation, not natural corpus or human validation',families=[dict(id=f,names=[a,b],past=p,base=v,agent_patient_unambiguous_by_template=True,uncertainty='No human review; regular transitive use intended. Verbs including accuse, remind, instruct can license omitted complements; identities have no stipulated gender or occupation.') for f,a,b,p,v in FAMILIES],earlier_manifests=prior,known_prompt_count=len(known_prompts),known_sentence_count=len(known_sentences)))
 save(HERE/'TOKENIZER_AUDIT.json',audits)
 from test_pipeline import check
 save(HERE/'PREFLIGHT_TESTS.json',check(rows))
 storage=dict(free_bytes=shutil.disk_usage(HERE).free,expected_max_raw_bytes=900_000_000,expected_max_archive_bytes=900_000_000,reserve_bytes=1024**3)
 assert storage['free_bytes']>sum(storage[k] for k in ['expected_max_raw_bytes','expected_max_archive_bytes','reserve_bytes'])
 save(HERE/'STORAGE_BUDGET.json',storage)
 names=['core.py','engine.py','instrument.py','prepare.py','run.py','analyze.py','test_pipeline.py','PROTOCOL.md','INPUTS.json','SOURCE_MANIFEST.json','TOKENIZER_AUDIT.json','READOUT_CALIBRATION.json','PREFLIGHT_TESTS.json','STORAGE_BUDGET.json']
 save(HERE/'LOCK.json',dict(status='FROZEN_BEFORE_FRESH_OUTCOMES',files={n:sha(HERE/n) for n in names},models=models,native_sources=old['native_sources'],runtime={p:importlib.metadata.version(p) for p in ['mlx','mlx-lm','numpy','transformers']},max_forwards=1562,max_per_model=781,prior_lock_sha256=sha(ROOT/'research/graded_relational_routing_2026-10-18/LOCK.json'),authoritative_panel='INPUTS.json; inherited core.py FAMILIES/jobs/select_prediction are unused',authoritative_protocol='PROTOCOL.md'))
 print('FROZEN:384 tokenized rows; zero model forwards',flush=True)
if __name__=='__main__':main()
