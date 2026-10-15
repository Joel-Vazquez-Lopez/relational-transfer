"""Bounded untouched-Qwen development probe; never reads reserved data."""
from pathlib import Path
import json,hashlib,re,time,sys,os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OLD=PROJECT/'research/frozen_query_diagnostic_2026-10-15'
MODEL=OLD/'local_qwen15'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def norm(s):return re.sub(r'\s+',' ',re.sub(r'[^\w\s]',' ',s.casefold())).strip()
def prepare():
 assert not (ROOT/'PROTOCOL.json').exists()
 source=[OLD/x for x in ['ITEMS.json','PROTOCOL.json','PREDICTIONS_1.5b.jsonl','FIELD_ONLY_1.5b.jsonl']]
 items=json.loads((OLD/'ITEMS.json').read_text())['items'];assert len(items)==24
 (ROOT/'ITEMS.json').write_text(json.dumps(items,indent=2)+'\n')
 cfg={'status':'exploratory development fixed before new inference','scope':'All 24 previously inspected active QA-SRL events; grammatical subject/object, NOT verified semantic roles. No sealed data, no optimization.','model':'Qwen2.5-1.5B-Instruct unchanged local weights','generation_count_cap':192,'teacher_forced_candidate_pairs':48,'factorial':{'instruction':['original','explicit_span_contract'],'assistant_prefill':['none','Answer:'],'question_role':['subject','object']},'prediction':'If output contract explains the old gain, explicit participant-only instruction and/or Answer prefix improve direct subject accuracy more than object accuracy and narrow the blind-routing discrepancy. Failure to improve weakens this account. Prompt gains alone do not isolate internal representation.','primary_contrast':'Within-event instruction x requested-role risk difference averaged across prefixes, plus every factorial cell. No significance-based selection or progression.','secondary':'Prefix x role; joint answers per event; label-based content diagnostics, exact span F1; constrained two-gold-participant sequence likelihood as oracle diagnostic only; historical raw baseline reproducibility.','limits':['One small checkpoint and reused 24 events; no new population or semantic generalization claim.','Different instructions change attention as well as format; no exclusive mechanism identification.','Likelihood candidate set uses gold spans and is not autonomous QA; token normalization may alter choice.','No tuning after partial outputs.'],'generation':{'device':'cpu','dtype':'float32','threads':2,'batch_size':4,'max_new_tokens':48,'do_sample':False,'seed':910215},'source_hashes':{str(p.relative_to(PROJECT)):sha(p) for p in source},'script_sha256':sha(Path(__file__)),'items_sha256':sha(ROOT/'ITEMS.json'),'weights_sha256':sha(MODEL/'model.safetensors'),'config_sha256':sha(MODEL/'config.json'),'tokenizer_sha256':sha(MODEL/'tokenizer.json')}
 (ROOT/'PROTOCOL.json').write_text(json.dumps(cfg,indent=2)+'\n');print('FROZEN',sha(ROOT/'PROTOCOL.json'),flush=True)
def run():
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 c=json.loads((ROOT/'PROTOCOL.json').read_text());assert c['script_sha256']==sha(Path(__file__));assert c['items_sha256']==sha(ROOT/'ITEMS.json')
 for rel,h in c['source_hashes'].items():assert sha(PROJECT/rel)==h
 assert sha(MODEL/'model.safetensors')==c['weights_sha256'];assert sha(MODEL/'config.json')==c['config_sha256']
 torch.set_num_threads(2);torch.manual_seed(910215)
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.padding_side='left'
 model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.float32).eval();model.requires_grad_(False)
 items=json.loads((ROOT/'ITEMS.json').read_text());tasks=[]
 for i in items:
  for instruction in ['original','explicit_span_contract']:
   for prefix in ['','Answer:']:
    for role in ['subject','object']:
     system='Answer the question with only an exact phrase from the sentence.'
     if instruction!='original':system+=' Return only the participant phrase. Do not turn the question into a statement or repeat the rest of the question.'
     user=f"Sentence: {i['sentence']}\nTarget verb: {i['verb']} (token index {i['predicate_index']}, counting from zero)\nQuestion: {i[role+'_question']}"
     prompt=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)+prefix
     tasks.append({'id':i['id'],'instruction':instruction,'prefix':prefix,'role':role,'prompt':prompt})
 assert len(tasks)==192
 path=ROOT/'PREDICTIONS.jsonl';key=lambda x:(x['id'],x['instruction'],x['prefix'],x['role']);done={}
 if path.exists():
  for line in path.read_text().splitlines():
   row=json.loads(line);assert row['protocol_sha256']==sha(ROOT/'PROTOCOL.json');assert key(row) not in done;done[key(row)]=row
 todo=[x for x in tasks if key(x) not in done];start=time.perf_counter()
 with torch.inference_mode():
  for off in range(0,len(todo),4):
   block=todo[off:off+4];enc=tok([x['prompt'] for x in block],return_tensors='pt',padding=True,add_special_tokens=False);t0=time.perf_counter()
   gen=model.generate(**enc,max_new_tokens=48,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
   dt=time.perf_counter()-t0
   with path.open('a') as f:
    for j,(task,output) in enumerate(zip(block,gen[:,enc['input_ids'].shape[1]:])):
     ids=output.tolist()
     if tok.eos_token_id in ids:ids=ids[:ids.index(tok.eos_token_id)+1]
     rec={**task,'raw':tok.decode(ids,skip_special_tokens=True),'input_token_ids':enc['input_ids'][j][enc['attention_mask'][j].bool()].tolist(),'generated_token_ids':ids,'input_tokens':int(enc['attention_mask'][j].sum()),'generated_tokens':len(ids),'finish_reason':'eos' if ids[-1]==tok.eos_token_id else 'length','batch_seconds':dt,'protocol_sha256':sha(ROOT/'PROTOCOL.json')};f.write(json.dumps(rec)+'\n');f.flush()
   print('completed',len(done)+min(off+4,len(todo)),'/192','elapsed',round(time.perf_counter()-start,1),flush=True)
  # Score the same exact direct prompts under the two participant alternatives.
  likelihood_path=ROOT/'CANDIDATE_LIKELIHOODS.jsonl';prior=[]
  if likelihood_path.exists():prior=[json.loads(x) for x in likelihood_path.read_text().splitlines()]
  seen={(x['id'],x['role']) for x in prior}
  for task in tasks:
   if task['instruction']!='original' or task['prefix'] or (task['id'],task['role']) in seen:continue
   item=next(i for i in items if i['id']==task['id']);prompt_ids=tok.encode(task['prompt'],add_special_tokens=False);scores={}
   for candidate_role in ['subject','object']:
    text=item[candidate_role];ids=tok.encode(text,add_special_tokens=False);full=torch.tensor([prompt_ids+ids]);out=model(full).logits[0,len(prompt_ids)-1:-1].log_softmax(-1);lp=out[torch.arange(len(ids)),torch.tensor(ids)];scores[candidate_role]={'candidate':text,'tokens':len(ids),'logprob_sum':float(lp.sum()),'logprob_mean':float(lp.mean()),'first_token_logprob':float(lp[0])}
   with likelihood_path.open('a') as f:f.write(json.dumps({'id':task['id'],'role':task['role'],'scores':scores,'protocol_sha256':sha(ROOT/'PROTOCOL.json')})+'\n')
 assert len(path.read_text().splitlines())==192
 assert len(likelihood_path.read_text().splitlines())==48
 for rel,h in c['source_hashes'].items():assert sha(PROJECT/rel)==h
 (ROOT/'COMPLETE.json').write_text(json.dumps({'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'prediction_sha256':sha(path),'likelihood_sha256':sha(likelihood_path),'records':192,'candidate_pairs':48,'elapsed_seconds':time.perf_counter()-start,'new_training':False,'sealed_data':False},indent=2)+'\n')
 print('COMPLETE',flush=True)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='run':run()
 else:raise SystemExit('prepare or run')
