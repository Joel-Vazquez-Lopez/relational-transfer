"""One bounded development-only visible-context experiment, no training or network."""
from pathlib import Path
import os,json,hashlib,re,time,sys,collections,math
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1];OLD=PROJECT/'research/frozen_query_diagnostic_2026-10-15';PREV=PROJECT/'research/frontier_answer_contract_2026-10-15';MODEL=OLD/'local_qwen15'
SYSTEM='Answer the question using only an exact participant phrase from the information provided. Return only the participant phrase. Do not turn the question into a statement or repeat the rest of the question.'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def norm(s):return re.sub(r'\s+',' ',re.sub(r'[^\w\s]',' ',(s or '').casefold())).strip()
def prepare():
 assert not (ROOT/'PROTOCOL.json').exists();items=json.loads((OLD/'ITEMS.json').read_text())['items'];sel=[];seen=set()
 for i in sorted(items,key=lambda i:hashlib.sha256(i['id'].encode()).hexdigest()):
  doc=i['sentence_id'].rsplit(':',2)[0]
  if doc in seen:continue
  seen.add(doc);sel.append(i)
  if len(sel)==12:break
 assert len(sel)==len(seen)==12
 # Provisional referent aliases fixed before new inference, distinct from human source span labels.
 aliases={'assist':{'subject':['The French-based airframer']},'pressed':{'object':['K']},'seen':{'object':['Hayley']},'accused':{'subject':['India and Pakistan','Pakistan and India']},'invited':{'object':['people']}}
 for i in sel:
  i['document_id']=i['sentence_id'].rsplit(':',2)[0]
  i['referent_aliases']={role:list(dict.fromkeys(i[role+'_accepted']+aliases.get(i['verb'],{}).get(role,[]))) for role in ['subject','object']}
  i['semantic_status']='Original human QA-SRL target spans; additional referent aliases are AI-provisional, NOT independent human semantic-role validation.'
  i['semantic_target']='Predicate-specific participant requested by original question; no subject=Agent or object=Patient assumption.'
 (ROOT/'ITEMS.json').write_text(json.dumps(sel,indent=2)+'\n')
 sources=[OLD/x for x in ['ITEMS.json','PREDICTIONS_1.5b.jsonl','FIELD_ONLY_1.5b.jsonl','PROTOCOL.json']]+[PREV/x for x in ['ITEMS.json','PREDICTIONS.jsonl','PROTOCOL.json']]
 p={'status':'EXPLORATORY DEVELOPMENT fixed before new inference','date':'2026-10-15','items':12,'documents':12,'selection':'SHA256 event-ID ascending, first event per source document from previously examined 24; no prediction-dependent selection','max_new_generations':540,'scope':'Untouched Qwen2.5-1.5B only; no training, external credits, downloads or sealed data. All variants dependent within event.','design':{'generated_fields':'2 sentence-present x 2 transcript-packaging x 2 field-order x 2 intact/swapped x 2 questions x 12 =384','reference_fields':'fresh sentence-present only x 2 orders x 2 intact/swapped x 2 questions x 12 =96; oracle diagnostic, not autonomous pipeline','direct':'12 x 2 questions x one-pass and self-review second pass =48 new calls; review reuses first answer','extraction_replay':'12 cached no-question extractions replayed to record exact token cost; values held at archived generated fields throughout'},'interface':'Same system and Answer: prefill for all readout and QA arms. Questions unchanged; both roles and field orders crossed. Transcript changes visible tokens/turn boundaries, not an unobserved generation-vs-supply state.','primary':['Fresh sentence-present generated fields versus strong one-pass and two-pass QA, averaging field order within event; both accuracy and cost','Requested-field following contrast intact versus swapped, and sentence/transcript effects holding field text and order fixed'],'secondary':['Sentence x transcript risk contrast; role/order effects; oracle-field readout to diagnose query interpretation; token-identical full versus split-cache next-logit technical control'],'scoring':{'source_span':'Frozen accepted original QA-SRL span exact match','semantic_proxy':'Referent-alias exact match ignoring punctuation, case and English articles. New aliases fixed in ITEMS; provisional AI judgment, not independent human semantic validation. No correctness credit merely for mentioning an answer somewhere.','compliance':'Output is one normalized contiguous phrase from available sentence/fields; answer-label repetition, empty/multiple clauses and budget failure separately counted. Compliance does not imply correctness.','field_following':'Answer equals supplied requested field; also record opposite field and first/last-field following separately. Correctness evaluated independently; following a wrong field is not success.'},'statistics':'Paired event bootstrap n=12,10000 resamples seed20261015; average orders/questions within event. Report gains/losses, all cells, intervals; descriptive only, no post hoc success gate or multiplicity-certified population claim.','compute':{'primary':'Common per-event cold-query forward-work-proxy ceiling: replayed extraction cost plus maximum field-only fresh intact generated readout at 48 output tokens. All readout/oracle arms charged full extraction; D2 charged actual D1+review.','proxy':'Architecture matmul/attention FLOP approximation; not exact measured hardware FLOPs. F(L,g)=2W(L+g-1)+2*d*V*g+4*N*d*(L^2+sum_{t=1}^{g-1}(L+t)); W=N*(2d^2+2d*kv+3d*m).','max_decode':48,'enforcement':'Before each generation choose largest g<=48 fitting remaining proxy budget. Cannot fit prefill+one token => scored budget failure, no generation. Actual token counts/time also logged. Equal budget ceiling is not equal consumed computation.','replay_token_budget':48},'generation':{'device':'cpu','dtype':'float32','threads':2,'batch_size':4,'greedy':True,'seed':910215},'source_hashes':{str(p.relative_to(PROJECT)):sha(p) for p in sources},'script_sha256':sha(Path(__file__)),'items_sha256':sha(ROOT/'ITEMS.json'),'weights_sha256':sha(MODEL/'model.safetensors'),'config_sha256':sha(MODEL/'config.json'),'tokenizer_sha256':sha(MODEL/'tokenizer.json'),'limits':['12 reused events and one compact checkpoint; no claim of unseen semantic generalization','Correct reference fields use source labels and do not validate semantic ontology or make an autonomous method','Transcript packaging changes token positions, length and conversational attribution; effects are visible-context package effects','Budget is approximate architecture-forward work, not certified exact hardware-compute matching','No adaptive changes or new inference after inspecting scores']}
 (ROOT/'PROTOCOL.json').write_text(json.dumps(p,indent=2)+'\n');print('FROZEN',sha(ROOT/'PROTOCOL.json'),flush=True)
def run():
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 cfg=json.loads((ROOT/'PROTOCOL.json').read_text());assert cfg['script_sha256']==sha(Path(__file__));assert cfg['items_sha256']==sha(ROOT/'ITEMS.json')
 for rel,h in cfg['source_hashes'].items():assert sha(PROJECT/rel)==h
 assert sha(MODEL/'model.safetensors')==cfg['weights_sha256'];torch.set_num_threads(2);torch.manual_seed(910215)
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.padding_side='left'
 model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.float32).eval();model.requires_grad_(False)
 items=json.loads((ROOT/'ITEMS.json').read_text());ix={i['id']:i for i in items};old=[json.loads(x) for x in (OLD/'PREDICTIONS_1.5b.jsonl').read_text().splitlines()];blind={x['id']:x for x in old if x['arm']=='query_blind'}
 d=model.config.hidden_size;layers=model.config.num_hidden_layers;kv=model.config.num_key_value_heads*(d//model.config.num_attention_heads);inter=model.config.intermediate_size;vocab=model.config.vocab_size;W=layers*(2*d*d+2*d*kv+3*d*inter)
 def work(L,g):return 2*W*(L+g-1)+2*d*vocab*g+4*layers*d*(L*L+(g-1)*L+g*(g-1)/2)
 (ROOT/'COMPUTE_PROXY.json').write_text(json.dumps({'hidden':d,'layers':layers,'kv_width':kv,'intermediate':inter,'vocab':vocab,'dense_layer_weights':W,'scope':'Approximate inference matmul/attention work, NOT actual measured FLOPs'},indent=2)+'\n')
 def makeprompt(msg):return tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)+'Answer:'
 def task(i,kind,role,sentence=True,history=False,order='subject_first',swap=False):
  fs={k:(i[k] if kind=='reference' else blind[i['id']]['fields'].get(k,'')) for k in ['subject','object']}
  if swap:fs={'subject':fs['object'],'object':fs['subject']}
  kk=['subject','object'] if order=='subject_first' else ['object','subject'];text='; '.join(k.title()+': '+fs[k] for k in kk)
  context=(f"Sentence: {i['sentence']}\n" if sentence else '')+f"Target verb: {i['verb']} (token index {i['predicate_index']}, counting from zero)"
  if history:
   extract='Identify the grammatical subject and direct object of the target verb in this active sentence. Use exact phrases from the sentence. Give exactly: '+('; '.join(k.title()+': SPAN' for k in kk))+'.'
   msgs=[{'role':'system','content':SYSTEM},{'role':'user','content':context+'\n'+extract},{'role':'assistant','content':text},{'role':'user','content':'Question: '+i[role+'_question']}]
  else:msgs=[{'role':'system','content':SYSTEM},{'role':'user','content':context+'\nFields: '+text+'\nQuestion: '+i[role+'_question']}]
  return {'id':i['id'],'kind':kind,'role':role,'sentence':sentence,'history':history,'order':order,'swap':swap,'supplied_fields':fs,'messages':msgs,'prompt':makeprompt(msgs)}
 path=ROOT/'PREDICTIONS.jsonl';records={}
 def key(t):return (t['id'],t['kind'],t.get('role'),t.get('sentence'),t.get('history'),t.get('order'),t.get('swap'))
 if path.exists():
  for line in path.read_text().splitlines():
   x=json.loads(line);assert x['protocol_sha256']==sha(ROOT/'PROTOCOL.json');assert key(x) not in records;records[key(x)]=x
 start=time.perf_counter()
 def generate(ts):
  todo=[t for t in ts if key(t) not in records];buckets=collections.defaultdict(list)
  for t in todo:
   L=len(tok.encode(t['prompt'],add_special_tokens=False));t['input_length']=L;available=t.get('budget',float('inf'))-t.get('prior_work',0);g=next((g for g in range(48,0,-1) if work(L,g)<=available),0);t['max_new_tokens']=g;buckets[g].append(t)
  for g,group in sorted(buckets.items()):
   for off in range(0,len(group),4):
    block=group[off:off+4]
    if not g:
     with path.open('a') as f:
      for t in block:
       x={**t,'raw':'','generated_token_ids':[],'input_token_ids':tok.encode(t['prompt'],add_special_tokens=False),'input_tokens':t['input_length'],'generated_tokens':0,'finish_reason':'budget_prefill_failure','forward_work':0,'batch_seconds':0,'protocol_sha256':sha(ROOT/'PROTOCOL.json')};f.write(json.dumps(x)+'\n');records[key(x)]=x
     continue
    enc=tok([t['prompt'] for t in block],padding=True,return_tensors='pt',add_special_tokens=False);t0=time.perf_counter();gen=model.generate(**enc,max_new_tokens=g,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True,logits_to_keep=1);dt=time.perf_counter()-t0
    with path.open('a') as f:
     for j,(t,output) in enumerate(zip(block,gen[:,enc['input_ids'].shape[1]:])):
      ids=output.tolist()
      if tok.eos_token_id in ids:ids=ids[:ids.index(tok.eos_token_id)+1]
      x={**t,'raw':tok.decode(ids,skip_special_tokens=True),'generated_token_ids':ids,'input_token_ids':enc['input_ids'][j][enc['attention_mask'][j].bool()].tolist(),'input_tokens':int(enc['attention_mask'][j].sum()),'generated_tokens':len(ids),'finish_reason':'eos' if ids[-1]==tok.eos_token_id else 'budget_or_length','forward_work':work(t['input_length'],len(ids)),'batch_seconds':dt,'protocol_sha256':sha(ROOT/'PROTOCOL.json')};assert x['forward_work']+x.get('prior_work',0)<=x.get('budget',float('inf'));f.write(json.dumps(x)+'\n');f.flush();records[key(x)]=x
    print('records',len(records),'/540','seconds',round(time.perf_counter()-start,1),flush=True)
 with torch.inference_mode():
  # Replay identical old extraction prompts. These are bookkeeping, not a new extractor.
  generate([{'id':i['id'],'kind':'extraction_replay','role':'both','prompt':blind[i['id']]['prompt']} for i in items])
  replay={i['id']:next(x for x in records.values() if x['id']==i['id'] and x['kind']=='extraction_replay') for i in items}
  assert all(replay[i['id']]['raw']==blind[i['id']]['raw'] for i in items),'Extraction replay mismatch; STOP before comparisons'
  budget={}
  for i in items:
   pr=[task(i,'generated',role,False,False,order,False)['prompt'] for role in ['subject','object'] for order in ['subject_first','object_first']]
   budget[i['id']]=replay[i['id']]['forward_work']+max(work(len(tok.encode(p,add_special_tokens=False)),48) for p in pr)
  (ROOT/'BUDGETS.json').write_text(json.dumps({'method':'Common per-event cold-query approximate forward-work ceiling, fixed from replayed extraction and preplanned fresh field-only prompts','budgets':budget},indent=2)+'\n')
  ts=[]
  for i in items:
   for sentence in [False,True]:
    for history in [False,True]:
     for order in ['subject_first','object_first']:
      for swap in [False,True]:
       for role in ['subject','object']:
        t=task(i,'generated',role,sentence,history,order,swap);t.update(budget=budget[i['id']],prior_work=replay[i['id']]['forward_work']);ts.append(t)
   for order in ['subject_first','object_first']:
    for swap in [False,True]:
     for role in ['subject','object']:
      t=task(i,'reference',role,True,False,order,swap);t.update(budget=budget[i['id']],prior_work=replay[i['id']]['forward_work']);ts.append(t)
   for role in ['subject','object']:
    user=f"Sentence: {i['sentence']}\nTarget verb: {i['verb']} (token index {i['predicate_index']}, counting from zero)\nQuestion: {i[role+'_question']}";msg=[{'role':'system','content':SYSTEM},{'role':'user','content':user}];ts.append({'id':i['id'],'kind':'direct','role':role,'messages':msg,'prompt':makeprompt(msg),'budget':budget[i['id']],'prior_work':0})
  assert len(ts)==504
  generate(ts)
  reviews=[]
  for i in items:
   for role in ['subject','object']:
    first=next(x for x in records.values() if x['id']==i['id'] and x['kind']=='direct' and x['role']==role)
    msg=first['messages']+[{'role':'assistant','content':'Answer:'+first['raw']},{'role':'user','content':'Review that answer against the sentence and the original question. Correct it if needed. Return only the participant phrase.'}]
    reviews.append({'id':i['id'],'kind':'direct_review','role':role,'messages':msg,'prompt':makeprompt(msg),'budget':budget[i['id']],'prior_work':first['forward_work'],'first_answer':first['raw']})
  generate(reviews)
  # A token-identical recomputation control: full versus split cached prefill.
  pr=next(x['prompt'] for x in records.values() if x['kind']=='generated' and x['history'] and x['sentence'] and not x['swap']);ids=tok(pr,return_tensors='pt',add_special_tokens=False)['input_ids'];mid=ids.shape[1]//2
  a=model(ids,use_cache=False,logits_to_keep=1).logits[:,-1];cached=model(ids[:,:mid],use_cache=True,logits_to_keep=1);b=model(ids[:,mid:],past_key_values=cached.past_key_values,use_cache=True,logits_to_keep=1).logits[:,-1];diff=float((a-b).abs().max());same=int(a.argmax(-1))==int(b.argmax(-1));assert same and diff<1e-3
  (ROOT/'IDENTICAL_TOKEN_CONTROL.json').write_text(json.dumps({'identical_input_token_ids':ids[0].tolist(),'full_vs_split_cached_max_abs_logit_difference':diff,'same_next_token':same,'tolerance':1e-3,'claim':'Technical recomputation check only; packaging conditions have different visible token sequences.'},indent=2)+'\n')
 assert len(records)==540,len(records)
 for rel,h in cfg['source_hashes'].items():assert sha(PROJECT/rel)==h
 (ROOT/'COMPLETE.json').write_text(json.dumps({'records':len(records),'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'predictions_sha256':sha(path),'elapsed_seconds':time.perf_counter()-start,'extraction_replays_identical':12,'new_training':False,'sealed_data':False,'compute_match':'common approximate forward-work ceiling, not equal consumed computation or exact hardware FLOPs'},indent=2)+'\n');print('COMPLETE',flush=True)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='run':run()
 else:raise SystemExit('prepare or run')
