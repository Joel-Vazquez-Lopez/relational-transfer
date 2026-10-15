"""Resumable frozen pretrained diagnostic. Development only; no optimization."""
import argparse, collections, gzip, hashlib, json, re, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
MODELS={
 '0.5b':Path('/Users/joelvazquezlopez/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'),
 '1.5b':ROOT/'local_qwen15'}
SOURCE=PROJECT/'llm_role_natural_repair/data/qasrl-v2/orig/dev.jsonl.gz'
SPLIT=PROJECT/'research/candidate_free_role_binding_2026-10-14/SPLITS.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def norm(s):return re.sub(r'\s+',' ',re.sub(r'[^\w\s]',' ',(s or '').casefold())).strip()
def accepted(q):
 spans=collections.Counter(tuple(s) for j in q['answerJudgments'] if j['isValid'] for s in j['spans'])
 return sorted([s for s,n in spans.items() if n>=2])
def prepare():
 assert not (ROOT/'PROTOCOL.json').exists()
 previous=json.loads(SPLIT.read_text())['dev']; ids={r['sentence_id'] for r in previous}; raw={}
 with gzip.open(SOURCE,'rt') as f:
  for line in f:
   r=json.loads(line)
   if r['sentenceId'] in ids:raw[r['sentenceId']]=r
 eligible=[]; excluded=[]
 for item in previous:
  rec=raw[item['sentence_id']];vi=item['id'].split(':')[-1];v=rec['verbEntries'][vi];qs=v['questionLabels'];a=qs[item['actor_question']];p=qs[item['patient_question']]
  valid=all(not q['isPassive'] and not q['isNegated'] and q['questionSlots']['prep']=='_' and q['questionSlots']['obj2']=='_' for q in [a,p])
  valid=valid and a['questionSlots']['subj']=='_' and a['questionSlots']['obj']=='someone' and p['questionSlots']['subj']=='someone' and p['questionSlots']['obj']=='_'
  if not valid:excluded.append({'id':item['id'],'reason':'not a simple original active subject/direct-object question pair'});continue
  x={**item,'subject':item['actor'],'object':item['patient'],'subject_question':item['actor_question'],'object_question':item['patient_question'],'predicate_index':int(vi)}
  for key,q in [('subject',a),('object',p)]:
   spans=accepted(q); assert spans
   x[key+'_accepted']=[' '.join(rec['sentenceTokens'][lo:hi]) for lo,hi in spans]
   assert norm(x[key]) in {norm(s) for s in x[key+'_accepted']}
  eligible.append(x)
 assert len(eligible)>=24,len(eligible)
 data={'items':eligible[:24],'eligible_count':len(eligible),'excluded':excluded,'selection':'First 24 eligible in previously frozen, hash-ordered development list. No model predictions used.'}
 (ROOT/'ITEMS.json').write_text(json.dumps(data,indent=2)+'\n')
 cfg={'date':'2026-10-15','status':'frozen_before_diagnostic_inference','scope':'Exploratory development. Original 100-item source already examined. No sealed data. No training.',
 'question':'Do untouched models show question-dependent grammatical participant outputs, and does question-independent extraction improve original QA answers?',
 'labels':'Grammatical active subject and direct object from original QA-SRL question slots, NOT verified semantic agent/patient. Consensus original spans, alternate consensus spans accepted.',
 'arms':['direct','query_visible','query_blind'],'items':24,'per_model_generations':120,
 'design':'Direct and query_visible receive each original question plus target verb. query_blind extracts subject/object once with sentence and target verb, then a gold question-role selector deterministically reads each field. Selector is oracle diagnostic; question interpretation and extra output differ, so this is not a fair end-to-end intervention claim.',
 'primary':'Both original QA answers exact per event; compare direct, query-visible final answers, and query-blind deterministic selection. Descriptive paired event bootstrap intervals, no selection or significance gate.',
 'secondary':'Accepted-span token F1, requested field exact, conditional answer-use failures, visible role-tuple changes across questions, parse/termination failures. Fields are printed outputs, not hidden states.',
 'limits':['All selected events active and actor-first; no construction generalization claim.','One pretrained family, two sizes, no independent human semantic review.','Question changes role and wording jointly; no isolated causal query mechanism.','Blind extraction has oracle role routing and different instruction; its invariance is guaranteed by reuse, not learned.','No model selection, prompt tuning or training will follow partial scores.'],
 'generation':{'device':'cpu','dtype':'float32','threads':2,'batch_size':4,'max_new_tokens':48,'do_sample':False},
 'hashes':{'source':sha(SOURCE),'previous_split':sha(SPLIT),'items':sha(ROOT/'ITEMS.json'),'script':sha(Path(__file__))},
 'models':{k:{'path':str(p),'weights_sha256':sha(p/'model.safetensors'),'config_sha256':sha(p/'config.json'),'tokenizer_sha256':sha(p/'tokenizer.json')} for k,p in MODELS.items()}}
 (ROOT/'PROTOCOL.json').write_text(json.dumps(cfg,indent=2)+'\n');print('FROZEN',sha(ROOT/'PROTOCOL.json'),'eligible',len(eligible),flush=True)
def validate():
 c=json.loads((ROOT/'PROTOCOL.json').read_text());assert c['hashes']['script']==sha(Path(__file__));assert c['hashes']['items']==sha(ROOT/'ITEMS.json');return c
def tasks(items):
 out=[]
 for item in items:
  for arm in ['direct','query_visible','query_blind']:
   for role in (['subject','object'] if arm!='query_blind' else ['both']):out.append({'id':item['id'],'arm':arm,'role':role,'item':item})
 return out
def prompt(tok,t):
 i=t['item'];u=f"Sentence: {i['sentence']}\nTarget verb: {i['verb']} (token index {i['predicate_index']}, counting from zero)"
 if t['arm']!='query_blind':u+='\nQuestion: '+i[t['role']+'_question']
 if t['arm']=='direct':system='Answer the question with only an exact phrase from the sentence.';prefix=''
 else:
  system='Identify the grammatical subject and direct object of the target verb in this active sentence. Use exact phrases from the sentence. '
  system+=('Give exactly: Subject: SPAN; Object: SPAN; Answer: SPAN. Answer the question in the Answer field.' if t['arm']=='query_visible' else 'Give exactly: Subject: SPAN; Object: SPAN.')
  prefix='Subject:'
 return tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':u}],tokenize=False,add_generation_prompt=True)+prefix,prefix
@torch.inference_mode()
def run(key,smoke=False):
 c=validate();p=MODELS[key];assert sha(p/'model.safetensors')==c['models'][key]['weights_sha256']
 torch.set_num_threads(2);torch.manual_seed(910215)
 tok=AutoTokenizer.from_pretrained(p,local_files_only=True);tok.padding_side='left'
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(p,local_files_only=True,dtype=torch.float32).eval();model.requires_grad_(False)
 ts=tasks(json.loads((ROOT/'ITEMS.json').read_text())['items']);path=ROOT/(f'SMOKE_{key}.jsonl' if smoke else f'PREDICTIONS_{key}.jsonl')
 if smoke:ts=ts[:5]
 done={}
 if path.exists():
  for line in path.read_text().splitlines():
   r=json.loads(line);assert r['protocol_sha256']==sha(ROOT/'PROTOCOL.json');k=(r['id'],r['arm'],r['role']);assert k not in done;done[k]=r
 todo=[t for t in ts if (t['id'],t['arm'],t['role']) not in done];begin=time.perf_counter()
 for off in range(0,len(todo),4):
  block=todo[off:off+4];prompts=[prompt(tok,t) for t in block];enc=tok([x[0] for x in prompts],padding=True,return_tensors='pt',add_special_tokens=False)
  gen=model.generate(**enc,max_new_tokens=48,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
  with path.open('a') as f:
   for t,(pr,prefix),tokens in zip(block,prompts,gen[:,enc['input_ids'].shape[1]:]):
    raw=tok.decode(tokens,skip_special_tokens=True);text=prefix+raw;fields={k.lower():v.strip().rstrip('.') for k,v in re.findall(r'(Subject|Object|Answer):\s*(.*?)(?=;\s*(?:Subject|Object|Answer):|\n|$)',text,re.I)}
    r={k:t[k] for k in ['id','arm','role']};r.update({'model':key,'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'prompt':pr,'raw':raw,'fields':fields,'direct_answer':raw.strip() if t['arm']=='direct' else None,'new_tokens':len(tokens),'hit_length_limit':int(tokens[-1])!=tok.eos_token_id and len(tokens)==48});f.write(json.dumps(r)+'\n');f.flush()
  print(key,'smoke' if smoke else 'run',len(done)+min(off+4,len(todo)),'/',len(ts),'seconds',round(time.perf_counter()-begin,1),flush=True)
 assert sum(1 for _ in path.open())==len(ts)
 (ROOT/f'{"SMOKE_" if smoke else ""}COMPLETE_{key}.json').write_text(json.dumps({'model':key,'records':len(ts),'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'predictions_sha256':sha(path),'elapsed_seconds':time.perf_counter()-begin},indent=2)+'\n')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','smoke','run']);ap.add_argument('--model',choices=MODELS);a=ap.parse_args()
 if a.mode=='prepare':prepare()
 else:run(a.model,a.mode=='smoke')
