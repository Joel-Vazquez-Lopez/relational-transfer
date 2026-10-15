"""Follow-up frozen development ablation: same extraction output with query visible.
Reuses query-blind outputs from the first diagnostic; no independent confirmation.
"""
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
import run as parent
ROOT=Path(__file__).resolve().parent

def prepare():
 p=ROOT/'FIELD_ONLY_PROTOCOL.json';assert not p.exists();parent.validate()
 c={'status':'fixed_after_primary_results_before_field_only_inference','date':'2026-10-15','scope':'Exploratory follow-up on same 24 examined events','question':'Does showing the original question alter extraction when the output schema and extraction instruction are identical?',
 'design':'Reuse question-blind extraction for each event. Generate extraction twice with each original question appended. Identical system instruction, Subject prefix, max tokens, target verb, weights and decoding. No final Answer field is requested in either arm. Query information and input length still differ. This does not isolate a hidden mechanism.',
 'models':['0.5b','1.5b'],'records_per_model':48,'primary':'Paired both-correct subject/object answers from requested generated fields versus blind extraction, with deterministic or existing trained question-role selector. Also role-tuple changes, parse failures, field-wise correctness. No success threshold or significance-based continuation.',
 'parent_protocol_sha256':parent.sha(ROOT/'PROTOCOL.json'),'script_sha256':parent.sha(Path(__file__)),
 'baseline_hashes':{k:parent.sha(ROOT/f'PREDICTIONS_{k}.jsonl') for k in parent.MODELS}}
 p.write_text(json.dumps(c,indent=2)+'\n');print('FROZEN FIELD ONLY',parent.sha(p),flush=True)
@torch.inference_mode()
def run(key):
 parent.validate();cfg=json.loads((ROOT/'FIELD_ONLY_PROTOCOL.json').read_text());assert cfg['script_sha256']==parent.sha(Path(__file__));assert cfg['baseline_hashes'][key]==parent.sha(ROOT/f'PREDICTIONS_{key}.jsonl')
 torch.set_num_threads(2);torch.manual_seed(910215);p=parent.MODELS[key];tok=AutoTokenizer.from_pretrained(p,local_files_only=True);tok.padding_side='left'
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(p,local_files_only=True,dtype=torch.float32).eval();model.requires_grad_(False)
 ts=[t for t in parent.tasks(json.loads((ROOT/'ITEMS.json').read_text())['items']) if t['arm']=='query_visible'];path=ROOT/f'FIELD_ONLY_{key}.jsonl';done=set()
 if path.exists():
  for line in path.read_text().splitlines():
   r=json.loads(line);assert r['protocol_sha256']==parent.sha(ROOT/'FIELD_ONLY_PROTOCOL.json');done.add((r['id'],r['role']))
 todo=[t for t in ts if (t['id'],t['role']) not in done];start=time.perf_counter()
 for off in range(0,len(todo),4):
  block=todo[off:off+4];prompts=[]
  for t in block:
   q=t['item'][t['role']+'_question'];pr,prefix=parent.prompt(tok,{**t,'arm':'query_blind'});marker='<|im_end|>\n<|im_start|>assistant\nSubject:';assert pr.endswith(marker);pr=pr[:-len(marker)]+'\nQuestion: '+q+marker;prompts.append(pr)
  enc=tok(prompts,padding=True,return_tensors='pt',add_special_tokens=False);gen=model.generate(**enc,max_new_tokens=48,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
  with path.open('a') as f:
   for t,pr,tokens in zip(block,prompts,gen[:,enc['input_ids'].shape[1]:]):
    raw=tok.decode(tokens,skip_special_tokens=True);fields={k.lower():v.strip().rstrip('.') for k,v in parent.re.findall(r'(Subject|Object|Answer):\s*(.*?)(?=;\s*(?:Subject|Object|Answer):|\n|$)','Subject:'+raw,parent.re.I)}
    f.write(json.dumps({'id':t['id'],'role':t['role'],'model':key,'protocol_sha256':parent.sha(ROOT/'FIELD_ONLY_PROTOCOL.json'),'prompt':pr,'raw':raw,'fields':fields})+'\n');f.flush()
  print(key,'field-only',len(done)+min(off+4,len(todo)),'/48',round(time.perf_counter()-start,1),flush=True)
 rows=[json.loads(l) for l in path.read_text().splitlines()];assert len(rows)==48 and len({(r['id'],r['role']) for r in rows})==48
 (ROOT/f'FIELD_ONLY_COMPLETE_{key}.json').write_text(json.dumps({'records':48,'protocol_sha256':parent.sha(ROOT/'FIELD_ONLY_PROTOCOL.json'),'predictions_sha256':parent.sha(path)},indent=2)+'\n')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);ap.add_argument('--model',choices=parent.MODELS);a=ap.parse_args()
 if a.mode=='prepare':prepare()
 else:run(a.model)
