"""Frozen exploratory API comparison. No retries or unsupported candidate logits."""
import hashlib,importlib.util,json,os,sys,time
from datetime import datetime,timezone
from decimal import Decimal
from provider_probe import ROOT,MODEL,load_key,client,safe_json
from budget import CreditGuard
QWEN=ROOT.parent/'stronger_model_development_2026-10-15'
sys.path.insert(0,str(QWEN))
from design import sentence
from run import file_sha
spec=importlib.util.spec_from_file_location('phi_helpers',ROOT.parent/'phi3_cross_family_development_2026-10-15'/'run.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

def messages(cfg,r):
 system=cfg['strong_system'] if r['panel']!='legacy' else 'Answer with exactly one of the two candidate names. Output only the name.'
 user='Sentence: '+sentence(r,r['form'])
 if r['panel']!='strong_free':
  names=[r['a'],r['b']] if r['order']=='ab' else [r['b'],r['a']]
  user+='\nCandidates: '+', '.join(names)
 user+='\nQuestion: '+r['question']
 return [{'role':'system','content':system},{'role':'user','content':user}]

def main():
 cfg=json.loads((ROOT/'GEMMA_PROTOCOL.json').read_text())
 for f,v in json.loads((ROOT/'GEMMA_SOURCE_LOCK.json').read_text()).items():assert file_sha(ROOT/f)==v,f
 key=load_key(); guard=CreditGuard(ROOT/'SPEND_LEDGER.jsonl'); pending,_,_=guard.state()
 if pending: raise RuntimeError('Unresolved billing reservation requires review')
 out=ROOT/'gemma.jsonl';done={}
 if out.exists():
  for line in out.read_text().splitlines():
   r=json.loads(line);assert r['key'] not in done;done[r['key']]=r
 fingerprint={r['system_fingerprint'] for r in done.values() if r['system_fingerprint'] is not None}
 assert len(fingerprint)<=1
 started=time.monotonic()
 with client(key) as c, out.open('a',buffering=1) as f:
  catalog=c.get('/v1/models');assert catalog.status_code==200
  info=next(x for x in catalog.json()['data'] if x['id']==MODEL)
  assert info['pricing']==cfg['pricing'],'Provider pricing changed'
  safe_json(ROOT/'GEMMA_PROVIDER_METADATA.json',info,key)
  for r in h.jobs(cfg):
   if r['key'] in done:continue
   rid='gemma-panel-v1/'+r['key']; msg=messages(cfg,r)
   body={'model':MODEL,'messages':msg,**cfg['generation']}
   assert len(json.dumps(body).encode())<4096
   if not guard.reserve(rid,'0.52430'):raise RuntimeError('Settled request missing raw record; manual reconciliation required')
   try:
    response=c.post('/v1/chat/completions',json=body)
    if response.status_code!=200:raise RuntimeError('HTTP_'+str(response.status_code))
    data=response.json();usage=data.get('usage',{});a=usage.get('prompt_tokens');b=usage.get('completion_tokens')
    assert type(a)==int and type(b)==int and 0<=a<=1048576 and 0<=b<=12
    assert data['model']==MODEL
    fp=data.get('system_fingerprint')
    if fp is not None:
     fingerprint.add(fp);assert len(fingerprint)<=1,'Serving fingerprint changed'
    choices=data['choices'];assert len(choices)==1
    answer=choices[0]['message']['content'];assert isinstance(answer,str)
    answer=answer.strip(); cost=(Decimal(a)*Decimal('.25')+Decimal(b)*Decimal('.5'))/Decimal(1000000)
    record=dict(**r,messages=msg,messages_sha256=hashlib.sha256(json.dumps(msg,sort_keys=True).encode()).hexdigest(),
      answer=answer,gold=r[r['role']],correct=answer==r[r['role']],valid=answer in [r['actor'],r['patient']],
      finish_reason=choices[0]['finish_reason'],ended_eos=choices[0]['finish_reason']=='stop',usage=usage,
      model=data['model'],system_fingerprint=fp,response_id=data.get('id'),created=data.get('created'),
      received_utc=datetime.now(timezone.utc).isoformat(),estimated_eur=str(cost),candidate_scores_available=False)
    serialized=json.dumps(record)
    if key in serialized:raise RuntimeError('Secret serialization guard')
    f.write(serialized+'\n');f.flush();os.fsync(f.fileno());guard.settle(rid,cost*2);done[r['key']]=record
   except Exception as exc:
    guard.unresolved(rid,'failure_'+type(exc).__name__)
    safe_json(ROOT/'GEMMA_STOP.json',{'request_id':rid,'error_type':type(exc).__name__,'http_status':response.status_code if 'response' in locals() else None,'retry':False},key)
    print(json.dumps({'stopped':True,'completed':len(done),'error_type':type(exc).__name__}),flush=True);return
   if len(done)%48==0:print(json.dumps({'completed':len(done),'total':1728,'elapsed_s':round(time.monotonic()-started,1),'conservative_accounted_eur':str(guard.state()[2])}),flush=True)
 assert len(done)==1728
 safe_json(ROOT/'GEMMA_COMPLETE.json',{'records':len(done),'sha256':file_sha(out),'seconds':time.monotonic()-started,'conservative_accounted_eur':str(guard.state()[2])},key)
 print('GEMMA_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  print(json.dumps({'stopped_before_request':True,'error_type':type(exc).__name__}),flush=True);sys.exit(1)
