"""One authorized, budget-guarded authentication test. No automatic retries."""
import json,sys
from decimal import Decimal
from provider_probe import ROOT,MODEL,load_key,client,safe_json
from budget import CreditGuard

def main():
 key=load_key(); guard=CreditGuard(ROOT/'SPEND_LEDGER.jsonl'); rid='gemma-auth-smoke-v1'
 if not guard.reserve(rid,'0.52430'):
  print('Minimal inference test already settled; no repeat request.'); return
 try:
  with client(key) as c:
   r=c.post('/v1/chat/completions',json={'model':MODEL,'messages':[{'role':'user','content':'Reply with only the word OK.'}],'temperature':0,'max_tokens':12,'reasoning_effort':'none','stream':False})
  if r.status_code!=200:
   guard.unresolved(rid,'http_'+str(r.status_code)); print(json.dumps({'inference_verified':False,'http_status':r.status_code,'automatic_retry':False})); return
  d=r.json(); u=d.get('usage',{}); a=u.get('prompt_tokens'); b=u.get('completion_tokens')
  if type(a)!=int or type(b)!=int or not 0<=a<=1048576 or not 0<=b<=12: raise ValueError('Usage outside guarded bounds')
  cost=(Decimal(a)*Decimal('.25')+Decimal(b)*Decimal('.5'))/Decimal(1000000)
  report={'inference_verified':True,'http_status':200,'model':d.get('model'),'usage':u,'estimated_eur':str(cost),'conservative_accounted_eur':str(cost*2),'response':d.get('choices',[]),'system_fingerprint':d.get('system_fingerprint')}
  safe_json(ROOT/'INFERENCE_SMOKE.json',report,key); guard.settle(rid,cost*2)
  print(json.dumps({k:report[k] for k in ['inference_verified','http_status','model','usage','estimated_eur','conservative_accounted_eur']}))
 except Exception as e:
  guard.unresolved(rid,'error_'+type(e).__name__); print(json.dumps({'inference_verified':False,'error_type':type(e).__name__,'automatic_retry':False}))
if __name__=='__main__':
 try: main()
 except Exception as e: print(json.dumps({'stopped':True,'error_type':type(e).__name__})); sys.exit(1)
