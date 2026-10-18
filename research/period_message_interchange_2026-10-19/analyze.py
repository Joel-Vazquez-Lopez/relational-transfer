"""Independent saved-output replay. No LLM imports or extra model calls."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;ROOT=P.parent.parent

def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def calc(rows):
 s={(r['role'],r['query']):r['yes_minus_no'] for r in rows};assert len(s)==4
 return (s[0,0]-s[0,1]-s[1,0]+s[1,1])/4,[s[0,0]-s[1,0],s[1,1]-s[0,1],s[0,0]-s[0,1],s[1,1]-s[1,0]]
def main():
 lock=json.loads((P/'LOCK.json').read_text())
 for f,h in lock['files'].items():assert digest(ROOT/f)==h,f
 lines=(P/'RAW.jsonl').read_bytes().splitlines(keepends=True);rows=[json.loads(x) for x in lines];budget=[json.loads(x) for x in (P/'BUDGET.jsonl').read_text().splitlines()]
 assert len(rows)==len(budget)==323 and [r['call'] for r in rows]==list(range(1,324))
 inputs={r['id']:r for r in json.loads((P/'INPUTS.json').read_text())};sci=[r for r in rows if r['kind']=='scientific'];lookup={(r['id'],r['condition']):r for r in sci};ledger=[];checks=0
 def arrays(r):
  with np.load(P/r['payload']) as z:return {k:z[k].astype(np.float32) for k in z.files}
 for r in rows:
  assert digest(P/r['payload'])==r['payload_sha256'];a=arrays(r);row=inputs[r['id']];cls={k:float(np.logaddexp.reduce(a['logits'][v].astype(np.float64))) for k,v in row['aliases'].items()}
  s=cls['Yes']-cls['No'];assert abs(s-r['yes_minus_no'])<1e-10
  best=int(a['logits'].argmax());actual=next((k for k,v in row['aliases'].items() if best in v),None);assert best==r['argmax_id'] and actual==r['actual'];assert all(abs(cls[k]-r['class_logits'][k])<1e-10 for k in cls)
  margin=s if row['expected']=='Yes' else -s;assert abs(margin-r['gold_margin'])<1e-10
  threshold=json.loads((P/'READOUT_CALIBRATION.json').read_text())['threshold'];assert threshold==r['threshold']
  cal='Yes' if s>threshold else 'No' if s<threshold else None;assert cal==r['calibrated_class'] and (cal==row['expected'])==r['calibrated_correct']
  assert r['correct']==bool(actual==row['expected'] and margin>0);checks+=1
  if r['kind']!='scientific':continue
  condition=r['condition'];base='native' if condition=='native' else 'N' if condition=='N' or condition.startswith('N_') else 'F';b=lookup[r['id'],base];bas=arrays(b)
  if condition not in ['native','F','N']:
   assert r['prefix_hashes']==b['prefix_hashes'];assert np.array_equal(a['trace'][:15],bas['trace'][:15]);assert np.array_equal(a['trace'][:,:2],bas['trace'][:,:2]);checks+=3
  target=row['expected'] if condition in ['native','F','N','P_V_voice'] else ('No' if row['expected']=='Yes' else 'Yes')
  ledger.append(dict(call=r['call'],id=r['id'],family=r['family'],split=r['split'],voice=r['voice'],role=r['role'],query=r['query'],condition=condition,native_expected=row['expected'],donation_target=target,actual=actual,native_correct=r['correct'],target_argmax_correct=actual==target,calibrated_class=r['calibrated_class'],baseline_actual=b['actual'],baseline_calibrated_class=b['calibrated_class'],native_label_changed=actual!=b['actual'],calibrated_label_changed=r['calibrated_class']!=b['calibrated_class'],newly_target_argmax=actual==target and b['actual']!=target,newly_target_calibrated=r['calibrated_class']==target and b['calibrated_class']!=target,target_calibrated_correct=r['calibrated_class']==target,yes_minus_no=s,score_shift_from_own_graph=s-b['yes_minus_no'],target_signed_shift_from_own_graph=((-1)**(r['role']+r['query']))*(b['yes_minus_no']-s)))
 # Replay all three validation comparisons and all pre-cut/Q-invariance checks.
 first=next(r for r in sci if r['condition']=='native');fid=first['id']
 for name,condition in [('validation_native','native'),('validation_F','F'),('validation_identity','F')]:
  a=arrays(next(r for r in rows if r['condition']==name));b=arrays(lookup[fid,condition]);assert np.max(abs(a['logits']-b['logits']))<=.05;assert np.max(abs(a['trace']-b['trace']))<=1e-6;checks+=1
 for row in inputs.values():
  for graph in ['F','N']:
   a=arrays(lookup[row['id'],graph]);b=arrays(lookup[row['id'],'native']);assert np.array_equal(a['trace'][:15],b['trace'][:15]);checks+=1
   qrow=next(t for t in inputs.values() if t['family']==row['family'] and t['voice']==row['voice'] and t['role']==row['role'] and t['query']!=row['query']);other=arrays(lookup[qrow['id'],graph]);j=1 if graph=='F' else 0
   assert np.array_equal(a['kv'][:,:,:,j],other['kv'][:,:,:,j]);checks+=1
 units=[]
 for split in ['development','evaluation']:
  report=json.loads((P/(split.upper()+'_SUMMARY.json')).read_text())
  for u in report['units']:
   sub=[r for r in sci if r['family']==u['family'] and r['voice']==u['voice']];conditions=sorted({r['condition'] for r in sub});ds={};ps={}
   for c in conditions:ds[c],ps[c]=calc([r for r in sub if r['condition']==c]);assert abs(ds[c]-u['D'][c])<1e-10
   b=ds['F'];valid=b>=.5 and min(ps['F'])>0;E={c:((ds['N'] if c.startswith('N_') else b)-ds[c])/(2*b) if valid else None for c in conditions if c not in ['native','F','N']}
   for c,e in E.items():assert e==u['E'][c] or abs(e-u['E'][c])<1e-10
   base={(r['role'],r['query']):r['yes_minus_no'] for r in sub if r['condition']=='F'}
   def shifts(c):return [(-1)**(r['role']+r['query'])*(base[r['role'],r['query']]-r['yes_minus_no']) for r in sub if r['condition']==c]
   voice=max(abs(base[r['role'],r['query']]-r['yes_minus_no']) for r in sub if r['condition']=='P_V_voice')
   control=valid and abs(E['P_V_voice'])<=.1 and voice<=.5*b
   hv=bool(control and E['P_V']>=.25 and E['P_V']-E['P_K']>=.25 and E['P_V']-E['N_V']>=.25 and min(shifts('P_V'))>.1)
   hk=bool(control and E['P_K']>=.25 and E['P_K']-E['P_V']>=.25 and E['P_K']-E['N_K']>=.25 and min(shifts('P_K'))>.1)
   hs=bool(valid and all(abs(E[c])<=.1 for c in ['P_K','P_V','P_KV']))
   assert (hv,hk,hs)==(u['HV'],u['HK'],u['HS']);checks+=1
   units.append(dict(split=split,**u))
 dev=[r for r in units if r['split']=='development'];prediction=next((h for h in ['HV','HK','HS'] if all(r[h] for r in dev)),None);pred=json.loads((P/'PREDICTION_BEFORE_EVALUATION.json').read_text());assert prediction==pred['prediction']
 devlines=[line for line,r in zip(lines,rows) if r['split']=='development'];assert digest(P/'RAW.jsonl') # Exact persisted prefix before first evaluation output.
 assert hashlib.sha256(b''.join(devlines)).hexdigest()==pred['raw_sha256'];assert max(r['call'] for r in rows if r['split']=='development')<min(r['call'] for r in rows if r['split']=='evaluation');checks+=1
 evaluated=[r for r in units if r['split']=='evaluation'];passed=bool(prediction and all(r[prediction] for r in evaluated));decision=json.loads((P/'DECISION.json').read_text());assert passed==decision['passed'] and prediction==decision['prediction'];checks+=1
 controls=json.loads((P/'CONTROLS.json').read_text());assert all(c['passed'] for c in controls)
 with (P/'ANSWER_LEDGER.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
 profiles=[]
 for u in units:
  profiles.append(dict(family=u['family'],voice=u['voice'],split=u['split'],**{f'D_{k}':v for k,v in u['D'].items()},**{f'E_{k}':v for k,v in u['E'].items()},HV=u['HV'],HK=u['HK'],HS=u['HS'],voice_max_shift=u['voice_max_absolute_shift']))
 with (P/'UNIT_LEDGER.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(profiles[0]));w.writeheader();w.writerows(profiles)
 save(P/'FAILURES.json',[r for r in ledger if not r['native_correct'] or not r['target_argmax_correct']]);save(P/'PAYLOAD_MANIFEST.json',[dict(path=r['payload'],sha256=r['payload_sha256'],bytes=(P/r['payload']).stat().st_size) for r in rows])
 summary=dict(decision=decision,technical_control_count=len(controls),independent_checks=checks,all_passed=True,normal=[dict(family=f,native_correct=sum(r['native_correct'] for r in ledger if r['family']==f and r['condition']=='native'),calibrated_correct=sum(r['target_calibrated_correct'] for r in ledger if r['family']==f and r['condition']=='native'),n=8) for f in ['g01','g02','h01','h02']],units=units,all_call_arrays_saved_locally=True,remote_archive_status='See ARRAY_ARCHIVE_MANIFEST and final GitHub preservation receipt')
 save(P/'INDEPENDENT_AUDIT.json',summary)
 colors=['#d97706','#2563eb','#7c3aed','#16a34a','#d97706','#2563eb','#7c3aed'];modes=['P_V','P_K','P_KV','P_V_voice','N_V','N_K','N_KV'];fig,axes=plt.subplots(1,2,figsize=(13.5,5.5),sharey=True)
 for ax,voice in zip(axes,['active','passive']):
  uu=[next(u for u in units if u['family']==f and u['voice']==voice) for f in ['g01','g02','h01','h02']]
  for j,(mode,c) in enumerate(zip(modes,colors)):
   vals=[np.nan if u['E'][mode] is None else u['E'][mode] for u in uu];ax.bar(np.arange(4)+(j-3)*.11,vals,width=.10,color=c,alpha=.45 if mode.startswith('N_') else 1,label=mode,hatch='//' if mode.startswith('N_') else None)
  ax.axhline(0,c='black',lw=.6);ax.axhline(.25,c='black',lw=.7,ls=':');ax.set_xticks(range(4),['g01\ndev','g02\ndev','h01\nevaluation','h02\nevaluation']);ax.set_title(voice);ax.set_xlabel('Complete lexical family');ax.grid(axis='y',alpha=.15)
 axes[0].set_ylabel('Contrast change / (2D_F); period full reversal = 1');axes[1].legend(fontsize=8,ncol=2);fig.suptitle('Projected message interchange: all fixed families and controls');fig.tight_layout()
 for ext in ['png','svg','pdf']:fig.savefig(P/('MESSAGE_TRANSFER.'+ext),dpi=165,bbox_inches='tight')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
