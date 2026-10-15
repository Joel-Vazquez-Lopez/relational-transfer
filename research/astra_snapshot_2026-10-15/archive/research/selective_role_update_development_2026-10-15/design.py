"""Prospective algebraically labelled multi-event development probe."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FAMILIES=[(['Amy','Ben','Fred','Jane'],['nudged','tagged']),(['Mark','Rose','Bill','Dan'],['pushed','pulled']),(['Carl','Matt','David','Susan'],['tapped','hugged']),(['John','Mary','Paul','Kate'],['tagged','pushed']),(['Tom','Jack','Anne','Grace'],['pulled','tapped']),(['James','Luke','Emma','Sarah'],['hugged','nudged'])]
CONDITIONS=['base','actor_edit','patient_edit','clause_order','passive','object_cleft','actor_edit_object_cleft']
def make_items():
 rows=[]
 for family,(names,verbs) in enumerate(FAMILIES):
  for direction in range(2):
   a,b,c,d=names if direction==0 else [names[1],names[0],names[3],names[2]]
   v,w=verbs
   for condition in CONDITIONS+['single_anchor']:
    events=[{'actor':a,'patient':b,'verb':v},{'actor':d,'patient':a,'verb':w}];background=c
    if condition in ['actor_edit','actor_edit_object_cleft']:events[0]['actor']=c;background=a
    if condition=='patient_edit':events[0]['patient']=c;background=b
    e=events[0];first=f"{e['actor']} {v} {e['patient']}"
    if condition=='passive':first=f"{b} was {v} by {a}"
    if condition in ['object_cleft','actor_edit_object_cleft']:first=f"It was {e['patient']} that {e['actor']} {v}"
    second=f'{d} {w} {a}'
    passage=f'{first}, while {second}. {background} was nearby.'
    if condition=='clause_order':passage=f'{second}, while {first}. {background} was nearby.'
    for event_idx,event in enumerate(events):
     text=passage if condition!='single_anchor' else f"{event['actor']} {event['verb']} {event['patient']}."
     for role in ['actor','patient']:
      for wording in ['lexical','explicit']:
       verb=event['verb']
       if wording=='lexical':q=f'Who {verb} someone in this passage?' if role=='actor' else f'Who was {verb} by someone in this passage?'
       else:q=f"For the event described by '{verb}', which person "+('carried out the action?' if role=='actor' else 'had the action done to them?')
       key=f'f{family}d{direction}|{condition}|e{event_idx}|{role}|{wording}'
       rows.append(dict(key=key,family=family,direction=direction,event_id=f'f{family}d{direction}',condition=condition,target_event=event_idx,role=role,wording=wording,passage=text,question=q,gold=event[role],events=events,people=names,background=background))
 return rows

def validate(rows):
 assert len(rows)==768 and len({r['key'] for r in rows})==768
 by={r['key']:r for r in rows}
 for r in rows:
  assert r['gold'] in r['passage'] and r['events'][0]['verb']!=r['events'][1]['verb']
  base=by[r['key'].replace('|'+r['condition']+'|','|base|')]
  change=(r['condition'] in ['actor_edit','actor_edit_object_cleft'] and r['target_event']==0 and r['role']=='actor') or (r['condition']=='patient_edit' and r['target_event']==0 and r['role']=='patient')
  assert (r['gold']!=base['gold'])==change
  if r['condition']!='single_anchor':
   assert {n:r['passage'].split().count(n)+r['passage'].split().count(n+',')+r['passage'].split().count(n+'.') for n in r['people']}=={n:base['passage'].split().count(n)+base['passage'].split().count(n+',')+base['passage'].split().count(n+'.') for n in r['people']}
 return {'rows':len(rows),'base_events':12,'name_families':6,'all_gold_change_masks_verified':True,'multi_event_name_frequencies_matched':True,'source':'new constructed examples, not naturally occurring text or human-reviewed natural data'}
if __name__=='__main__':
 rows=make_items();report=validate(rows)
 (ROOT/'ITEMS.json').write_text(json.dumps(rows,indent=2)+'\n');(ROOT/'VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
