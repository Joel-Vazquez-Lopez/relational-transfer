"""Static tokenization, source annotations and numerical-graph preflight; zero forwards."""
import json,shutil,importlib.metadata,argparse
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from core import *
def build():
 oldpath=ROOT/'research/competence_selection_confirmation_2026-10-19';old=json.loads((oldpath/'LOCK.json').read_text());model=old['models']['qwen7b'];tok=AutoTokenizer.from_pretrained(ROOT/model['path'],local_files_only=True,trust_remote_code=False);pad=tok.encode('x',add_special_tokens=False);assert len(pad)==1
 def row(fam,split,a,b,past,base,r,v,q,qv,lex,m,pos,panel):
  agent,patient=(a,b) if r==0 else (b,a);sub,obj=(a,b) if q==0 else (b,a);sent=f'{agent} {past} {patient}.' if v=='active' else f'{patient} was {past} by {agent}.';question=f'Did {sub} {base} {obj}?' if qv=='active' else f'Was {obj} {past} by {sub}?'
  symbols=['Yes','No'] if lex=='YN' else ['A','B'];ent,non=symbols if m==0 else symbols[::-1];legend=f'Response legend: entailed = {ent}; not entailed = {non}.'
  if panel=='old_reference':system=OLD_SYSTEM;user=BACKGROUND+'Sentence: '+sent+'\nQuestion: '+question
  else:system=SYSTEM;user=BACKGROUND+PREFIXES[pos]+'Sentence: '+sent+'\nQuestion: '+question+'\n'+legend
  prompt=tok.apply_chat_template([dict(role='system',content=system),dict(role='user',content=user)],tokenize=False,add_generation_prompt=True);enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];off=enc['offset_mapping'];assert len(ids)<LENGTH
  sc=prompt.index('Sentence: ')+len('Sentence: ');pc=sc+len(sent)-1;qc=prompt.index('Question: ')+len('Question: ')+len(question)-1
  locate=lambda char:next(i for i,(lo,hi) in enumerate(off) if lo<=char<hi)
  s,p,qe=map(locate,[sc,pc,qc]);assert tok.decode([ids[p]]).strip()=='.';assert '?' in tok.decode([ids[qe]])
  lo=s-(p-s+1);assert off[lo][0]>=prompt.index('Background:')
  aliases={}
  for cls,word in [('entailed',ent),('not_entailed',non)]:
   vv=[tok.encode(w,add_special_tokens=False) for w in [word,' '+word]];assert all(len(x)==1 for x in vv);aliases[cls]=sorted({x[0] for x in vv})
  assert not set(aliases['entailed'])&set(aliases['not_entailed'])
  return dict(id=f'{fam}_{panel}_r{r}_{v}_q{q}_{qv}_{lex}_m{m}_p{pos}',family=fam,split=split,panel=panel,role=r,voice=v,query=q,question_voice=qv,lexicon=lex,mapping=m,position=pos,names=[a,b],agent=agent,patient=patient,past=past,base=base,sentence=sent,question=question,legend=legend if panel=='factorial' else None,entailed_symbol=ent,nonentailed_symbol=non,expected='entailed' if r==q else 'not_entailed',expected_symbol=ent if r==q else non,prompt=prompt,token_ids=ids,offsets=off,source_start=s,period=p,question_end=qe,prompt_end=len(ids)-1,neutral_span=[lo,s],pad=pad[0],aliases=aliases)
 rows=[]
 for fam,split,a,b,past,base in FAMILIES:
  for r in [0,1]:
   for v in ['active','passive']:
    for q in [0,1]:
     rows.append(row(fam,split,a,b,past,base,r,v,q,'active','YN',0,0,'old_reference'))
     for qv in ['active','passive']:
      for lex in ['YN','AB']:
       for m in [0,1]:
        for pos in [0,1]:rows.append(row(fam,split,a,b,past,base,r,v,q,qv,lex,m,pos,'factorial'))
 assert len(rows)==816
 for n,record in model['files'].items():assert sha(ROOT/n)==record['sha256'],n
 prior=[];known=set()
 def visit(x):
  if isinstance(x,dict):
   if isinstance(x.get('sentence'),str):known.add(x['sentence'])
   for y in x.values():visit(y)
  elif isinstance(x,list):
   for y in x:visit(y)
 for p in sorted((ROOT/'research').glob('**/INPUTS.json')):
  if HERE in p.parents:continue
  visit(json.loads(p.read_text()));prior.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
 assert all(r['sentence'] not in known for r in rows)
 audit=[]
 for fam,*_ in FAMILIES:
  rr=[r for r in rows if r['family']==fam and r['panel']=='factorial'];lookup={(r['role'],r['voice'],r['query'],r['question_voice'],r['lexicon'],r['mapping'],r['position']):r for r in rr}
  for r in rr:
   other=lookup[r['role'],'passive',r['query'],r['question_voice'],r['lexicon'],r['mapping'],0]
   active=lookup[r['role'],'active',r['query'],r['question_voice'],r['lexicon'],r['mapping'],1]
   assert (other['period'],other['question_end'],other['prompt_end'])==(active['period'],active['question_end'],active['prompt_end']),('Position mismatch',fam,other['period'],active['period'])
   for lex in ['YN','AB']:
    for m in [0,1]:
     o=lookup[r['role'],r['voice'],r['query'],r['question_voice'],lex,m,r['position']];assert r['question_end']==o['question_end'];assert r['token_ids'][:r['question_end']+1]==o['token_ids'][:o['question_end']+1]
   pair=lookup[1-r['role'],r['voice'],r['query'],r['question_voice'],r['lexicon'],r['mapping'],r['position']]
   assert (r['source_start'],r['period'],r['question_end'],r['prompt_end'])==(pair['source_start'],pair['period'],pair['question_end'],pair['prompt_end'])
   assert r['token_ids'][r['period']:]==pair['token_ids'][pair['period']:]
   for mode in ['closed','frozen','irrelevant']:
    assert np.array_equal(mask(r,mode,14,13),np.tri(LENGTH,dtype=bool))
   assert np.count_nonzero(~mask(r,'closed',14,14))==np.count_nonzero(~mask(r,'irrelevant',14,14))
  audit.append(dict(family=fam,n_factorial=128,period_question_and_answer_position_match=True,legend_prefix_exact=True,role_donor_surviving_suffix_exact=True))
 oldrows=json.loads((oldpath/'INPUTS.json').read_text());control=next(r for r in oldrows if r['model']=='qwen7b' and r['id']=='c01_r0_active_q0');control={**control,'id':'exposed_c01','family':'exposed','split':'control','panel':'exposed_control','question_voice':'active','lexicon':'YN','mapping':0,'position':0,'expected':'entailed','expected_symbol':'Yes','entailed_symbol':'Yes','nonentailed_symbol':'No','aliases':{'entailed':control['aliases']['Yes'],'not_entailed':control['aliases']['No']}}
 qc=control['prompt'].index('Question: ')+len('Question: ')+len(control['question'])-1;control['question_end']=next(i for i,(lo,hi) in enumerate(control['offsets']) if lo<=qc<hi)
 raw=[json.loads(s) for s in (oldpath/'qwen7b/RAW.jsonl').read_text().splitlines()];prior_control=next(x for x in raw if x['id']=='c01_r0_active_q0' and x['kind']=='scientific' and x['mode']=='baseline')
 save(HERE/'INPUTS.json',rows);save(HERE/'CONTROL_INPUT.json',control);save(HERE/'TOKENIZER_AUDIT.json',audit)
 save(HERE/'SOURCE_MANIFEST.json',dict(annotation='AI-only authored and reviewed templates. Directed argument assignment, not a universal Agent/Patient ontology. Non-entailment is not asserted falsity.',families=[dict(id=f,split=s,names=[a,b],past=p,base=v,intended_relation='Asymmetric directed predicate under the stated clause. No intrinsic reciprocity intended.',uncertainty='No human validation. Photographed/supervised objects are not uniformly thematic Patients. Pragmatic or extra unmentioned events are irrelevant under explicit entailment instruction.') for f,s,a,b,p,v in FAMILIES],earlier_manifests=prior,new_complete_sentences=True,position_note='Record: adds two tokens. Active prefixed period/question/answer positions equal unprefixed passive for matched content; individual word positions/order remain different. Prefix is crossed in both voices and is not assumed inert.',prior_shape_control=dict(path=str((oldpath/'qwen7b'/prior_control['payload']).relative_to(ROOT)),sha256=prior_control['payload_sha256'])))
 return old,model

def main():
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();assert not (HERE/'LOCK.json').exists()
 old,model=build()
 if not a.freeze:print('DRAFT',len(json.loads((HERE/'INPUTS.json').read_text())),'rows; zero forwards');return
 from test_pipeline import check
 save(HERE/'PREFLIGHT_TESTS.json',check(json.loads((HERE/'INPUTS.json').read_text())))
 free=shutil.disk_usage(HERE).free;assert free>2_100_000_000,('Insufficient reserve',free)
 save(HERE/'STORAGE_BUDGET.json',dict(free_bytes=free,estimated_compressed_raw_bytes=850_000_000,reserve_bytes=1024**3,packaging='Verified tar bundles up to32MiB; expanded staging NPZs removed only after full member-byte verification. All original payload bytes remain recoverable locally and remotely.'))
 files=['core.py','instrument.py','engine.py','prepare.py','run.py','analyze.py','test_pipeline.py','storage.py','PROTOCOL.md','INPUTS.json','CONTROL_INPUT.json','SOURCE_MANIFEST.json','TOKENIZER_AUDIT.json','PREFLIGHT_TESTS.json','STORAGE_BUDGET.json']
 save(HERE/'LOCK.json',dict(status='FROZEN_BEFORE_FRESH_OUTCOMES',files={n:sha(HERE/n) for n in files},model=model,native_sources=old['native_sources'],runtime={p:importlib.metadata.version(p) for p in ['mlx','mlx-lm','numpy','transformers']},output_shapes={'vocab_size':152064,'hidden_size':3584},max_forwards=1119,scientific=1104,validation=15,length=LENGTH,parent_results_commit='65719a7634222ae49f5eab4260180736f2905cfc',no_paid_calls_downloads_or_training=True))
 print('FROZEN816 native rows;1119 total planned forwards; zero fresh outcomes')
if __name__=='__main__':main()
