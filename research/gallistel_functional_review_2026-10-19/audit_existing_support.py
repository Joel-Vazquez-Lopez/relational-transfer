"""Metadata/annotation support audit only. No inference or probe fitting."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent

def main():
 p=ROOT/'research/graded_relational_routing_2026-10-18/INPUTS.json'
 inputs=json.loads(p.read_text());table={}
 for r in inputs:
  a=int(r['expected']=='Yes');assert a==int(r['role']==r['query'])
  table.setdefault((a,r['query']),set()).add(r['role'])
 assert all(len(s)==1 for s in table.values())
 atlas=[];sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()}
 for m in ['qwen7b','llama3b']:
  f=ROOT/'research/linguistic_development_atlas_continuation_2026-10-18'/m/'RAW.json'
  sources[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
  rows=json.loads(f.read_text())['rows']
  for fam in ['a01','a02']:
   rr=[r for r in rows if r['family']==fam and r['question_key']=='agent']
   assert len(rr)==4 and len({r['question'] for r in rr})==1 and len({r['expected'] for r in rr})==1
   assignment=lambda r:json.dumps({k:sorted(v) if isinstance(v,list) else v for k,v in r['annotations'].items()},sort_keys=True)
   within=[r for r in rr if r['frame']=='within'];across=[r for r in rr if r['frame']=='across']
   assert assignment(within[0])==assignment(within[1]);assert assignment(across[0])!=assignment(across[1])
   atlas.append(dict(model=m,family=fam,question=rr[0]['question'],same_gold_answer=rr[0]['expected'],distinct_role_assignments=len({assignment(r) for r in rr}),physical_sentences=4,canonical_labels=[dict(id=r['id'],roles=json.loads(assignment(r))) for r in rr]))
 result=dict(no_model_inference=True,no_probe_fitting=True,binary_rows=len(inputs),role_determined_by_gold_answer_and_query=True,
    mapping=[dict(answer_yes=a,query=q,role=next(iter(v))) for (a,q),v in sorted(table.items())],
    atlas_constant_answer_groups=atlas,independent_lexical_families=2,
    scope='Metadata support, not evidence of internal accessibility or causal use',sources_sha256=sources)
 (HERE/'IDENTIFIABILITY_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(binary_rows=len(inputs),atlas_groups=len(atlas),all_checks_passed=True,no_model_inference=True)))
if __name__=='__main__':main()
