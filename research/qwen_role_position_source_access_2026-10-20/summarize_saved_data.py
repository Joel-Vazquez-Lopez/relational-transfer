"""Resource and full failure accounting from saved journals; no model calls."""
from pathlib import Path
import json,collections
HERE=Path(__file__).resolve().parent
def rows(name):return [json.loads(l) for l in (HERE/name).read_text().splitlines()]
def save(name,value):(HERE/name).write_text(json.dumps(value,indent=2)+'\n')
def main():
 r=json.loads((HERE/'RESULTS.json').read_text());idx={x['id']:x for x in json.loads((HERE/'INPUTS.json').read_text())};roles=['sender','sent_person','destination_person','distractor'];c=collections.Counter();fail=[]
 for x in r['native_summary']['failures']:
  row=idx[x['id']]
  if row['variant']=='sham':continue
  actual=roles[row['assignment'].index(row['names'].index(x['actual']))] if x['actual'] in row['names'] else 'other_or_unterminated'
  key=(roles[row['query_index']],actual);c[key]+=1;fail.append(dict(id=x['id'],source=row['source'],question=row['question'],gold=x['gold'],actual=x['actual'],requested_role=key[0],returned_role=key[1]))
 out=dict(scope='Descriptive complete-primary error accounting, not a new hypothesis test',errors=len(fail),confusions=[dict(requested_role=k[0],returned_role=k[1],count=v) for k,v in c.items()],all_failures=fail)
 save('ERROR_SUMMARY.json',out)
 fw=rows('FORWARDS.jsonl');co=rows('CONTROLS.jsonl');bud=rows('BUDGET.jsonl')
 resource=dict(main_qwen_forwards=len(fw),prior_technical_attempt_forwards=5,total_study_qwen_forwards_including_technical_stop=len(fw)+5,workers=max(f['worker'] for f in fw),production_checks=len(co),all_production_passed=all(c['passed'] for c in co),max_peak_mlx_bytes=max(f['metrics']['peak_mlx_bytes'] for f in fw),minimum_predispatch_free_disk_bytes=min(f['free_disk_bytes'] for f in bud),max_predispatch_swap_bytes=max(f['swap_used_bytes'] for f in bud),all_predispatch_pressure_normal=all(f['memory_pressure_level']==1 for f in bud),first_dispatch=bud[0]['utc'],last_dispatch=bud[-1]['utc'],engineering_reference_passes_separate=2,engineering_batch_size_used_only_in_failed_benchmark=4)
 save('COMPUTE_ACCOUNTING.json',resource)
 native={x['id']:x for x in rows('NATIVE.jsonl')};units=json.loads((HERE/'UNITS.json').read_text());competent={u['id'] for u in units if all(native[q[s]]['native_correct'] for q in u['queries'] for s in ['a','b'])}
 secondary=[x for x in rows('HYBRIDS.jsonl') if x['cut']!=14];coherent=r['complete_three_question_hybrids'];cut=[]
 for depth in [7,14,21]:
  pp=[x for x in r['profiles'] if x['eligible'] and x['cut']==depth];vv=sorted(x['authority'] for x in pp)
  cut.append(dict(cut=depth,eligible=len(pp),mean=sum(vv)/len(vv),median=vv[len(vv)//2],minimum=min(vv),maximum=max(vv)))
 save('DESCRIPTIVE_SUMMARY.json',dict(conditional_population='Native-only eligibility; no new filtering or significance tests',cut_summaries=cut,complete_native_edges=len(competent),coherent_hybrids_all=dict(correct=sum(x['source_correct'] for x in coherent),total=len(coherent)),coherent_hybrids_on_native_complete_edges=dict(correct=sum(x['source_correct'] for x in coherent if x['unit'] in competent),total=sum(x['unit'] in competent for x in coherent)),secondary_next_token_measurements=len(secondary),secondary_outside_name_candidates=[x for x in secondary if x['first_token_name'] is None],one_excluded_edge=[k for k,v in r['native_gate']['eligibility'].items() if not v]))
if __name__=='__main__':main()
