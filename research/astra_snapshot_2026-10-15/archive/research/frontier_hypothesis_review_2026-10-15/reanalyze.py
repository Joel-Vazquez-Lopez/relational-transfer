"""Read-only archival margin analysis; no models, inference, training or sealed data."""
from pathlib import Path
import json,hashlib,statistics,collections
R=Path(__file__).resolve().parent;P=R.parents[1];S=P/'research/qwen_role_equivariance_diagnostic_2026-10-12'
mean=statistics.mean
names=['untouched_qwen15','untouched_qwen05','fresh1_source','fresh1_natural','fresh3_source','fresh3_natural'];pairs=[('actor_performed','patient_verb'),('actor_acted','patient_affected'),('actor_performed','patient_affected'),('actor_acted','patient_verb')];forms=['active','passive','subject_cleft','object_cleft'];raw={};hashes={};details={};summary={}
for name in names:
 path=S/(name+'.json');hashes[str(path.relative_to(P))]=hashlib.sha256(path.read_bytes()).hexdigest();rows=json.loads(path.read_text())['records'];assert len(rows)==720;present={(x['id'],x['form'],x['wording']):x for x in rows if not x['omitted']};omitted={(x['id'],x['wording']):x for x in rows if x['omitted']};ids=sorted({x['id'] for x in rows});assert len(ids)==36
 raw[name]=present;summary[name]={};details[name]={};mismatch=sum((x['actor_margin']>0 if x['role']=='actor' else x['actor_margin']<0)!=x['correct'] for x in present.values());summary[name]['candidate_margin_vs_generated_correctness_mismatches']=mismatch
 for a,q in pairs:
  label=a+'/'+q;summary[name][label]={};details[name][label]={}
  for form in forms:
   data=[]
   for eid in ids:
    A=present[eid,form,a]['actor_margin'];B=present[eid,form,q]['actor_margin'];b=(A+B)/2;r=(A-B)/2;both=A>0 and B<0;oa=omitted[eid,a]['actor_margin'];ob=omitted[eid,q]['actor_margin'];ca=A-oa;cb=B-ob
    row={'id':eid,'pair_index':present[eid,form,a]['pair_index'],'direction':present[eid,form,a]['direction'],'verb':present[eid,form,a]['verb'],'A':A,'B':B,'bias':b,'role_contrast':r,'both_correct':both,'positive_contrast':r>0,'bias_dominant_failure':r>0 and not both,'nonpositive_contrast_failure':r<=0 and not both,'calibrated_both_correct':ca>0 and cb<0,'calibrated_A':ca,'calibrated_B':cb};assert both==(r>abs(b));data.append(row)
   details[name][label][form]=data;summary[name][label][form]={'n':36,'mean_bias':mean(x['bias'] for x in data),'mean_abs_bias':mean(abs(x['bias']) for x in data),'mean_role_contrast':mean(x['role_contrast'] for x in data),'positive_contrast':sum(x['positive_contrast'] for x in data),'joint':sum(x['both_correct'] for x in data),'bias_dominant_failure':sum(x['bias_dominant_failure'] for x in data),'nonpositive_contrast_failure':sum(x['nonpositive_contrast_failure'] for x in data),'content_free_calibrated_joint':sum(x['calibrated_both_correct'] for x in data),'calibration_gains':sum(x['calibrated_both_correct'] and not x['both_correct'] for x in data),'calibration_losses':sum(x['both_correct'] and not x['calibrated_both_correct'] for x in data)}
changes={}
for start in ['fresh1','fresh3']:
 changes[start]={}
 for label in [a+'/'+q for a,q in pairs]:
  changes[start][label]={}
  for form in forms:
   a=details[start+'_source'][label][form];b=details[start+'_natural'][label][form];assert [x['id'] for x in a]==[x['id'] for x in b];s=1 if form!='passive' else -1
   delta=[{'id':x['id'],'pair_index':x['pair_index'],'subject_bias_change':s*(y['bias']-x['bias']),'role_contrast_change':y['role_contrast']-x['role_contrast']} for x,y in zip(a,b)];groups=sorted({x['pair_index'] for x in delta})
   changes[start][label][form]={'mean_subject_oriented_bias_change':mean(x['subject_bias_change'] for x in delta),'mean_role_contrast_change':mean(x['role_contrast_change'] for x in delta),'positive_subject_bias_groups':sum(mean(x['subject_bias_change'] for x in delta if x['pair_index']==g)>0 for g in groups),'positive_role_contrast_groups':sum(mean(x['role_contrast_change'] for x in delta if x['pair_index']==g)>0 for g in groups),'groups':len(groups)}
# Cross-bank construction tradeoff, independently reconstruct gain/loss counts.
tradeoff={}
for bank,folder in [('bank1','role_subject_decoupling_pilot_2026-10-12'),('bank2','role_subject_balanced_followup_2026-10-12')]:
 tradeoff[bank]={}
 for start in ['fresh1','fresh3']:
  arms={}
  for arm in ['aligned','decoupled']:
   path=P/'research'/folder/f'EVAL_{start}_{arm}.json';hashes[str(path.relative_to(P))]=hashlib.sha256(path.read_bytes()).hexdigest();rec=json.loads(path.read_text())['records'];arms[arm]={(x['id'],x['form'],x['role']):x for x in rec}
  tradeoff[bank][start]={}
  for form in ['active','passive']:
   ids=sorted({k[0] for k in arms['aligned'] if k[1]==form});rows=[]
   for eid in ids:
    good=lambda arm:all(arms[arm][eid,form,r]['correct'] for r in ['actor','patient']);a,b=good('aligned'),good('decoupled');rows.append((a,b))
   tradeoff[bank][start][form]={'n':len(ids),'gains':sum(b and not a for a,b in rows),'losses':sum(a and not b for a,b in rows),'aligned_joint':sum(a for a,b in rows),'decoupled_joint':sum(b for a,b in rows),'invalid_outputs':sum(not x['valid'] for arm in arms.values() for k,x in arm.items() if k[1]==form)}
for rel,h in hashes.items():assert hashlib.sha256((P/rel).read_bytes()).hexdigest()==h
out={'scope':'Post hoc reanalysis of previously completed constructed-language studies; no independent confirmation or neural mechanistic identification','decomposition':'b=(m_actor+m_patient)/2; r=(m_actor-m_patient)/2; both right iff r>|b|. This is an exact algebraic reparameterization, not an explanatory discovery. Scores from two prompts are not a probe of one hidden state.','calibration':'Subtract each query-specific sentence-omitted margin, already archived. Standard context-free calibration diagnostic, not a proposed novel intervention.','sources':hashes,'summary':summary,'natural_adaptation_changes':changes,'cross_bank_tradeoff':tradeoff}
(R/'ARCHIVAL_REANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n');(R/'MARGIN_DECOMPOSITIONS.json').write_text(json.dumps(details,indent=2)+'\n')
print('UNTUCHED1.5 EXPLICIT PAIR')
for form,v in summary['untouched_qwen15']['actor_performed/patient_verb'].items():print(form,v)
print('NATURAL ADAPTATION EXPLICIT')
for start,v in changes.items():print(start,v['actor_performed/patient_verb'])
print('TRADEOFF',tradeoff)
