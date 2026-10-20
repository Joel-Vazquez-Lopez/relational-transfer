"""Archive-only endpoint audit and design feasibility. No model imports or fitting."""
from pathlib import Path
import collections,csv,hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=ROOT/'COMPASS_RESEARCH_FOUNDATION_2026-10-15/source/phase3/post_p3_2_sol_discovery'
def main():
    inputs=[]
    def read(name):
        f=SOURCE/name;inputs.append(dict(path=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        with f.open() as s:return list(csv.DictReader(s))
    scene_groups=[]
    for name in ['d15_scene_rows.csv','d21_scene_rows.csv']:
        rows=read(name);groups=collections.defaultdict(list)
        for r in rows:groups[(r['condition'],r['seed'],r['factor'])].append(r)
        for (condition,seed,factor),rs in sorted(groups.items()):
            n=len(rs);base=sum(int(r['base_correct']) for r in rs);after=sum(int(r['redirected']) for r in rs)
            joint=sum(int(r['base_correct'])*int(r['redirected']) for r in rs)
            scene_groups.append(dict(source=name,condition=condition,seed=int(seed),factor=factor,n=n,base_correct=base,target_after=after,base_correct_and_target_after=joint,target_after_without_base_correct=after-joint,legacy_target_after_rate=after/n,paired_success_all_rows=joint/n,paired_success_given_base_correct=joint/base if base else None))
    zero=[];records=0
    for name in ['d29_developmental.csv','d33_switch_developmental.csv','d38_long_withdrawal.csv','d43_h_developmental.csv','d45_release_developmental.csv','d49_tied_e_to_a_developmental.csv']:
        rs=read(name);records+=len(rs)
        for r in rs:
            if float(r['query_delta'])==0 and float(r['redirected'])>0:
                zero.append(dict(source=name,**r))
    for name in ['diagnostic_15_shared_scene_redirection.py','diagnostic_21_matched_readout.py','diagnostic_45_release_readout.py','d53_input_audit.json','d57_input_audit.json','d64_input_audit.json','discovery_plan.json']:
        f=SOURCE/name;inputs.append(dict(path=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    # Exact identity-operation counterexample; this is an algebra check, not model evidence.
    before=[0.,1.];after=before.copy();old_target=0;new_target=1
    legacy=int(max(range(2),key=after.__getitem__)==new_target)
    paired=int(max(range(2),key=before.__getitem__)==old_target and max(range(2),key=after.__getitem__)==new_target)
    assert legacy==1 and paired==0
    assert all(g['base_correct_and_target_after']<=min(g['base_correct'],g['target_after']) for g in scene_groups)
    out=dict(status='COMPLETE_ARCHIVE_ONLY',interpretation='redirected in these historical scripts denotes target-after accuracy, not an actual baseline-to-target switch. Preserve both endpoint accuracy and paired success; do not retrospectively change frozen original decisions.',scene_groups=scene_groups,development_summary_rows=records,reported_zero_query_change_with_positive_target_after=zero,identity_operation_counterexample=dict(before=before,after=after,legacy_target_after=legacy,paired_success=paired),scope_limits=['D15 and D21 reuse A1701 scenes; tables must not be pooled as independent evidence.','Paired success here requires original four-way gold then counterfactual four-way gold. It is not specifically a hidden-state-patch estimate.','Saved aggregate developmental rows lack per-scene baselines, so their actual switch rates cannot be recovered from these CSVs.','No p-values, no new model calls, no protected checkpoints.','This targeted correction does not overturn D5 training-support effects or recent Qwen studies using different scoring.'],input_manifest=inputs,model_forwards=0,training_runs=0)
    (HERE/'ARCHIVE_RECHECK.json').write_text(json.dumps(out,indent=2)+'\n')
    fields=list(scene_groups[0])
    with (HERE/'PAIRED_ENDPOINTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(scene_groups)
    print(json.dumps(dict(scene_groups=scene_groups,development_rows=records,zero_change_positive_endpoint_rows=len(zero)),indent=2))
if __name__=='__main__':main()
