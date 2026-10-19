"""Descriptive summaries of all frozen cells; no new predictions or tests."""
import json,statistics
from core import HERE,save

def main():
    r=json.loads((HERE/'RESULTS.json').read_text())
    raw=[json.loads(s) for s in (HERE/'qwen7b/RAW.jsonl').read_text().splitlines()]
    baseline=[x for x in raw if x['kind']=='scientific' and x['mode']=='baseline']
    masked=[x for x in raw if x['kind']=='scientific' and x['mode']!='baseline']
    new=[x for x in baseline if x['panel']=='factorial']
    old=[x for x in baseline if x['panel']=='old_reference']
    groups=[x for x in r['groups'] if x['mode']=='baseline']
    oldgaps=[x['old_positive_gap'] for x in r['profiles']]
    gaps=[c['gap'] for p in r['profiles'] for c in p['contrasts']]
    cells={}
    for voice in ('active','passive'):
        for panel,rr in (('old_reference',old),('factorial',new)):
            aa=[x for x in rr if x['voice']==voice]
            gg=[x for x in groups if x['panel']==panel and x['voice']==voice]
            cells[panel+'_'+voice]={'correct':sum(x['native_correct'] for x in aa),'answers':len(aa),'complete_units':sum(x['native_complete'] for x in gg),'units':len(gg)}
    summary={'kind':'Descriptive summaries of frozen cells, not new confirmatory hypotheses','behavior_development_candidate':r['prediction']['behavior']['candidate'],'routing_development_candidate':r['prediction']['routing']['candidate'],'behavior_prospectively_replicated':r['held_behavior_replicated'],'routing_prospectively_replicated':r['held_routing_replicated'],'old_positive_gap_mean':statistics.mean(oldgaps),'old_positive_gap_range':[min(oldgaps),max(oldgaps)],'old_gap_at_least4_families':sum(x>=4 for x in oldgaps),'all_factorial_contrasts':len(gaps),'factorial_gap_range':[min(gaps),max(gaps)],'contrasts_at_least4':sum(x>=4 for x in gaps),'contrasts_at_most_minus4':sum(x<=-4 for x in gaps),'contrasts_within1':sum(abs(x)<=1 for x in gaps),'pooled_contrasts_not_independent':True,'native_cells':cells,'families_passing_canonical_baseline_eligibility':sum(x['valid_positive_ordinal_baselines'] for x in r['routing']),'families_passing_entire_neutral_routing_gate':sum(x['neutral_drift_gate'] for x in r['routing']),'wrong_unperturbed_first_token_decisions':sum(not x['native_correct'] for x in baseline),'wrong_masked_first_token_decisions':sum(not x['native_correct'] for x in masked),'off_format_unperturbed_first_tokens':sum(not x['format_compliant'] for x in baseline),'off_format_masked_first_tokens':sum(not x['format_compliant'] for x in masked),'no_new_model_calls':True,'no_new_p_values_or_hypothesis_selection':True}
    save(HERE/'DESCRIPTIVE_SUMMARY.json',summary)
    text=['## What the completed observations support','',f'The old-prompt passive−active entailed-score gap was at least4 nats in {summary["old_gap_at_least4_families"]}/6 families; its mean was {summary["old_positive_gap_mean"]:.3f} nats and its range was {min(oldgaps):.3f} to {max(oldgaps):.3f}. This is a descriptive fresh-family recurrence of the original score signature, not a new population test.','',f'Across the 144 prespecified new-task comparisons, passive−active score gaps ranged from {min(gaps):.3f} to {max(gaps):.3f} nats: {summary["contrasts_at_least4"]} were at least+4, {summary["contrasts_at_most_minus4"]} were at most−4, and {summary["contrasts_within1"]} fell within±1. These comparisons share data and are not independent observations. Variation across the controlled question, code, mapping and position cells prevents treating the old score signature as invariant to the tested context. It does not uniquely identify which internal computation produces it.','',f'Canonical native-baseline eligibility passed in {summary["families_passing_canonical_baseline_eligibility"]}/6 families. The full clean routing gate passed in {summary["families_passing_entire_neutral_routing_gate"]}/6. Ratios from failing units or neutral controls cannot establish a clean relational information route.','',f'The unexpected-answer ledger contains {summary["wrong_unperturbed_first_token_decisions"]} incorrect unperturbed first-token decisions and {summary["wrong_masked_first_token_decisions"]} incorrect masked decisions. Off-format first tokens were {summary["off_format_unperturbed_first_tokens"]} unperturbed and {summary["off_format_masked_first_tokens"]} masked. These are not full generated-response accuracy measurements.','', '**The prospective decisions are unchanged: neither a behavioral nor a routing candidate was selected on development. Held-only regularities cannot be promoted to a confirmatory success.**','']
    report=HERE/'REPORT.md';original=report.read_text();report.write_text(original+'\n'+'\n'.join(text))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
