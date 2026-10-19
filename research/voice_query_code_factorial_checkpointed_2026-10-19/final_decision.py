"""Editorial scientific conclusion from the frozen decisions and their limits."""
import json
from core import HERE,save

def main():
    r=json.loads((HERE/'RESULTS.json').read_text())
    s=json.loads((HERE/'DESCRIPTIVE_SUMMARY.json').read_text())
    assert r['all_checks_passed']
    decision='NARROW' if r['held_behavior_replicated'] or r['held_routing_replicated'] else 'NO-GO'
    correct=sum(s['native_cells']['factorial_'+v]['correct'] for v in ('active','passive'))
    n=sum(s['native_cells']['factorial_'+v]['answers'] for v in ('active','passive'))
    verdict={'decision':decision,'scope':'Current assay does not support a distinct linguistic binding or event-selection mechanism.','prespecified_behavior_replicated':r['held_behavior_replicated'],'prespecified_routing_replicated':r['held_routing_replicated'],'counterbalanced_first_token_correct':correct,'counterbalanced_first_token_answers':n,'counterbalanced_first_token_accuracy':correct/n,'clean_routing_families':s['families_passing_entire_neutral_routing_gate'],'families':6,'interpretation':'The old construction-dependent output signature recurs. Poor native performance in the crossed task and failed clean-routing gates prevent a clean mechanistic interpretation. They do not identify whether failures arise from role processing, question interpretation, legend use or output preparation.','classification_is_scientific_editorial_judgment_not_additional_test':True,'collection_complete':True,'no_further_inference':True}
    save(HERE/'SCIENTIFIC_DECISION.json',verdict)
    section=f'''## Mechanistic conclusion: {decision}

The collection and technical audits completed successfully. The counterbalanced assay produced **{correct}/{n} correct first-token decisions ({correct/n:.1%})**, and **{s['families_passing_entire_neutral_routing_gate']}/6 families** passed the full clean-routing gate. Neither prespecified pattern family selected a development candidate or achieved prospective replication.

The original-prompt comparison recurred: active sentences gave14/24 correct first-token decisions and0/6 complete four-answer units; passive sentences gave24/24 and6/6. The stricter controls leave a real construction-dependent output phenomenon but no cleanly identified mechanism. In particular, the low crossed-task performance prevents attributing score changes uniquely to relational computation rather than question interpretation, answer-code use or output preparation. This is the decision about the current assay; no population significance or novelty claim is supported.

The figure includes every frozen contrast. Column labels “active/passive” refer to **question voice**; colors show passive-sentence minus active-sentence entailed-score gaps. Prefix0 is empty, prefix1 is `Record: `, and `matched_period` compares passive prefix0 with active prefix1. Mapping0 is normal; mapping1 is reversed. The dashed line separates development and held-out families.

![All frozen factorial contrasts](FACTORIAL_GAPS.png)

'''
    report=HERE/'REPORT.md';text=report.read_text();heading,rest=text.split('\n',1);report.write_text(heading+'\n\n'+section+rest)
    print(json.dumps(verdict,indent=2))

if __name__=='__main__':main()
