"""Frozen answer/pair summaries. No fit, exclusions, calibration or p values."""
import csv
import itertools
import json
from collections import defaultdict
from core import *


def main():
    rows=[json.loads(s) for s in (HERE/'ANSWERS.jsonl').read_text().splitlines()]
    assert len(rows)==108 and len({r['id'] for r in rows})==108
    summaries=[]
    for family in sorted({r['family'] for r in rows}):
        for form in ['active','passive','goal_fronted']:
            group=[r for r in rows if (r['family'],r['form'])==(family,form)]
            complete=sum(all(r['native_correct'] for r in group if tuple(r['assignment'])==p) for p in itertools.permutations(range(3)))
            summaries.append(dict(family=family,form=form,answers=len(group),native_correct=sum(r['native_correct'] for r in group),score_correct=sum(r['score_correct'] for r in group),format_compliant=sum(r['format_compliant'] for r in group),three_question_complete=complete,three_question_units=6))
    pairs=eligible_pairs(rows)
    pair_summary=[]
    family_gates=[]
    for family in sorted({r['family'] for r in rows}):
        cross=[p for p in pairs if p['family']==family and p['recipient_form']!=p['donor_form']]
        for invariant in ['sender','sent_person','destination_person']:
            group=[p for p in cross if p['invariant_role']==invariant]
            pair_summary.append(dict(family=family,invariant_role=invariant,cross_form_pairs=len(group),eligible=sum(p['eligible'] for p in group)))
        qualified=[p for p in cross if p['eligible']]
        coverage={form:any(form in [p['recipient_form'],p['donor_form']] for p in qualified) for form in ['active','passive','goal_fronted']}
        roles={q:any(p['invariant_role']==q for p in qualified) for q in ['sender','sent_person','destination_person']}
        family_gates.append(dict(family=family,forms_covered=coverage,invariant_roles_covered=roles,passed=all(coverage.values()) and all(roles.values())))
    controls=[json.loads(s) for s in (HERE/'CONTROLS.jsonl').read_text().splitlines()]
    ready=all(g['passed'] for g in family_gates) and all(c['passed'] for c in controls)
    result=dict(scope='108 fixed development prompts, two AI-authored name families, one local quantized Llama checkpoint. Eligibility only; not causal or population evidence.',native_correct=sum(r['native_correct'] for r in rows),n=108,complete_three_question_units=sum(r['three_question_complete'] for r in summaries),n_three_question_units=36,summary=summaries,pair_summary=pair_summary,family_gates=family_gates,eligible_pairs=sum(p['eligible'] for p in pairs),n_pairs=len(pairs),baseline_ready_for_causal_protocol=ready,decision='BASELINE_READY_FOR_CAUSAL_PROTOCOL' if ready else 'STOP_BASELINE_GATE',scientific_interventions=0,held_out_answers=0,new_p_values=0,technical_checks=len(controls),technical_checks_passed=all(c['passed'] for c in controls))
    save(HERE/'RESULTS.json',result);save(HERE/'PAIR_LEDGER.json',pairs);save(HERE/'UNEXPECTED_ANSWERS.json',[r for r in rows if not r['native_correct'] or not r['score_correct']])
    for name,records in [('ANSWER_LEDGER.csv',rows),('SUMMARY.csv',summaries),('PAIR_SUMMARY.csv',pair_summary)]:
        with (HERE/name).open('w') as f:
            columns=list(records[0]);writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(records)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
