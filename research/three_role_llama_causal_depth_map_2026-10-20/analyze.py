"""Frozen depth-profile accounting, all answers and dependent units retained."""
import csv,json
import numpy as np
from core import *
HYP=['role_table','mention_order','fixed_one_answer','syntax_aware_parser','complete_future_answer_table']
COND=['main','same_role_voice','neighbor','cross_lexical']
DEPTHS=[7,14,21]

def gates(answers,technical):
    out=[]
    for block in DEPTHS:
        rows=[a for a in answers if a['block']==block]
        endpoint=[a for a in rows if a['condition'] in ['main','same_role_voice']]
        exact={h:len(endpoint)==72 and all(a['hypothesis_matches'][h] for a in endpoint) for h in HYP}
        neighbor=sum(all(a['hypothesis_matches']['role_table'] for a in rows if a['unit']==u and a['condition']=='neighbor') for u in sorted({a['unit'] for a in rows}))
        qualified=bool(technical and exact['role_table'] and neighbor<=2)
        out.append(dict(block=block,joint_72_answer_accounts=exact,neighbor_complete_bidirectional_units=neighbor,role_candidate_qualified=qualified,mention_account_qualified=bool(technical and exact['mention_order'])))
    qualifying=[o['block'] for o in out if o['role_candidate_qualified']]
    decision='DEVELOPMENT_ROLE_CANDIDATE_QUALIFIED_FOR_FROZEN_HELDOUT' if qualifying else ('NARROW_MENTION_ORDER_STEERING' if any(o['mention_account_qualified'] for o in out) else 'NO_GO_FIXED_DEPTH_PROFILE')
    return dict(decision=decision,depth_gates=out,qualifying_blocks=qualifying,selected_block=min(qualifying) if qualifying else None)

def main():
    a=[json.loads(x) for x in (HERE/'ANSWERS.jsonl').read_text().splitlines()];assert len(a)==432
    controls=[json.loads(x) for x in (HERE/'CONTROLS.jsonl').read_text().splitlines()]
    units=json.loads((HERE/'UNITS.json').read_text());unitout=[];summary=[]
    for block in DEPTHS:
        for c in COND:
            rows=[r for r in a if r['condition']==c and r['block']==block];assert len(rows)==36
            margins=[r['candidate_log_probabilities'][r['predictions']['role_table']]-r['candidate_log_probabilities'][r['predictions']['mention_order']] for r in rows if all(r['predictions'][h] in r['candidate_log_probabilities'] for h in ['role_table','mention_order'])]
            switched=[r for r in rows if r['query_role']!=r['invariant_role']];invariant=[r for r in rows if r['query_role']==r['invariant_role']]
            summary.append(dict(block=block,condition=c,n=len(rows),terminated=sum(r['terminated'] for r in rows),recipient_name_compliant=sum(r['format_compliant'] for r in rows),donor_name_compliant=sum(r['donor_name_compliant'] for r in rows),preserves_recipient_answer=sum(r['native_correct'] for r in rows),role_match_switched=sum(r['hypothesis_matches']['role_table'] for r in switched),n_switched=len(switched),role_match_invariant=sum(r['hypothesis_matches']['role_table'] for r in invariant),n_invariant=len(invariant),mean_role_minus_mention_logprob=float(np.mean(margins)) if margins else None,n_margin=len(margins),**{h:sum(r['hypothesis_matches'][h] for r in rows) for h in HYP}))
            for u in units:
                six=[r for r in rows if r['unit']==u['unit']];assert len(six)==6
                directions={d:{h:all(r['hypothesis_matches'][h] for r in six if r['direction']==d) for h in HYP} for d in ['forward','reverse']}
                unitout.append(dict(block=block,unit=u['unit'],condition=c,n=6,complete_account={h:all(r['hypothesis_matches'][h] for r in six) for h in HYP},direction_complete=directions,unexpected=[dict(id=r['id'],text=r['generated_text'],terminated=r['terminated'],predictions=r['predictions']) for r in six if not r['hypothesis_matches']['role_table']]))
    technical=all(r['passed'] for r in controls)
    result=dict(**gates(a,technical),technical_pass=technical,technical_checks=len(controls),summary=summary,units=unitout,scope='Three prospectively fixed depths, six dependent development units in two AI-only template families. No statistical significance or unique semantic algorithm. Syntax-aware parser and complete future-answer table remain equivalent. No adaptive site search.')
    save(HERE/'RESULTS.json',result)
    with (HERE/'SUMMARY.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    with (HERE/'ANSWER_LEDGER.csv').open('w') as f:
        fields=['id','block','unit','direction','condition','query_role','invariant_role','gold','generated_text','terminated','native_correct','donor_native_correct']+HYP
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in a:w.writerow({**{k:r[k] for k in fields if k not in HYP},**r['hypothesis_matches']})
    save(HERE/'UNEXPECTED_ANSWERS.json',[r for r in a if not r['hypothesis_matches']['role_table']])
    print(json.dumps({k:v for k,v in result.items() if k not in ['units','scope']},indent=2))
if __name__=='__main__':main()
