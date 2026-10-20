"""Prespecified all-answer and bidirectional-unit accounting; no threshold fitting."""
import csv,json
from core import *
HYP=['role_table','mention_order','fixed_one_answer','syntax_aware_parser','complete_future_answer_table']
COND=['main','same_role_voice','neighbor','cross_lexical']
def main():
    a=[json.loads(x) for x in (HERE/'ANSWERS.jsonl').read_text().splitlines()];assert len(a)==144
    controls=[json.loads(x) for x in (HERE/'CONTROLS.jsonl').read_text().splitlines()]
    units=json.loads((HERE/'UNITS.json').read_text());unitout=[];summary=[]
    for c in COND:
        rows=[r for r in a if r['condition']==c]
        summary.append(dict(condition=c,n=len(rows),terminated=sum(r['terminated'] for r in rows),recipient_name_compliant=sum(r['format_compliant'] for r in rows),donor_name_compliant=sum(r['donor_name_compliant'] for r in rows),preserves_recipient_answer=sum(r['native_correct'] for r in rows),**{h:sum(r['hypothesis_matches'][h] for r in rows) for h in HYP}))
        for u in units:
            six=[r for r in rows if r['unit']==u['unit']];assert len(six)==6
            directions={d:{h:all(r['hypothesis_matches'][h] for r in six if r['direction']==d) for h in HYP} for d in ['forward','reverse']}
            unitout.append(dict(unit=u['unit'],condition=c,n=6,complete_account={h:all(r['hypothesis_matches'][h] for r in six) for h in HYP},direction_complete=directions,unexpected=[dict(id=r['id'],text=r['generated_text'],terminated=r['terminated'],predictions=r['predictions']) for r in six if not r['hypothesis_matches']['role_table']]))
    technical=all(r['passed'] for r in controls)
    endpoint=[r for r in a if r['condition'] in ['main','same_role_voice']]
    exact={h:all(r['hypothesis_matches'][h] for r in endpoint) for h in HYP}
    neighbor=sum(u['complete_account']['role_table'] for u in unitout if u['condition']=='neighbor')
    qualified=technical and exact['role_table'] and neighbor<=2
    decision='DEVELOPMENT_ROLE_CANDIDATE_QUALIFIED_FOR_FROZEN_HELDOUT' if qualified else ('NARROW_MENTION_ORDER_STEERING' if technical and exact['mention_order'] else 'NO_GO_FIXED_DEVELOPMENT_CANDIDATE')
    save(HERE/'RESULTS.json',dict(decision=decision,role_candidate_qualified=qualified,technical_pass=technical,technical_checks=len(controls),joint_72_answer_accounts=exact,neighbor_complete_bidirectional_units=neighbor,summary=summary,units=unitout,scope='One chosen block/site and six dependent development units in two AI-only template families. Not statistical significance or unique semantic algorithm. Syntax-aware parser and complete future-answer table remain equivalent. No adaptive site search.'))
    with (HERE/'SUMMARY.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    with (HERE/'ANSWER_LEDGER.csv').open('w') as f:
        fields=['id','unit','direction','condition','query_role','invariant_role','gold','generated_text','terminated','native_correct','donor_native_correct']+HYP
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in a:w.writerow({**{k:r[k] for k in fields if k not in HYP},**r['hypothesis_matches']})
    save(HERE/'UNEXPECTED_ANSWERS.json',[r for r in a if not r['hypothesis_matches']['role_table']])
    print(json.dumps(dict(decision=decision,summary=summary),indent=2))
if __name__=='__main__':main()
