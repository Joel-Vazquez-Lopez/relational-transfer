"""Posthoc descriptive role-confusion/answer summaries; no fits or new tests."""
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    rows=[json.loads(s) for s in (HERE/'ANSWERS.jsonl').read_text().splitlines()]
    inputs=json.loads((HERE/'INPUTS.json').read_text());lookup={r['id']:r for r in inputs}
    roles=['sender','sent_person','destination_person']
    table=[]
    for family in ['t01','t02','pooled']:
        for form in ['active','passive','goal_fronted']:
            for query in roles:
                group=[r for r in rows if (family=='pooled' or r['family']==family) and r['form']==form and r['query_role']==query]
                counts=Counter()
                for r in group:
                    item=lookup[r['id']];actual=r['actual']
                    assignment=[item['names'][i] for i in item['assignment']]
                    label=roles[assignment.index(actual)] if actual in assignment else 'off_format'
                    counts[label]+=1
                table.append(dict(family=family,form=form,question_role=query,n=len(group),actual_source_role=dict(counts),native_correct=sum(r['native_correct'] for r in group),score_correct=sum(r['score_correct'] for r in group)))
    disagreements=[dict(id=r['id'],native_correct=r['native_correct'],score_correct=r['score_correct'],text=r['generated_text'],gold=r['gold'],margin=r['full_name_margin']) for r in rows if r['native_correct']!=r['score_correct']]
    output=dict(scope='Posthoc descriptive characterization of already collected development outcomes; not a new prediction, fit, significance test or causal interpretation.',confusion_table=table,native_score_disagreements=disagreements,unanswered=sum(not r['terminated'] for r in rows),format_compliant=sum(r['format_compliant'] for r in rows))
    (HERE/'DESCRIPTIVE_CONFUSIONS.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(pooled=[x for x in table if x['family']=='pooled'],native_score_disagreements=len(disagreements),unanswered=output['unanswered']),indent=2))

if __name__=='__main__':main()
