"""Independent score replay and the frozen32-case decision; no model calls."""
import csv,json,collections
from core import *
def main():
    assert json.loads((HERE/'COLLECTION.json').read_text())['status']=='COMPLETE_FIXED_32'
    lock=json.loads((HERE/'LOCK.json').read_text())
    for n,h in lock['files'].items():assert sha(HERE/n)==h,n
    rows=json.loads((HERE/'INPUTS.json').read_text());lookup={r['id']:r for r in rows}
    control=json.loads((HERE/'CONTROL_INPUT.json').read_text());lookup[control['id']]=control
    raw=[json.loads(s) for s in (HERE/'RAW.jsonl').read_text().splitlines()];budget=[json.loads(s) for s in (HERE/'BUDGET.jsonl').read_text().splitlines()]
    assert len(raw)==len(budget)==33 and [r['call'] for r in raw]==list(range(1,34))
    checks=[]
    for r,b in zip(raw,budget):
        assert all(r[k]==b[k] for k in b)
        row=lookup[r['id']];assert sha(HERE/r['payload'])==r['payload_sha256']
        with np.load(HERE/r['payload'],allow_pickle=False) as z:
            logits=z['logits'].astype(np.float64);assert logits.shape==(152064,) and np.isfinite(logits).all()
            best=int(np.argmax(logits));actual=next((k for k,v in row['aliases'].items() if best in v),None)
            s=float(np.logaddexp.reduce(logits[row['aliases']['entailed']])-np.logaddexp.reduce(logits[row['aliases']['not_entailed']]))
            assert actual==r['actual'] and best==r['argmax_id'] and abs(s-r['s_entail'])<=1e-10
            assert r['native_correct']==(actual==row['expected'])
        checks.append(dict(call=r['call'],payload_hash_shape_score_argmax_and_journal_verified=True))
    scientific=[r for r in raw if r['kind']=='scientific'];assert [r['id'] for r in scientific]==[r['id'] for r in rows]
    result=classify(scientific);result['native_correct']=sum(r['native_correct'] for r in scientific);result['format_compliant']=sum(r['format_compliant'] for r in scientific);result['model_forwards']=33
    units=[]
    for f,*_ in FAMILIES:
        for order in ['entailed_first','nonentailed_first']:
            for mapping in [0,1]:
                rr=[r for r in scientific if (r['family'],r['legend_order'],r['mapping'])==(f,order,mapping)];assert len(rr)==4
                units.append(dict(family=f,legend_order=order,mapping=mapping,native_correct=sum(r['native_correct'] for r in rr),complete=all(r['native_correct'] for r in rr)))
    result['four_answer_units']=units;result['complete_units']=sum(u['complete'] for u in units)
    save(HERE/'RESULTS.json',result);save(HERE/'INDEPENDENT_AUDIT.json',dict(passed=True,checks=checks,no_model_calls=True))
    with (HERE/'ANSWER_LEDGER.csv').open('w') as f:
        cols=['call','id','family','role','query','mapping','legend_order','last_class','last_symbol','expected','expected_symbol','actual','argmax_token','native_correct','format_compliant','s_entail','payload','payload_sha256'];w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(scientific)
    save(HERE/'UNEXPECTED_ANSWERS.json',[r for r in scientific if not r['native_correct']])
    text=['# Legend-order diagnostic: completed32-case pilot','','## Fixed decision','',f'Qualifying operational pattern: **{result["candidate"] or "none (mixed/unclassified)"}**. Native first-token correctness:{result["native_correct"]}/32; complete four-answer units:{result["complete_units"]}/8. Causal-task competence gate (all32 correct): **{result["causal_task_ready"]}**.','','|Prediction|Matches /32|Family l01 /16|Family l02 /16|Entailed-first /16|Nonentailed-first /16|Qualifies|','|---|---:|---:|---:|---:|---:|---|']
    for h,c in result['counts'].items():text.append(f'|{h}|{c["matches"]}|{c["by_family"]["l01"]}|{c["by_family"]["l02"]}|{c["by_order"]["entailed_first"]}|{c["by_order"]["nonentailed_first"]}|{c["qualifies"]}|')
    text+=['','Qualification was frozen before outcomes:≥28/32 overall,≥14/16 in each family and each legend order. These are strict operational patterns, not fitted explanations or population tests. All32 cases were fresh, AI-authored/reviewed clauses. Only active source/question syntax and A/B codes were used.','','## Every condition','','|Family|Legend order|Mapping|Correct /4|Complete unit|','|---|---|---:|---:|---|']
    for u in units:text.append(f'|{u["family"]}|{u["legend_order"]}|{u["mapping"]}|{u["native_correct"]}|{u["complete"]}|')
    text+=['','## First-token choices by legend order','','|Legend order|Predicted entailed|Predicted not entailed|Off format|Correct /16|','|---|---:|---:|---:|---:|']
    for order in ['entailed_first','nonentailed_first']:
        rr=[r for r in scientific if r['legend_order']==order];text.append(f'|{order}|{sum(r["actual"]=="entailed" for r in rr)}|{sum(r["actual"]=="not_entailed" for r in rr)}|{sum(r["actual"] is None for r in rr)}|{sum(r["native_correct"] for r in rr)}|')
    text+=['','## Interpretation and limits','','Changing the legend order changes category recency and the last-listed symbol together. Following the last option supports order-sensitive output steering over a fixed negative-category preference in this panel; it does not uniquely identify literal token copying or a neural circuit. A negative-category result is an output-category preference, not proof the model evaluated the relation and rejected it. Correct relational answering is a behavioral criterion, not mechanistic identification. Mixed outcomes remain unclassified.','','This32-case/two-family pilot has no population p value or novelty claim. Four-answer units and individual decisions are kept separate. The all32-correct competence prerequisite is independent of pattern qualification. No causal intervention or extension is authorized by a positive result here. Scores are first-token decisions, not complete generated-response accuracy.','','All33 full output vectors, every wrong/off-format token, source manifests, model/runtime hashes, static controls and resource records are preserved. The one exposed numerical control must pass the inherited .05 full-vector tolerance before fresh cases. The audit independently verified every payload hash, shape, score, argmax and journal entry. The original32-case collection used batch1 and128 tokens, scored at the original assistant end; no answer conditioning or cache was carried between requests.','','A connection to established prompt/answer bias ([Zhao et al.,2021](https://proceedings.mlr.press/v139/zhao21c.html)) remains plausible. This is not a new binding-ID subspace or intervention result of the kind studied by [Dai et al.,2024](https://aclanthology.org/2024.emnlp-main.967/). No comparison establishes novelty.','','**This bounded pilot is complete. No additional inference or adaptive experiment is scheduled.**','']
    (HERE/'REPORT.md').write_text('\n'.join(text));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
