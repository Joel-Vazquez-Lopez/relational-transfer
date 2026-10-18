"""Offline preflight guard for future single-token designs, not this closed run.

This file does not alter frozen inputs, scores, gates or labels. It rejects the
inadequate endpoint before inference and includes the missed adversarial fixture.
"""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def validate_forms(name,forms):
    required=[name,' '+name]
    missing=[s for s in required if s not in forms]
    multi=[s for s in required if s in forms and len(forms[s])!=1]
    return dict(valid=not missing and not multi,missing=missing,unsupported=multi)

def main():
    # The previously accepted leading-space alias must not rescue an unsupported
    # no-leading-space realization. Highest first-part token is not a full answer.
    bad=validate_forms('Nina',{'Nina':[45,2210],' Nina':[63888]})
    assert not bad['valid'] and bad['unsupported']==['Nina']
    assert validate_forms('Luke',{'Luke':[58347],' Luke':[24453]})['valid']
    highest_token=45
    assert highest_token==[45,2210][0] and len([45,2210])>1
    # The fixture is rejected at design level, with no claim about completion.
    rows=json.loads((HERE/'TOKENIZER_AUDIT.json').read_text())
    failures=[]
    for r in rows:
        result=validate_forms(r['name'],r['forms'])
        if not result['valid']: failures.append(dict(model=r['model'],name=r['name'],**result))
    assert any(r['model']=='qwen7b' and r['name']=='Nina' for r in failures)
    report=dict(status='OFFLINE_GUARD_FIX_VERIFIED',target_model_calls=0,
        frozen_results_changed=False,adversarial_multitoken_fixture_pass=True,
        original_manifest_rejected_before_inference=True,unsupported_realizations=failures,
        scope='Prevents reuse of the faulty single-token design; does not implement a full-name sequence scorer or authorize a new run')
    (HERE/'SCORING_GUARD_TESTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],len(failures),'unsupported model/name entries; zero model calls')

if __name__=='__main__':main()
