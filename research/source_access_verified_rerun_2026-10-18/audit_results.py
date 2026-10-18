"""Independent offline audit of saved sequence probabilities and greedy tokens.

Reporting/integrity tool; no changes to scoring, gates or target-model outputs.
Uses logaddexp reduction rather than the production max/logsumexp helper.
"""
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from core import sha,save

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent

def main():
    lock=json.loads((HERE/'LOCK.json').read_text())
    unchanged={n:sha(HERE/n)==h for n,h in lock['files'].items()}
    assert all(unchanged.values())
    old=ROOT/'research/source_access_feasibility_2026-10-18'
    old_lock=json.loads((old/'LOCK.json').read_text())
    assert all(sha(old/n)==h for n,h in old_lock['files'].items())
    budget=json.loads((HERE/'BUDGET.json').read_text())
    assert [x['call'] for x in budget]==list(range(1,len(budget)+1))
    assert len(budget)<=lock['max_forwards']
    assert all((HERE/x['model']/f"CALL_{x['call']:04d}_logits.npy").exists() for x in budget)
    inputs={(r['model'],r['id']):r for r in json.loads((HERE/'INPUTS.json').read_text())}
    score_error=0.; audited=0;model_counts={};prediction_hashes={}
    for model,source in lock['models'].items():
        model_counts[model]=sum(x['model']==model for x in budget)
        assert model_counts[model]<=lock['max_per_model']
        p=HERE/model
        if not (p/'RAW.json').exists():continue
        tok=AutoTokenizer.from_pretrained(ROOT/source['path'],local_files_only=True,trust_remote_code=False)
        raw=json.loads((p/'RAW.json').read_text())['rows']
        if (p/'PREDICTION_BEFORE_EVALUATION.json').exists():
            import hashlib
            prediction=json.loads((p/'PREDICTION_BEFORE_EVALUATION.json').read_text())
            developmental=dict(model=model,rows=[r for r in raw if r['split']=='development'])
            h=hashlib.sha256((json.dumps(developmental,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
            assert h==prediction['raw_sha256']
            prediction_hashes[model]=dict(matches_development_only_raw=True,raw_sha256=h)
        for r in raw:
            if r['correct'] is None:continue
            row=inputs[model,r['id']]
            mapping={():r['call'],**{tuple(x['prefix']):x['call'] for x in r['child_calls']}}
            arrays={prefix:np.load(p/f'CALL_{call:04d}_logits.npy').astype(np.float64) for prefix,call in mapping.items()}
            scores={}
            for name,paths in r['candidate_paths'].items():
                expected_paths={tuple(x) for x in row['candidate_sequences'][name]}
                assert {tuple(x['token_ids']) for x in paths}==expected_paths
                recomputed=[]
                for path in paths:
                    seq=path['token_ids'];terms=[]
                    for j,token in enumerate(seq):
                        logits=arrays[tuple(seq[:j])]
                        terms.append(float(logits[token]-np.logaddexp.reduce(logits)))
                    value=sum(terms)
                    score_error=max(score_error,abs(value-path['log_probability']),
                        max(abs(a-b) for a,b in zip(terms,path['token_log_probabilities'])))
                    recomputed.append(value)
                scores[name]=float(np.logaddexp.reduce(recomputed))
                score_error=max(score_error,abs(scores[name]-r['full_name_log_probability'][name]))
            rival=next(x for x in row['names'] if x!=row['expected'])
            margin=scores[row['expected']]-scores[rival]
            score_error=max(score_error,abs(margin-r['margin']))
            tokens=r['generated_token_ids']
            for j,token in enumerate(tokens):assert int(np.argmax(arrays[tuple(tokens[:j])]))==token
            if r['terminated']:
                assert r['eos_id'] in row['eos_ids']
                assert int(np.argmax(arrays[tuple(tokens)]))==r['eos_id']
            else:assert len(tokens)==4
            assert tok.decode(tokens)==r['generated_text']
            expected_correct=bool(r['terminated'] and tok.decode(tokens).strip()==row['expected'] and margin>0)
            assert expected_correct==r['correct']
            audited+=1
    assert score_error<1e-8,score_error
    result=dict(status='PASS_INDEPENDENT_SAVED_OUTPUT_AUDIT',target_model_calls=0,
        audited_scientific_records=audited,max_sequence_probability_error=score_error,
        all_generated_tokens_and_observed_EOS_verified=True,all_payloads_present=True,
        frozen_rerun_files_unchanged=unchanged,original_frozen_attempt_unchanged=True,
        new_attempted_forwards=len(budget),prior_failed_attempt_forwards=11,
        cumulative_source_access_attempts=11+len(budget),model_counts=model_counts)
    result['pre_evaluation_prediction_hashes']=prediction_hashes
    save(HERE/'INDEPENDENT_OUTPUT_AUDIT.json',result)
    print('Independent output audit passed:',audited,'records;',len(budget),'new forwards; error',score_error)

if __name__=='__main__':main()
