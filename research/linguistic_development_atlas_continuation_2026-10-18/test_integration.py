"""Complete saving/analysis rehearsal with synthetic states; zero model calls."""
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from transformers import AutoTokenizer
import run
import analyze

HERE=Path(__file__).resolve().parent


def synthetic_capture(data):
    ids=data['token_ids'];positions=data['retained_positions']
    embed=[];prefix=[]
    for p in positions:
        e=hashlib.sha256(str(ids[p]).encode()).digest()[:7]
        h=hashlib.sha256(json.dumps(ids[:p+1]).encode()).digest()[:7]
        embed.append(np.frombuffer(e,dtype=np.uint8).astype(np.float32)-128)
        prefix.append(np.frombuffer(h,dtype=np.uint8).astype(np.float32)-128)
    e=np.stack(embed);h=np.stack(prefix)
    residual=e[None]+np.arange(29,dtype=np.float32)[:,None,None]*h[None]
    attention=np.broadcast_to(h[None]+3,(28,len(positions),7)).copy()
    mlp=np.full_like(attention,-3)
    assert np.array_equal(residual[1:]-residual[:-1],attention+mlp)
    return dict(residual=residual,attention=attention,mlp=mlp,
                reconstruction_error_norm=np.zeros((28,len(positions)),dtype=np.float64))


def main():
    rows,tokens=run.inputs();checks={};metadata=json.loads((run.DESIGN/'TOKENIZER_AUDIT.json').read_text())['models']
    with tempfile.TemporaryDirectory(prefix='atlas_integration_') as name:
        tmp=Path(name);(tmp/'LOCK.json').write_text('{}')
        (tmp/'ANNOTATION_DECISION.json').write_text(json.dumps({'status':'SYNTHETIC_TEST_ONLY_NO_MODEL_CALLS'}))
        for model in ['qwen7b','llama3b']:
            out=tmp/model;out.mkdir();raw=[]
            t=AutoTokenizer.from_pretrained(run.ROOT/run.MODELS[model],local_files_only=True,trust_remote_code=False)
            aliases=metadata[model]['candidate_alias_ids'];vocab=len(t)
            for row in rows:
                logits=np.full(vocab,-8,dtype=np.float32)
                logits[aliases[row['expected']][row['expected']][0]]=20
                arrays=synthetic_capture(tokens[model,row['id']])
                result=run.persist_result(out,model,row,logits,arrays,0.,aliases,t.decode,raw)
                assert result['expected']==row['expected'] and result['pass_competence']
                loaded=json.loads((out/'RAW.json').read_text())['rows'][-1]
                assert loaded==result
                with np.load(out/(row['id']+'.npz'),allow_pickle=False) as z:
                    for k,v in arrays.items():assert np.array_equal(z[k],v)
                assert np.array_equal(np.load(out/(row['id']+'_logits.npy'),allow_pickle=False),logits)
            assert len(raw)==64
            # Deliberately reject duplicate records and scorer/input disagreement before writing.
            try:run.persist_result(out,model,rows[-1],logits,arrays,0.,aliases,t.decode,raw)
            except AssertionError:pass
            else:raise AssertionError('Duplicate main record accepted')
            wrong='No' if rows[-1]['expected']=='Yes' else 'Yes'
            with patch.object(run,'score',return_value={'expected':wrong}):
                try:run.persist_result(out,model,rows[-1],logits,arrays,0.,aliases,t.decode,[])
                except AssertionError:pass
                else:raise AssertionError('Gold mismatch accepted')
            ctrl={split:analyze.technical_controls(model,[r for r in rows if r['split']==split],out,tokens)
                  for split in ['development','held_out']}
            assert all(c['prefix_pass'] and c['components_pass'] for c in ctrl.values())
            run.save(out/'CONTROLS.json',ctrl)
            run.save(out/'COMPLETE.json',dict(status='SYNTHETIC_ONLY',total_forwards=0))
        checks['all_128_actual_input_rows_scored_assembled_saved_reloaded']=True
        checks['duplicate_main_record_rejected']=True
        checks['scorer_input_gold_disagreement_rejected']=True
        checks['causal_prefix_controls_on_causal_synthetic_states']=True
        # Rehearse the entire production analysis/export path on known synthetic captures.
        with patch.object(run,'HERE',tmp),patch.object(analyze,'HERE',tmp),patch.object(run,'assert_lock',return_value={}):
            analyze.main()
        summary=json.loads((tmp/'SUMMARY.json').read_text())
        assert summary['model_forwards']==0
        assert all(s['correct_answers']==64 and len(s['groups'])==16 for s in summary['models'].values())
        assert (tmp/'DEVELOPMENT_MAP.csv').stat().st_size>0
        assert len(list(tmp.glob('*.png')))==6
        checks['whole_analysis_json_csv_grams_and_six_figures_exported']=True
        # Actual previously captured arrays and logits, only read/serialized offline.
        old=run.ORIGINAL/'qwen7b';r=rows[0]
        actual=np.load(old/(r['id']+'_logits.npy'),allow_pickle=False)
        with np.load(old/(r['id']+'.npz'),allow_pickle=False) as z:captured={k:z[k].copy() for k in z.files}
        realout=tmp/'actual_saved_capture';realout.mkdir()
        tok=AutoTokenizer.from_pretrained(run.ROOT/run.MODELS['qwen7b'],local_files_only=True,trust_remote_code=False)
        x=run.persist_result(realout,'qwen7b',r,actual,captured,0.0005146823373110973,
                             metadata['qwen7b']['candidate_alias_ids'],tok.decode,[])
        assert x['argmax_token']=='Yes' and x['pass_competence']
        checks['actual_saved_capture_complete_result_path']=True
    checks['additional_model_calls']=0
    (HERE/'INTEGRATION_TEST_RESULTS.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks),flush=True)


if __name__=='__main__':main()
