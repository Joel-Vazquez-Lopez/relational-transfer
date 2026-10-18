"""Adversarial scoring + complete production-controller rehearsal, no LLM calls."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import numpy as np
from core import BLOCKS,LENGTH,CONDITIONS,mask,save,toy_calibration
from scoring import full_score

def scoring_fixtures():
    row=dict(names=['Nina','Luke'],expected='Nina',candidate_sequences={'Nina':[[2,3],[4]],'Luke':[[5],[6]]},eos_ids=[7])
    text={2:'N',3:'ina',4:' Nina',5:'Luke',6:' Luke',7:''}
    decode=lambda ids:''.join(text.get(i,'?') for i in ids)
    def logits(prefix,bad=False,unterminated=False):
        x=np.full(8,-20.,dtype=np.float32)
        if not prefix:x[2]=9;x[4]=7;x[5]=1;x[6]=0
        elif tuple(prefix)==(2,):x[7 if bad else 3]=9
        else:x[2 if unterminated else 7]=9
        return x
    good=full_score(lambda p:logits(p),row,decode)
    assert good['argmax_token']=='N' and good['generated_text']=='Nina' and good['correct']
    assert not full_score(lambda p:logits(p,bad=True),row,decode)['correct']
    assert not full_score(lambda p:logits(p,unterminated=True),row,decode)['correct']
    duplicate={**row,'candidate_sequences':{'Nina':[[2,3],[4],[2,3]],'Luke':[[5],[6],[5]]}}
    dup=full_score(lambda p:logits(p),duplicate,decode)
    assert dup['full_name_log_probability']==good['full_name_log_probability']
    expected=sum(float(logits(seq[:j])[t])-float(np.logaddexp.reduce(logits(seq[:j]).astype(np.float64))) for seq in [[2,3]] for j,t in enumerate(seq))
    path=next(x for x in good['candidate_paths']['Nina'] if x['token_ids']==[2,3])
    assert abs(path['log_probability']-expected)<1e-10
    return dict(multi_token_success=True,partial_name_rejected=True,unterminated_rejected=True,
                duplicate_paths_deduplicated=True,teacher_forced_sum_equivalence=True)

def check(rows):
    from transformers import AutoTokenizer
    import run
    root=Path(__file__).resolve().parent.parent.parent
    old=root/'research/source_access_feasibility_2026-10-18'
    r=rows[0];s=r['suffix_start'];lo,hi=r['source_span'];base=np.tri(LENGTH,dtype=bool)
    zero=next(c for c in CONDITIONS if c['name']=='cut_0')
    for c in CONDITIONS:
        for b in range(BLOCKS):
            m=mask(r,c,b)
            assert np.array_equal(m[:s],base[:s]) and m.any(axis=1).all() and np.all(~m|base)
    dependency=np.zeros(LENGTH,dtype=bool);dependency[lo:hi]=True
    for b in range(BLOCKS):
        dependency|=(mask(r,zero,b)&dependency[None,:]).any(axis=1)
        assert not dependency[s:].any()
    paths={'qwen7b':'research/stronger_model_development_2026-10-15/models/qwen7b',
           'llama3b':'research/llama_gemma_cross_family_2026-10-15/model'}
    toks={m:AutoTokenizer.from_pretrained(root/p,local_files_only=True,trust_remote_code=False) for m,p in paths.items()}
    fixtures=scoring_fixtures()
    # Actual saved FIRST-token logits + expressly hypothetical continuations.
    # This verifies the endpoint and is not evidence about the missing completion.
    qrow=next(x for x in rows if x['model']=='qwen7b' and x['id']=='d1_0_agent')
    initial=np.load(old/'qwen7b/CALL_003_logits.npy')
    nseq=next(x for x in qrow['candidate_sequences']['Nina'] if len(x)==2)
    def conditional(prefix):
        if not prefix:return initial
        x=np.full_like(initial,-20.)
        x[nseq[1] if tuple(prefix)==tuple(nseq[:1]) else qrow['eos_ids'][0]]=20.
        return x
    replay=full_score(conditional,qrow,toks['qwen7b'].decode)
    assert replay['correct'] and replay['generated_text']=='Nina'

    class FakeEngine:
        fail=False
        def __init__(self,m,lock):self.model=m;self.tok=toks[m];self.decode=self.tok.decode
        def compute(self,row,condition,prefix=(),wrapped=True,native=False,capture=True):
            target=row['expected']
            if condition['name']=='cut_0':target=row['names'][0]
            if self.fail and row['id']=='d1_0_agent':target=row['names'][1]
            vocab=max([i for seqs in row['candidate_sequences'].values() for seq in seqs for i in seq]+row['eos_ids'])+1
            def distribution(pref):
                x=np.full(vocab,-20.,dtype=np.float32)
                if not pref:
                    for name,seqs in row['candidate_sequences'].items():
                        value=10. if name==target else 0.
                        if condition['name'] in ['cut_7','cut_14','cut_21'] and name==target:
                            k=condition['cut'];value-=({7:1.,14:3.,21:.3}[k] if row['query']=='agent' else 0.)
                        for seq in seqs:x[seq[0]]=max(x[seq[0]],value)
                    # Prefer the no-leading-space path for actual greedy completion.
                    no_space=self.tok.encode(target,add_special_tokens=False);x[no_space[0]]+=.1
                else:
                    next_ids=[seq[len(pref)] for seqs in row['candidate_sequences'].values() for seq in seqs if len(seq)>len(pref) and tuple(seq[:len(pref)])==tuple(pref)]
                    x[next_ids[0] if next_ids else row['eos_ids'][0]]=20.
                return x
            arrays={}
            if capture and wrapped:
                last=np.ones((28,4,4),dtype=np.float32);last[:,1:3]=0
                arrays={'last':last,'prefix':np.ones((28,row['suffix_start'],4),dtype=np.float32)}
                if condition['name']=='cut_0':arrays['suffix']=np.ones((28,row['prompt_end']-row['suffix_start']+1,4),dtype=np.float32)
            return distribution(prefix),arrays,distribution(())

    old_here,old_check=run.HERE,run.assert_lock
    with tempfile.TemporaryDirectory() as td:
        temp=Path(td);save(temp/'INPUTS.json',rows)
        lock=dict(max_forwards=2000,max_per_model=1000,models={m:dict(path=p) for m,p in paths.items()})
        save(temp/'LOCK.json',lock)
        run.HERE=temp;run.assert_lock=lambda:lock
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                q=run.execute('qwen7b',FakeEngine)
                assert q['evaluation_supported'] and q['prediction']['cut']==14,q
                l=run.execute('llama3b',FakeEngine)
                assert l['evaluation_supported'],l
            journal=json.loads((temp/'BUDGET.json').read_text())
            raw=json.loads((temp/'qwen7b/RAW.json').read_text())['rows']
            assert len(journal)>len(raw) and any(x['prefix_token_ids'] for x in journal)
            assert all((temp/x['model']/f"CALL_{x['call']:04d}_logits.npy").exists() for x in journal)
            assert len({x['call'] for x in journal})==len(journal)
            assert (temp/'qwen7b/PREDICTION_BEFORE_EVALUATION.json').exists()
            full_pipeline_calls=len(journal)
            # A repeated attempt must fail before any new dispatch.
            try:
                with contextlib.redirect_stdout(io.StringIO()):run.execute('qwen7b',FakeEngine)
            except AssertionError:pass
            else:raise AssertionError('Repeated run unexpectedly allowed')
            assert len(json.loads((temp/'BUDGET.json').read_text()))==len(journal)
        finally:run.HERE=old_here;run.assert_lock=old_check
    with tempfile.TemporaryDirectory() as td:
        temp=Path(td);save(temp/'INPUTS.json',rows);save(temp/'LOCK.json',lock)
        run.HERE=temp;run.assert_lock=lambda:lock;FakeEngine.fail=True
        try:
            with contextlib.redirect_stdout(io.StringIO()):decision=run.execute('qwen7b',FakeEngine)
            assert decision['status']=='STOP_DEVELOPMENT_COMPETENCE'
            journal=json.loads((temp/'BUDGET.json').read_text())
            assert not any(x['condition'].startswith('cut_') for x in journal)
            assert not (temp/'qwen7b/PREDICTION_BEFORE_EVALUATION.json').exists()
        finally:run.HERE=old_here;run.assert_lock=old_check;FakeEngine.fail=False
    return dict(status='PASS_SYNTHETIC_ONLY',target_model_calls=0,scoring_fixtures=fixtures,
        real_saved_first_logits_hypothetical_continuation_fixture=True,
        hypothetical_fixture_is_not_empirical_completion=True,all_suffix_information_path_check=True,
        complete_production_controller_two_models_and_evaluation=True,
        simulated_dispatches=full_pipeline_calls,continuations_counted_and_persisted=True,
        repeated_attempt_rejected=True,negative_competence_gate_prevents_cuts=True,
        known_programs=toy_calibration())
