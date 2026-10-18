"""Offline recovery of saved files from the aborted run; never imports MLX.

This does not repair/resume the frozen runner or execute any model call.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent


def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    spec=importlib.util.spec_from_file_location('frozen_runner',HERE/'run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.assert_lock()
    out=HERE/'qwen7b';rows,tokens=module.inputs()
    row=next(r for r in rows if r['id']=='a01_within_0_agent')
    logits=np.load(out/(row['id']+'_logits.npy'),allow_pickle=False)
    aliases=json.loads((module.DESIGN/'TOKENIZER_AUDIT.json').read_text())['models']['qwen7b']['candidate_alias_ids']
    tokenizer=AutoTokenizer.from_pretrained(module.ROOT/module.MODELS['qwen7b'],local_files_only=True,trust_remote_code=False)
    score=module.score(logits,row,aliases,tokenizer.decode)
    assert score['expected']==row['expected']
    with np.load(out/(row['id']+'.npz'),allow_pickle=False) as z:
        shapes={k:list(z[k].shape) for k in z.files}
        recon=float(np.max(z['reconstruction_error_norm']/np.maximum(
            np.linalg.norm(z['residual'][1:].astype(np.float64),axis=-1),1e-12)))
        assert all(np.isfinite(z[k]).all() for k in z.files)
    partial=dict(status='recovered_offline_from_saved_arrays_no_model_rerun',row=row,
        tokenization=tokens['qwen7b',row['id']],score=score,array_shapes=shapes,
        max_native_reconstruction_relative_error=recon,
        warning='Single development cell only. Wrapper/repeat, causal-prefix and family competence gates were not completed. No valid linguistic comparison.')
    save('qwen7b/PARTIAL_OUTPUT.json',partial)
    journal=json.loads((out/'FORWARD_JOURNAL.json').read_text());assert len(journal)==2
    # Reproduce the serialization fault using only the recovered Python dictionaries.
    reproduced=False
    try:dict(**row,**score,max_reconstruction_relative_error=recon)
    except TypeError as e:
        assert "multiple values for keyword argument 'expected'" in str(e)
        reproduced=True
    assert reproduced
    proposed={**row,**score,'max_reconstruction_relative_error':recon}
    assert proposed['expected']==row['expected']
    assert all(proposed[k]==v for k,v in score.items())
    save('OFFLINE_REPAIR_CHECK.json',dict(
        duplicate_keyword_failure_reproduced=True,
        proposed_dictionary_literal_preserves_every_original_scorer_field=True,
        expected_field_agreement_asserted=True,
        frozen_runner_modified=False,repair_applied=False,new_model_forwards=0))
    save('EXECUTION_STATUS.json',dict(
        status='CLOSED_TECHNICAL_FAILURE_NO_GO',conclusion='NO-GO for this execution; linguistic hypotheses untested',
        cause="run.py line 167: dict(**row, **score(...)) duplicates the expected keyword",
        responsibility='Codex implementation defect; preflight did not cover merging a real-shaped row and score',
        total_model_forwards=2,maximum_authorized=132,
        main_capture_forwards=1,unwrapped_validation_forwards=1,wrapped_repeat_forwards=0,
        unwrapped_validation_output='Computed but not persisted before the process failed; unavailable for numerical comparison',
        qwen_saved_answer=dict(id=row['id'],expected=score['expected'],argmax=score['argmax_token'],
                              correct=score['pass_competence'],gold_margin=score['gold_margin']),
        qwen_family_competence='Not evaluated; 0 complete 16-answer families',
        llama='Not started: same known serialization defect; no inference spent reproducing it',
        held_out_forwards=0,paid_calls=0,downloads=0,
        annotation_status='human_explicitly_waived; AI-only annotation',
        native_single_capture_reconstruction_relative_error=recon,
        numerical_gate='Incomplete; no wrapper/unwrapped comparison, no repeat or prefix checks',
        full_atlas_analysis='Not run: no complete four-cell comparison exists',
        frozen_source_hashes_verified=True,no_post_output_code_changes=True,
        repair='One-line mechanical fix proposed separately and checked offline; not applied to frozen code',
        further_inference_started=False,run_closed=True))
    print(json.dumps(dict(status='offline_recovery_complete',total_model_forwards=2,
        saved_argmax=score['argmax_token'],reconstruction=recon,frozen_source_unchanged=True)))


if __name__=='__main__':main()
