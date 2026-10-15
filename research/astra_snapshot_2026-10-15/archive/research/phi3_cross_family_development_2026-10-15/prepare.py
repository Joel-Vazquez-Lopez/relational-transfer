"""Freeze one local cross-family development diagnostic, without opening new/sealed examples."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
QWEN=ROOT.parent/'stronger_model_development_2026-10-15'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
    assert not (ROOT/'PROTOCOL.json').exists()
    q=json.loads((QWEN/'PROTOCOL.json').read_text())
    p=dict(status='frozen before Phi-3 inference; exploratory reused constructed examples',
        hub_id='mlx-community/Phi-3-mini-4k-instruct-4bit',revision='5b3819ed6317784fb20eddeae9bed984f778d0d0',
        weight_bytes=2149696167,weight_sha256='8d75680621a09474f6601e9176f2f61f92a5e4c079d68d583901f51699fda50a',
        license='MIT; verified community and Microsoft model cards',
        runtime='Existing MLX-LM 0.32.0 / MLX 0.32.4 on Apple M2 16 GiB; free local inference only',
        resource_estimate='11 GiB disk available before download; 2.15 GB weights; estimated 3–5 GB allocation; no old files deleted',
        qwen_protocol_sha256=sha(QWEN/'PROTOCOL.json'),qwen_source_lock_sha256=sha(QWEN/'SOURCE_LOCK.json'),
        qwen_checksums= json.loads((QWEN/'CHECKSUMS.json').read_text()),
        events=q['events'],strong_event_ids=q['strong_event_ids'],forms=q['forms'],
        legacy_questions=q['legacy_questions'],strong_questions=q['strong_questions'],
        strong_system=q['strong_system'],strong_assistant_prefix=q['strong_assistant_prefix'],
        panels={'legacy':'36 events x 4 forms x 4 wordings x 2 candidate orders = 1152; AB matches Qwen semantic prompt content, BA extra within-Phi control',
                'strong_free':'same 12 predetermined events x 4 forms x 2 roles x 2 wordings = 192',
                'strong_list':'same 192 x 2 candidate orders = 384'},
        scoring={'candidate':'logsumexp over complete bare-name and leading-space-name continuation path probabilities; normalize at EACH step, no length normalization; shared whitespace prefix evaluated once per prompt. These paths are disjoint for this tokenizer. Record individual path scores.',
                 'contrast':'actor candidate log probability minus patient candidate log probability; r=(m_actor_question-m_patient_question)/2; negative r means reversal; use signs and correctness, NOT absolute values across families',
                 'generation':'greedy, 12-token maximum on ALL Phi panels; stop on tokenizer EOS or config EOS; strip whitespace only, exact name correctness and compliance; malformed output wrong',
                 'legacy_comparison':'Qwen legacy/list records contain one next-token answer, whereas Phi complete generation can require whitespace plus name. Compare candidate-score choices; retain Phi first-token answers separately and do not equate full Phi generation with Qwen one-token accuracy.',
                 'primary_cross_family':'strong_free full generation, same semantic system/user content, same Answer: prefill, same maximum generation tokens; native chat template differs',
                 'joint':'actor AND patient correct for each event/form/question pair and candidate order',
                 'format_controls':'report full response, first token, exact compliance, full generation length and EOS status; no prompt tuning'},
        analysis={'unit':'paired events across constructions and wordings; bootstrap six participant-pair clusters, keeping both directions',
                  'bootstrap_replicates':10000,'bootstrap_seed':20261015,
                  'primary':'object-cleft minus active joint correctness and question reversal frequency, separately by lexical/explicit wording',
                  'secondary':'all constructions and roles, candidate order, both wrong vs same-participant answers, paired wording gains/losses and fixed heuristic fit',
                  'heuristics':'first/last mention, event grammatical subject/other, first/last as actor then complementary patient, always actor/patient, candidate first/second; no mechanistic inference',
                  'limits':'same previously examined synthetic data; only six name pairs; different size, training, native tokenization/template and quantized weights; no causal architecture/size attribution'},
        guards=['Never read or evaluate NEW_REVIEW.csv or NEW_PROPOSED_LABELS.json','No sealed data','No fine-tuning or paid inference','One selected checkpoint only','Stop on invalid model loading, nonfinite scores, invalid tokenization or memory problems'])
    (ROOT/'PROTOCOL.json').write_text(json.dumps(p,indent=2)+'\n')
