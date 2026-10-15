"""Generate the bounded exploratory protocol; refuse overwriting it."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'qwen_role_equivariance_diagnostic_2026-10-12'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def sentence(e, form):
    a, p, v = e['actor'], e['patient'], e['verb']
    return {'active': f'{a} {v} {p}.', 'passive': f'{p} was {v} by {a}.',
            'subject_cleft': f'It was {a} who {v} {p}.',
            'object_cleft': f'It was {p} that {a} {v}.'}[form]

def build():
    assert not (ROOT / 'PROTOCOL.json').exists()
    old = json.loads((OLD / 'PROTOCOL.json').read_text())
    events = []
    for i, (a,b) in enumerate(old['pairs']):
        for verb in old['verbs']:
            for direction, (actor,patient) in enumerate(((a,b),(b,a))):
                events.append(dict(id=f'p{i}-{verb}-d{direction}', pair_index=i,
                                   a=a,b=b,actor=actor,patient=patient,verb=verb,direction=direction))
    subset = [e for e in events if e['verb'] == old['verbs'][e['pair_index'] % 3]]
    assert len(events)==36 and len(subset)==12
    models = {
        'qwen05': ['mlx-community/Qwen2.5-0.5B-Instruct-4bit','a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3'],
        'qwen15': ['mlx-community/Qwen2.5-1.5B-Instruct-4bit','8b403126fc14f14cfc99bb4cfa72ecbc129ea677'],
        'qwen7b': ['mlx-community/Qwen2.5-7B-Instruct-4bit','c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed']}
    protocol = dict(status='frozen before new inference; development/exploratory only',
        models=models, model_order=['qwen05','qwen15','qwen7b'], events=events,
        strong_event_ids=[e['id'] for e in subset], forms=old['forms'], legacy_questions=old['questions'],
        strong_questions={
            'lexical': {'actor':'Who {verb} someone in this sentence?',
                        'patient':'Who was {verb} by someone in this sentence?'},
            'explicit': {'actor':'Which person does the sentence describe as carrying out the action?',
                         'patient':'Which person does the sentence describe as having the action done to them?'}},
        strong_system="Read the sentence and answer the question about that event. Return only the person's name, copied exactly from the sentence. Do not explain.",
        strong_assistant_prefix='Answer:',
        panels={'legacy': '36 events x 4 forms x 4 archived questions = 576 next-token prompts; identical old interface; no new omitted-sentence calibration',
                'strong_free':'12 predetermined events x 4 forms x 2 roles x 2 question wordings = 192 candidate-free greedy generations, max 12 tokens',
                'strong_list':'same 192 prompts x both candidate orders = 384 next-token contrast prompts; otherwise identical strong interface'},
        scoring={'contrast':'logsumexp of bare and leading-space single-token actor names minus equivalent patient names; fail if names are not single tokens',
                 'r':'(actor-question actor-minus-patient contrast - patient-question actor-minus-patient contrast)/2; r<0 is question-to-argument contrast reversal, r=0 tie',
                 'joint':'both actor and patient correct for same event/form/wording, plus all-form joint; both wrong reported separately',
                 'free':'strip whitespace only; exact full name correctness and strict candidate compliance; invalid outputs wrong; first-token answer recorded separately',
                 'list':'candidate contrast correctness is distinct from unconstrained generation; first-token compliance reported',
                 'heuristics':'always first/last sentence participant; always grammatical agent-verb subject (patient in passive), its complement; always candidate first/second; compare errors and predictions, do not fit after results'},
        inference={'backend':'MLX-LM 0.32.0, MLX 0.32.4; community 4bit weights; verify actual per-model quantization config',
                   'decoding':'greedy, no stochastic sampling, no fine-tuning, sequential models',
                   'resource_bound':'M2 16 GiB; estimated 5.43 GB weights total; no paid inference; stop on resource failures; no prompt search'},
        analysis={'unit':'event paired across forms/questions/models; uncertainty resample SIX name-pair clusters with both directions and all observations retained',
                  'bootstrap_replicates':10000,'bootstrap_seed':20261015,
                  'primary':'strong candidate-free lexical-question object-cleft vs active joint accuracy, and negative question-contrast rates; compare matched 7B vs 1.5B',
                  'secondary':'explicit wording, legacy replication, candidate order, construction tables, individual gains/losses; no confirmatory p-values',
                  'limits':'synthetic previously examined development items; only six participant-pair clusters, intervals conditional on these templates; quantization and model differences limit pure scaling conclusions; no mechanistic claims'},
        new_review_gate='NEW_REVIEW.csv is a separate new constructed review set, NOT human-validated or authentic corpus evidence. No inference on it until independent review is recorded.',
        old_hashes={str(p.relative_to(ROOT.parent)):sha(p) for p in [OLD/'PROTOCOL.json', OLD/'untouched_qwen05.json', OLD/'untouched_qwen15.json', ROOT.parent/'FRONTIER_DIAGNOSTICS_CLOSED_2026-10-15.json']})
    (ROOT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    # Separate questions and empty annotation fields, so reviewers are not shown suggested labels or predictions.
    fresh=[('Nora','Leo','pushed'),('Maya','Owen','followed'),('Elena','Hugo','kicked'),
           ('Iris','Felix','photographed'),('Lena','Oscar','hugged'),('Clara','Milo','tapped')]
    gold=[]
    with (ROOT/'NEW_REVIEW.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['event_id','form','sentence','actor_span','patient_span','equivalent_to_active_yes_no','ambiguous_or_unnatural','reviewer','notes'])
        w.writeheader()
        for i,(a,b,v) in enumerate(fresh):
            for d,(actor,patient) in enumerate(((a,b),(b,a))):
                e=dict(id=f'new{i}-d{d}',actor=actor,patient=patient,verb=v)
                gold.append(e)
                for form in old['forms']:
                    w.writerow(dict(event_id=e['id'],form=form,sentence=sentence(e,form)))
    (ROOT/'NEW_PROPOSED_LABELS.json').write_text(json.dumps({'status':'AI proposed; not independent labels; do not show to reviewer before annotation','events':gold},indent=2)+'\n')
    (ROOT/'REVIEW_INSTRUCTIONS.txt').write_text('Independent review needed before inference. These 48 sentences are newly constructed, not authentic corpus examples. Do not inspect proposed labels or model outputs. For each sentence, copy the person intentionally carrying out the stated physical/perceptual action and the person the action is directed at. Check whether all four forms describe the same event. Mark ambiguity or unnatural wording rather than forcing an answer. Record reviewer identity and notes. Review is not completed by this file being generated.\n')

if __name__=='__main__': build()
