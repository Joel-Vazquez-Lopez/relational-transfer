"""Describe the fixed panel after collection; no new model calls or fitted rules."""
from pathlib import Path
import csv
import json
import statistics

HERE = Path(__file__).resolve().parent

def main():
    r = json.loads((HERE / 'RESULTS.json').read_text())
    d = json.loads((HERE / 'qwen7b/DECISION.json').read_text())
    assert d['status'] == 'COMPLETE_FIXED_PANEL' and d['all_controls_passed']
    assert json.loads((HERE / 'INDEPENDENT_AUDIT.json').read_text())['passed']
    assert json.loads((HERE / 'PACKAGED_REPLAY_AUDIT.json').read_text())['passed']
    raw = [json.loads(s) for s in (HERE / 'qwen7b/RAW.jsonl').read_text().splitlines()]
    scientific = [x for x in raw if x['kind'] == 'scientific']
    groups = r['groups']
    native = [x for x in groups if x['mode'] == 'baseline' and x['panel'] == 'factorial']
    candidate = r['prediction']['behavior']['candidate']
    route = r['prediction']['routing']['candidate']
    decision = 'Replicated operational pattern' if r['held_behavior_replicated'] or r['held_routing_replicated'] else 'No prespecified pattern prospectively replicated'
    text = [
        '# Sentence voice × question voice × answer code: completed fixed experiment',
        '', '## Decision', '', f'**{decision}. No distinct event-selection or role-binding mechanism is established.**', '',
        'This is one existing local 4-bit Qwen 7B checkpoint and six AI-authored lexical families. Three families were development cases and three were held out. The conclusions concern the frozen operational patterns; they are not population significance or novelty claims.', '',
        f'All **{r["forwards"]} planned calls** completed: 816 native scientific prompts, 288 fixed cut-14 interventions and 13 validation calls. 248 of these logical outputs are reused byte-for-byte from the preserved partial run. This continuation uses 869 new panel forwards plus 28 worker-equivalence controls. Including 252 previous attempts, the combined cap is **1,149 attempts**; prior failed and incomplete outputs remain preserved. The checkpointed continuation and bounded resource controls were frozen before any new forward; see FREEZE_RECEIPT.json for the commit.', '',
        'The original 128/192 numerical gate remains failed (.0859375 > .05). The subsequent interrupted first fresh validation remains recorded with no payload. This execution uses the already frozen uniform 128-token design and the explicitly recorded restart exception. No failed tolerance was raised and no scientific source, prompt, site, score or hypothesis was changed.', '',
        '## Prospective decisions', '',
        f'- Development-selected behavioral candidate: `{candidate}`. Held-out qualifying families: {r["held_behavior_count"]}/3; replicated: {r["held_behavior_replicated"]}.',
        f'- Development-selected routing candidate: `{route}`. Held-out qualifying families: {r["held_routing_count"]}/3; replicated: {r["held_routing_replicated"]}.',
        '- `None` means no development candidate; a held-only pattern cannot become a prospective success.', '',
        '|Family|Split|Old-prompt positive gap, nats|New-task symbol compliance|Qualifying behavioral pattern|Clean routing control|Qualifying routing pattern|',
        '|---|---|---:|---:|---|---|---|',
    ]
    routes = {p['family']: p for p in r['routing']}
    for p in r['profiles']:
        rt = routes[p['family']]
        text.append(f'|{p["family"]}|{"development" if p["family"] in ("j01","j02","j03") else "held out"}|{p["old_positive_gap"]:.3f}|{p["format_compliance"]:.1%}|{", ".join(k for k,v in p["patterns"].items() if v) or "none"}|{rt["neutral_drift_gate"]}|{", ".join(k for k,v in rt["patterns"].items() if v) or "none"}|')
    text += ['', 'The old-prompt gap is passive minus active mean target-aligned score on entailed cases. All 24 factorial contrasts per family, including the matched-position comparisons, remain in RESULTS.json and the figure. A family must satisfy every frozen contrast and compliance condition to qualify.', '',
             '## Native answers and complete four-answer units', '',
             'These counts use vocabulary-wide first-token argmax, not generated full responses or a calibrated classifier. Each unit crosses role orientation × question direction. Aggregation below pools question voice, mapping and prefix only for readability; NATIVE_CELL_TABLE.csv retains all 32 cells.', '',
             '|Sentence voice|Code lexicon|Correct answers|Complete four-answer units|Symbol-compliant answers|',
             '|---|---|---:|---:|---:|']
    for voice in ('active', 'passive'):
        for lex in ('YN', 'AB'):
            uu = [x for x in native if x['voice'] == voice and x['lexicon'] == lex]
            text.append(f'|{voice}|{lex}|{sum(x["native_correct"] for x in uu)}/{len(uu)*4}|{sum(x["native_complete"] for x in uu)}/{len(uu)}|{sum(x["format_compliant"] for x in uu)}/{len(uu)*4}|')
    text += ['', '### Every family', '', '|Family|Original-prompt answers|Original whole family correct|New-task answers|New-task whole family correct|', '|---|---:|---|---:|---|']
    fams = json.loads((HERE / 'COMPLETE_FAMILY_OUTCOMES.json').read_text())
    for family in sorted({x['family'] for x in fams}):
        old = next(x for x in fams if x['family'] == family and x['panel'] == 'old_reference')
        new = next(x for x in fams if x['family'] == family and x['panel'] == 'factorial')
        text.append(f'|{family}|{old["native_correct"]}/{old["n_answers"]}|{old["all_correct"]}|{new["native_correct"]}/{new["n_answers"]}|{new["all_correct"]}|')
    wrong = [x for x in scientific if not x['native_correct']]
    off = [x for x in scientific if not x['format_compliant']]
    text += ['', f'All **{len(wrong)} incorrect scientific answers**, including **{len(off)} off-format answers**, are retained in UNEXPECTED_ANSWERS.json with expected/actual classes, emitted tokens, top-12 output tokens, scores and payload hashes. ANSWER_LEDGER.csv includes every answer. No competence exclusion changed collection or group means.', '',
             '## Fixed causal panel', '',
             'All interventions use canonical A/B normal mapping, empty prefix and cut 14. Frozen, closed and irrelevant-source masks were run for every canonical prompt regardless of native correctness. Retention is D_mask/D_native only when D_native ≥ .5; D combines all four role × question answers. The clean routing criterion additionally requires all ordinal comparisons positive and irrelevant-mask retention within .25 of 1 in every canonical unit.', '',
             '|Family|Sentence voice|Question voice|Native D|Frozen retention|Closed retention|Irrelevant retention|', '|---|---|---|---:|---:|---:|---:|']
    def fmt(x):
        return 'undefined' if x is None else f'{x:.3f}'
    for rt in r['routing']:
        for c in rt['cells']:
            e = c['effects']
            text.append(f'|{rt["family"]}|{c["voice"]}|{c["question_voice"]}|{c["native"]["D"]:.3f}|{fmt(e["frozen"]["retention"])}|{fmt(e["closed"]["retention"])}|{fmt(e["irrelevant"]["retention"])}|')
    text += ['', 'An irrelevant-mask failure remains a failed interpretability control. The irrelevant mask matches the edge count, not attention mass or all token positions. Closed-mask rescue does not identify role content or establish that the period carries a unique relational message.', '',
             '## Verification and artifacts', '',
             f'- All {r["production_controls"]} production controls passed, including native/wrapped/dense/repeated forward comparisons, sham interventions, causal prefix checks, future-legend invariance and cut-0 disconnections.',
             f'- Independent analysis passed {r["independent_checks"]} checks, replaying full-vector scores, argmax decisions, tensor shapes, hashes, call journals and the pre-evaluation prediction hash.',
             '- Every call retains its full vocabulary output vector losslessly. Native calls retain all 29 residual stages at sentence period, question end and assistant end; masked calls retain five depth anchors and full-trace/prefix hashes.',
             '- PAYLOAD_BUNDLES.json and verified tar bundles preserve the original compressed NPZ bytes. PACKAGED_REPLAY_AUDIT.json verifies every member and score after packaging. The expanded staging copies are removed only after exact member verification.',
             '- INPUTS.json, SOURCE_MANIFEST.json and TOKENIZER_AUDIT.json retain source text, AI-only annotation uncertainties and position matching. Names and predicates are confounded at family level; these are authored templates, not natural documents.', '',
             '## Interpretation and prior work', '',
             'The tested explanations are strict operational predictions, not an exhaustive account of linguistic computation. A failed pattern rules out its specified joint predictions in this panel. It does not establish that relational processing is absent. A successful pattern establishes only the tested construction/question/code dependence. General syntax-aware copying, output preparation and instruction interpretation can remain viable.', '',
             '[Dai, Heinzerling and Inui (2024)](https://aclanthology.org/2024.emnlp-main.967/) study entity–attribute binding, identify a low-rank binding-ID subspace and edit bindings by intervention. This experiment does not estimate that subspace or compare against its mechanism. A voice-dependent output or routing signature is not a new binding mechanism.', '',
             '[Zhao et al. (2021)](https://proceedings.mlr.press/v139/zhao21c.html) show that prompt and answer biases can change language-model classification performance. Here, answer mapping and syntax are explicitly crossed; a sensitivity result is not, by itself, novel calibration research.', '',
             '[Zhang and Nanda (2024)](https://arxiv.org/abs/2309.16042) show that patching metrics and corruption choices can change interpretability conclusions. The fixed controls and preserved failures here address that concern without establishing unique causal localization.', '',
             'No cross-model replication, attention/MLP decomposition, general linguistic stage sequence, entropy-based internal variable or population p value is inferred from this run. Earlier Llama failed gates remain failed. The saved residual traces support later analysis only after a separate design review.', '',
             '**Collection stops after this panel. No adaptive inference, new site search or additional model is included.**', '']
    (HERE / 'REPORT.md').write_text('\n'.join(text))
    with (HERE / 'NATIVE_CELL_TABLE.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(r['native_tables'][0]))
        w.writeheader(); w.writerows(r['native_tables'])
    print(json.dumps({'decision':decision,'behavior_candidate':candidate,'behavior_replicated':r['held_behavior_replicated'],'routing_candidate':route,'routing_replicated':r['held_routing_replicated'],'unexpected_scientific_answers':len(wrong)}))

if __name__ == '__main__':
    main()
