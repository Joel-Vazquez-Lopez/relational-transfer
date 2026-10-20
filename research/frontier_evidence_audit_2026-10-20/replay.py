"""Independent scalar-record replay of completed studies; zero model forwards.

Does not replace the original full-vector/tensor audits. Uses stdlib only.
No new hypothesis tests: the p value replayed here was frozen in the source study.
"""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT / 'research/competence_selection_confirmation_2026-10-19'
MODELS = ['qwen7b', 'llama3b']
checks = []


def check(name, ok):
    checks.append({'name': name, 'passed': bool(ok)})
    if not ok:
        raise AssertionError(name)


def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2) + '\n')


records = []
for model in MODELS:
    rows = [json.loads(s) for s in (SOURCE / model / 'RAW.jsonl').read_text().splitlines()]
    check(model + '_781_actual_records', len(rows) == 781)
    records.extend(rows)
groups = defaultdict(list)
for row in records:
    score = row['class_logits']['Yes'] - row['class_logits']['No']
    check('score_' + str(row['call']), abs(score - row['yes_minus_no']) < 1e-10)
    if row['kind'] == 'scientific':
        groups[row['model'], row['family'], row['voice'], row['mode']].append(row)


def contrast(rows):
    check('four_cells', {(r['role'], r['query']) for r in rows} == {(0, 0), (0, 1), (1, 0), (1, 1)} and len(rows) == 4)
    return sum((-1) ** (r['role'] + r['query']) * (r['class_logits']['Yes'] - r['class_logits']['No']) for r in rows) / 4


units = []
for key in sorted(k for k in groups if k[-1] == 'baseline'):
    model, family, voice, _ = key
    rows = groups[key]
    native = [r['actual'] == r['expected'] for r in rows]
    calibrated = [('Yes' if r['yes_minus_no'] > r['threshold'] else 'No' if r['yes_minus_no'] < r['threshold'] else None) == r['expected'] for r in rows]
    check('native_definition_' + str(key), native == [r['correct'] for r in rows])
    check('calibrated_definition_' + str(key), calibrated == [r['calibrated_correct'] for r in rows])
    d = contrast(rows)
    df = contrast(groups[model, family, voice, 'frozen'])
    units.append(dict(model=model, family=family, voice=voice, native_correct=sum(native), calibrated_correct=sum(calibrated), native_complete=all(native), calibrated_complete=all(calibrated), D_native=d, D_frozen=df, retention=df / d if abs(d) >= .5 else None))
check('96_complete_units', len(units) == 96)
old = json.loads((SOURCE / 'RESULTS.json').read_text())
old_lookup = {(u['model'], u['family'], u['voice']): u for u in old['units']}
for u in units:
    ref = old_lookup[u['model'], u['family'], u['voice']]
    check('unit_replay_' + str((u['model'], u['family'], u['voice'])), all(u[k] == ref[k] for k in ['native_correct', 'calibrated_correct', 'native_complete', 'calibrated_complete']) and abs(u['D_native'] - ref['D_native']) < 1e-12 and abs(u['D_frozen'] - ref['interventions']['frozen']['D']) < 1e-12)
totals = []
for model in MODELS:
    for voice in ['active', 'passive']:
        us = [u for u in units if u['model'] == model and u['voice'] == voice]
        totals.append(dict(model=model, voice=voice, n_answers=96, n_units=24, **{k: sum(u[k] for u in us) for k in ['native_correct', 'calibrated_correct', 'native_complete', 'calibrated_complete']}))
families = sorted({u['family'] for u in units})
values = []
for family in families:
    a, p = [next(u for u in units if (u['model'], u['family'], u['voice']) == ('qwen7b', family, v)) for v in ['active', 'passive']]
    values.append(int(a['calibrated_complete']) - int(a['native_complete']) - int(p['calibrated_complete']) + int(p['native_complete']))
positive, negative = sum(x > 0 for x in values), sum(x < 0 for x in values)
n = positive + negative
pvalue = sum(math.comb(n, j) for j in range(positive, n + 1)) / 2 ** n
check('sole_existing_confirmatory_test_replay', (positive, negative, len(values) - n, pvalue) == (old['primary']['positive'], old['primary']['negative'], old['primary']['ties'], old['primary']['one_sided_p']))
profiles = []
for gate in ['all', 'native_complete', 'calibrated_complete']:
    for policy in ['own', 'common']:
        selected = {m: {(u['family'], u['voice']) for u in units if u['model'] == m and (gate == 'all' or u[gate])} for m in MODELS}
        common = selected[MODELS[0]] & selected[MODELS[1]]
        stats = {}
        for m in MODELS:
            us = [u for u in units if u['model'] == m and (u['family'], u['voice']) in (selected[m] if policy == 'own' else common)]
            valid = [u for u in us if u['retention'] is not None]
            stats[m] = dict(n_selected=len(us), n_valid=len(valid), mean_retention=mean(u['retention'] for u in valid))
        gap = stats['llama3b']['mean_retention'] - stats['qwen7b']['mean_retention']
        ref = next(x for x in old['profiles'] if (x['mode'], x['gate'], x['policy']) == ('frozen', gate, policy + '_selected'))
        check('profile_replay_' + gate + policy, abs(gap - ref['gap']) < 1e-12)
        profiles.append(dict(gate=gate, policy=policy, models=stats, llama_minus_qwen=gap))

# Exact design aliasing, not a statement about an LLM's native algorithm.
truth_table = [dict(role=r, query=q, truth=int(r == q)) for r in [0, 1] for q in [0, 1]]
check('binary_role_recoverable_from_question_and_truth', all(x['role'] == (x['query'] if x['truth'] else 1 - x['query']) for x in truth_table))
legend = json.loads((ROOT / 'research/legend_order_diagnostic_2026-10-20/RESULTS.json').read_text())
check('legend_report_counts', legend['counts']['negative_category']['matches'] == 32 and legend['counts']['last_option']['matches'] == 16 and legend['native_correct'] == 16 and not legend['causal_task_ready'])
save('REPLAY.json', dict(scope='Scalar RAW.jsonl replay; original tensor audits remain separate. No new model inference, fitting, thresholds, layers or significance tests.', passed=all(c['passed'] for c in checks), n_checks=len(checks), checks=checks, actual_source_records=len(records), totals=totals, sole_existing_test=dict(positive=positive, negative=negative, ties=len(values)-n, one_sided_p=pvalue), profiles=profiles, design_aliasing_truth_table=truth_table))
with (HERE / 'RECOMPUTED_UNITS.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(units[0])); w.writeheader(); w.writerows(units)
paths = set((ROOT / 'research').glob('*/REPORT.md'))
paths.discard(HERE / 'REPORT.md')
paths.update([ROOT / 'ACL_HANDOFF_2026-10-15_REV2/EXPERIMENT_INVENTORY.md', ROOT / 'COMPASS_RESEARCH_FOUNDATION_2026-10-15/EXPERIMENT_INDEX.md'])
paths.update(SOURCE / name for name in ['RESULTS.json', 'DESCRIPTIVE_ROBUSTNESS.json', 'SELECTION_DECOMPOSITION.json', 'LITERATURE_SCOPE.md', 'ENDPOINT_AUDIT.json', 'PROTOCOL.md', 'LOCK.json'])
paths.update(SOURCE / m / 'RAW.jsonl' for m in MODELS)
paths.add(ROOT / 'research/legend_order_diagnostic_2026-10-20/RESULTS.json')
save('SOURCE_MANIFEST.json', dict(scope='Recent report inventory and explicitly used evidence. Older 57 reports reviewed through existing handoff inventory; not independently reproduced here. Report count is not an independent-experiment count.', recent_report_files=len(list((ROOT / 'research').glob('*/REPORT.md'))) - int((HERE / 'REPORT.md').exists()), files=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths)]))
print(json.dumps(dict(passed=True, checks=len(checks), totals=totals, primary_replayed=pvalue, profiles=[{k:v for k,v in p.items() if k!='models'} for p in profiles]), indent=2))
