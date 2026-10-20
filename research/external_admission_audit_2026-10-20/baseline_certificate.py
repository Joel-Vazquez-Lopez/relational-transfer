"""Static AST certificate + authored arithmetic fixtures, no remote code execution.

Checks a suspected baseline correspondence defect in the released Dai notebook.
Fixture values are synthetic, not LLM outputs. No claim of paper invalidation.
"""
import ast
import hashlib
import json
import math
import zipfile
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent


def colmean(matrix):
    return [mean(col) for col in zip(*matrix)]


def colstd(matrix):
    return [math.sqrt(mean((x - mean(col)) ** 2 for x in col)) for col in zip(*matrix)]


def minus(a, b):
    return [x - y for x, y in zip(a, b)]


def fixture(name, baseline, intervened):
    # Mathematical transcription of the inspected baseline correspondence.
    stale = [minus(y, baseline[-1]) for y in intervened]
    paired = [minus(y, b) for y, b in zip(intervened, baseline)]
    return dict(name=name, synthetic=True, baseline=baseline, intervened=intervened, stale_differences=stale, paired_differences=paired, stale_mean=colmean(stale), paired_mean=colmean(paired), predicted_mean_bias=minus(colmean(baseline), baseline[-1]), measured_mean_bias=minus(colmean(stale), colmean(paired)), stale_std=colstd(stale), paired_std=colstd(paired), predicted_labels=[max(range(len(y)), key=lambda i: y[i]) for y in intervened])


def main():
    with zipfile.ZipFile(HERE / 'dai_software.zip') as z:
        b = z.read('model/Binding_ID_subspace_base_activation_patching.ipynb')
    nb = json.loads(b)
    matches = []
    for ci, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'resid_act_patch_pca_ld':
                matches.append((ci, source, node))
    assert len(matches) == 1
    ci, source, fn = matches[0]
    parents = {}
    for parent in ast.walk(fn):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    stores = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == 'logit6_c' and isinstance(n.ctx, ast.Store)]
    assert len(stores) == 1
    ancestry = []
    node = stores[0]
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If):
            ancestry.append(dict(type='If', test=ast.unparse(node.test)))
        if isinstance(node, ast.For):
            ancestry.append(dict(type='For', target=ast.unparse(node.target), iterator=ast.unparse(node.iter)))
    assert ancestry[:3] == [dict(type='If', test='pi == 0'), dict(type='For', target='data', iterator='tqdm(datas[:])'), dict(type='For', target='pi', iterator='range(nb_pi)')]
    diff = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'logit_diff' for t in n.targets)]
    assert len(diff) == 1 and ast.unparse(diff[0].value) == 'logit6 - logit6_c'
    uses = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == 'logit6_c' and isinstance(n.ctx, ast.Load)]
    assert len(uses) == 1
    # Four fixed counterexamples/checks; no empirical outcomes are substituted.
    base = [[8., 0., -1.], [1., 9., 0.], [0., 2., 10.]]
    unchanged = fixture('zero_true_effect', base, base)
    permutation = [2, 1, 0]
    reordered = fixture('same_zero_effect_reordered', [base[i] for i in permutation], [base[i] for i in permutation])
    delta = [1., -2., 3.]
    shifted = [[v + d for v, d in zip(row, delta)] for row in base]
    uniform = fixture('uniform_true_effect', base, shifted)
    twice = fixture('twice_uniform_true_effect', base, [[v + 2*d for v, d in zip(row, delta)] for row in base])
    fixtures = [unchanged, reordered, uniform, twice]
    checks = [dict(name='zero_true_effect_is_zero_with_paired_baseline', passed=unchanged['paired_mean'] == [0., 0., 0.]), dict(name='stale_baseline_reports_spurious_zero_effect', passed=max(abs(x) for x in unchanged['stale_mean']) > 0), dict(name='stale_result_changes_with_example_order', passed=unchanged['stale_mean'] != reordered['stale_mean']), dict(name='paired_result_invariant_to_example_order', passed=unchanged['paired_mean'] == reordered['paired_mean']), dict(name='uniform_effect_recovered', passed=uniform['paired_mean'] == delta), dict(name='uniform_paired_effect_has_zero_std', passed=uniform['paired_std'] == [0., 0., 0.]), dict(name='stale_std_can_be_nonzero_for_constant_effect', passed=max(uniform['stale_std']) > 0), dict(name='between_nonzero_steps_mean_difference_preserved', passed=all(abs(a-b) < 1e-12 for a,b in zip(minus(twice['stale_mean'],uniform['stale_mean']),delta)))]
    for f in fixtures:
        checks.append(dict(name='exact_bias_identity_' + f['name'], passed=all(abs(a-b) < 1e-12 for a,b in zip(f['predicted_mean_bias'], f['measured_mean_bias']))))
    assert all(c['passed'] for c in checks)
    out = dict(scope='Posthoc static source audit and authored arithmetic examples, not an external LLM reanalysis. No downloaded code executed; no external outcome values inspected.', notebook_sha256=hashlib.sha256(b).hexdigest(), archive_sha256=hashlib.sha256((HERE/'dai_software.zip').read_bytes()).hexdigest(), source_cell_zero_based=ci, function=fn.name, source_cell_sha256=hashlib.sha256(source.encode()).hexdigest(), baseline_store_count=len(stores), baseline_load_count=len(uses), baseline_store_ancestry=ancestry, checks=checks, passed=True, fixtures=fixtures, limits=['Does not recover actual empirical distortion without per-example baseline/intervention logits.', 'Does not establish that this released notebook produced the final published figures.', 'Label proportions use argmax of intervened logits and do not depend on this subtraction.', 'Mean differences between two nonzero intervention steps cancel the constant reference error.', 'Other independent evidence for binding is not tested by this certificate.'])
    (HERE/'BASELINE_CERTIFICATE.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(dict(passed=True, checks=len(checks), cell=ci, zero_true_effect_reported=unchanged['stale_mean'], reordered_zero_true_effect_reported=reordered['stale_mean'], uniform_true_effect=delta, uniform_reported=uniform['stale_mean']),indent=2))

if __name__ == '__main__':
    main()
