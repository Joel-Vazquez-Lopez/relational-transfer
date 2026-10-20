"""Verify local audit provenance without network, inference or remote execution."""
import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    checks = []

    def check(name, ok):
        checks.append(dict(name=name, passed=bool(ok)))
        if not ok:
            raise AssertionError(name)

    lock = json.loads((HERE/'AVAILABILITY_LOCK.json').read_text())
    for name, h in lock['files'].items():
        check('availability_lock_' + name, hashlib.sha256((HERE/name).read_bytes()).hexdigest() == h)
    inventory = json.loads((HERE/'REPOSITORY_INVENTORY.json').read_text())
    check('six_pinned_repositories', len(inventory) == 6 and len({x['repo'] for x in inventory}) == 6)
    by_repo = {x['repo']: x for x in inventory}
    for rec in inventory:
        p = HERE/(rec['repo'].replace('/', '__')+'.json')
        check('metadata_hash_' + rec['repo'], hashlib.sha256(p.read_bytes()).hexdigest() == rec['metadata_sha256'])
        d = json.loads(p.read_text())
        check('metadata_commit_' + rec['repo'], d['head'] == rec['head'])
    fetched = json.loads((HERE/'FETCH_MANIFEST.json').read_text())
    total = 0
    for x in fetched['files']:
        b = (HERE/x['local']).read_bytes()
        check('source_sha256_' + x['local'], hashlib.sha256(b).hexdigest() == x['sha256'])
        check('source_git_blob_' + x['local'], hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest() == x['git_blob'])
        check('source_commit_' + x['local'], x['commit'] == by_repo[x['repo']]['head'])
        check('source_size_' + x['local'], len(b) == x['bytes'] and len(b) <= 20*1024**2)
        total += len(b)
    check('source_total', total == fetched['total_bytes'])
    supplements = json.loads((HERE/'SUPPLEMENT_FETCH.json').read_text())
    for x in supplements:
        if x['status'] == 'fetched':
            b = (HERE/x['name']).read_bytes()
            check('supplement_' + x['name'], len(b) == x['bytes'] and hashlib.sha256(b).hexdigest() == x['sha256'])
            total += len(b)
    check('bounded_downloads', total < 200*1024**2)
    check('figshare_unresolved_not_data_absence', next(x for x in supplements if x['name']=='figshare_share.html')['bytes'] == 0)
    with zipfile.ZipFile(HERE/'dai_software.zip') as z:
        members = z.infolist()
        check('six_dai_input_datasets', sum(x.filename.endswith('.jsonl') for x in members) == 6)
        check('no_path_traversal_in_zip_manifest', all(not Path(x.filename).is_absolute() and '..' not in Path(x.filename).parts for x in members))
        notebook_bytes = z.read('model/Binding_ID_subspace_base_activation_patching.ipynb')
    cert = json.loads((HERE/'BASELINE_CERTIFICATE.json').read_text())
    check('certificate_source_hash', hashlib.sha256(notebook_bytes).hexdigest() == cert['notebook_sha256'])
    check('all_12_arithmetic_checks', cert['passed'] and len(cert['checks']) == 12 and all(x['passed'] for x in cert['checks']))
    check('all_fixtures_labelled_synthetic', all(x['synthetic'] for x in cert['fixtures']))
    for name in ['technion-cs-nlp/llm-arithmetic-heuristics/llm-arithmetic-analysis-main.ipynb', 'technion-cs-nlp/llm-arithmetic-heuristics/pythia-heuristics-analysis-notebook.ipynb', 'HumanCompatibleAI/leela-interp/notebooks/act_patching.ipynb', 'HumanCompatibleAI/leela-interp/notebooks/other_figures.ipynb', 'kmeng01/rome/notebooks/average_causal_effects.ipynb']:
        nb = json.loads((HERE/'sources'/name).read_text())
        check('no_retained_outputs_' + name, not any(c.get('outputs') for c in nb['cells']))
    ledger = json.loads((HERE/'AVAILABILITY_LEDGER.json').read_text())
    check('no_eligible_external_study', len(ledger['candidates']) == 6 and not any(x['eligible'] for x in ledger['candidates']) and ledger['eligible_studies'] == 0)
    check('no_external_empirical_claim', all(ledger[k] == 0 for k in ['external_model_forwards', 'external_empirical_reanalyses', 'new_significance_tests']))
    out = dict(passed=all(c['passed'] for c in checks), n_checks=len(checks), selected_source_bytes=fetched['total_bytes'], selected_sources_plus_supplements_bytes=total, checks=checks, scope='Static provenance and arithmetic verification only; does not verify original model results or exhaustive global data availability.')
    (HERE/'VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='checks'},indent=2))

if __name__ == '__main__':
    main()
