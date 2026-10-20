"""Check documentary coverage, source bytes and generated views; no model execution."""
from pathlib import Path
import csv,hashlib,json,re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def main():
    ledger=json.loads((HERE/'EXPERIMENT_LEDGER.json').read_text())['records']
    coverage=json.loads((HERE/'COVERAGE.json').read_text())
    inventory=json.loads((HERE/'SOURCE_INVENTORY.json').read_text())
    manifest=json.loads((HERE/'EVIDENCE_MANIFEST.json').read_text())['files']
    assert len(ledger)==coverage['catalogue_entries']==308
    assert len({r['id'] for r in ledger})==308
    assert len(inventory['entries'])==coverage['covered_parent_keys']==128
    assert {r['coverage_key'] for r in ledger if r['coverage_key']}=={r['key'] for r in inventory['entries']}
    assert all(r['result'] and r['lesson'] and r['limit'] and r['source_paths'] for r in ledger)
    for s in manifest:
        b=(ROOT/s['path']).read_bytes()
        assert hashlib.sha256(b).hexdigest()==s['sha256'],s['path']
        assert len(b)==s['bytes'],s['path']
    md=(HERE/'EXPERIMENT_LEDGER.md').read_text()
    assert len(re.findall(r'^#### ',md,re.M))==308
    assert all(f'<a id="{r["id"]}"></a>' in md for r in ledger)
    with (HERE/'EXPERIMENT_LEDGER.csv').open() as f:csvrows=list(csv.DictReader(f))
    assert [r['id'] for r in csvrows]==[r['id'] for r in ledger]
    assert (HERE/'SEARCHABLE_LEDGER.html').read_text().count('<tr>')==309
    broken=[]
    for name in ['REPORT.md','EXPERIMENT_LEDGER.md']:
        for target in re.findall(r'\]\(([^)]+)\)',(HERE/name).read_text()):
            if target.startswith(('https:','http:')):continue
            if not (HERE/target.split('#')[0]).exists():broken.append([name,target])
    assert not broken,broken
    out=dict(status='PASS',scope='Document consistency and provenance, not independent scientific reproduction.',unique_catalogue_ids=308,source_files_sha256_verified=len(manifest),parent_keys_covered=128,generated_csv_json_markdown_html_agree=True,main_and_ledger_file_links_exist=True,archived_reports=57,new_model_forwards=0,new_training_runs=0)
    (HERE/'VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
