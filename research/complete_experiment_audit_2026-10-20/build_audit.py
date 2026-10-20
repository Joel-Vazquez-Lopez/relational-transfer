"""Build a documentary ledger, not a new scientific experiment. Python stdlib only."""
from pathlib import Path
import collections,csv,hashlib,html,json,re
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FOUNDATION=Path('COMPASS_RESEARCH_FOUNDATION_2026-10-15/source/phase3/post_p3_2_sol_discovery')
def read(name):
    with (HERE/name).open() as f:
        rows=list(csv.DictReader(f,delimiter='\t'))
    assert all(None not in r and all(v is not None for v in r.values()) for r in rows),name
    return rows
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    inv=json.loads((HERE/'SOURCE_INVENTORY.json').read_text())
    entries=inv['entries'];records=[]
    def add(id,title,lineage,kind,status,execution,result,lesson,limit,sources,coverage=None):
        sources=list(dict.fromkeys(sources));assert sources,(id,'no source')
        assert all((ROOT/p).is_file() for p in sources),(id,sources)
        records.append(dict(id=id,title=title,lineage=lineage,kind=kind,status=status,execution=execution,result=result,lesson=lesson,limit=limit,source_paths=sources,coverage_key=coverage,evidence_grade='Documentary audit of preserved reports/results; no model reproduction in this audit.'))
    for r in read('curated_current.tsv'):
        n=int(r['index']);e=entries[n]
        add(f'S{n:03}',e['key'],e['key'],*[r[k] for k in ['kind','status','execution','result','lesson','limit']],e['paths'],e['key'])
    for r in read('curated_canonical.tsv'):
        src=str((HERE/'source_snapshots'/r['source_suffix']).relative_to(ROOT))
        add(r['id'],r['title'],'Canonical synthetic programme',*[r[k] for k in ['kind','status','execution','result','lesson','limit']],[src])
    plan=json.loads((ROOT/FOUNDATION/'discovery_plan.json').read_text())
    plan_by_id={d['id']:d for d in plan['diagnostics']}
    drows=read('curated_discovery.tsv');assert set(plan_by_id)<=set(r['id'] for r in drows)
    for r in drows:
        n=int(re.search(r'\d+',r['id'])[0]);memo='research_memo.md' if n<=51 else ('d52_d59_results_memo_2026-09-30.md' if n<=61 else 'd62_d65_prediction_memo_2026-09-30.md')
        src=[str(FOUNDATION/memo)]
        p=plan_by_id.get(r['id'])
        if p:
            src.append(str(FOUNDATION/'discovery_plan.json'))
            for name in p.get('planned_outputs',[]):
                f=ROOT/FOUNDATION/name
                if f.is_file() and f.suffix in ['.json','.csv'] and f.stat().st_size<1024**2 and any(t in name for t in ['summary','review','manifest']):src.append(str(f.relative_to(ROOT)))
        add(r['id'],r['title'],'Post-P3.2 adaptive synthetic discovery',r['kind'],r['status'],'Reported diagnostic/review in the post-P3.2 discovery programme; exact shared-model scope in linked source.',r['result'],r['lesson'],'Exploratory/adaptive lineage on permitted synthetic models, not independent confirmation or evidence about pretrained language models. Review, design and plotting entries are not additional experiments.',src)
    for r in read('curated_track_b.tsv'):
        n=int(r['id'][3:]);memo='track_b_notebook_2026-10-01.md' if n<=31 else ('track_b_notebook_2026-10-02.md' if n<=36 else 'track_b_arc2_notebook_2026-10-02.md')
        sources=[str(FOUNDATION/memo)]
        # Some BD32–36 narratives span both days; preserve both when relevant.
        if 32<=n<=36:sources.append(str(FOUNDATION/'track_b_notebook_2026-10-01.md'))
        for folder in (ROOT/FOUNDATION).glob(f'track_b*bd{n}_*'):
            for f in folder.iterdir():
                if f.is_file() and f.suffix=='.json' and 'result' in f.name and f.stat().st_size<1024**2:sources.append(str(f.relative_to(ROOT)))
        add(r['id'],r['title'],'Track B adaptive synthetic discovery',r['kind'],r['status'],'Reported Track-B diagnostic, training comparison or design; precise model/sample scope in linked notebook/result.',r['result'],r['lesson'],'Adaptive synthetic programme. New-seed predictions are bounded within-programme checks; not independent task/architecture confirmation or evidence of LLM linguistic mechanisms.',sources)
    for r in read('curated_children.tsv'):
        n=int(r['parent_index']);e=entries[n]
        matches=[p for p in e['paths'] if Path(p).name==r['source_name']];assert len(matches)==1,(r['id'],matches)
        add(r['id'],r['title'],f'S{n:03} / '+e['key'],*[r[k] for k in ['kind','status','execution','result','lesson','limit']],matches)
    assert len({r['id'] for r in records})==len(records)
    assert {r['coverage_key'] for r in records if r['coverage_key']}=={e['key'] for e in entries}
    assert len(read('curated_current.tsv'))==len(entries)
    bykind=dict(sorted(collections.Counter(r['kind'] for r in records).items()))
    manifest={p['path']:dict(p) for p in inv['source_files']}
    origins=json.loads((HERE/'SOURCE_ORIGINS.json').read_text())
    for s in origins['files']:
        f=ROOT/s['snapshot'];assert digest(f)==s['sha256'];manifest[s['snapshot']]=dict(path=s['snapshot'],bytes=f.stat().st_size,sha256=digest(f))
    for r in records:
        for p in r['source_paths']:
            f=ROOT/p;manifest[p]=dict(path=p,bytes=f.stat().st_size,sha256=digest(f))
    for p,v in manifest.items():assert digest(ROOT/p)==v['sha256'],p
    out=dict(date='2026-10-20',scope='Recoverable preserved research reports, result records, designs and explicitly indexed diagnostics. Entries overlap, share models/data and include nonexperiments. Not an independent experiment count.',records=records)
    (HERE/'EXPERIMENT_LEDGER.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    fields=list(records[0]);
    with (HERE/'EXPERIMENT_LEDGER.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:w.writerow({**r,'source_paths':'; '.join(r['source_paths'])})
    lines=['# Complete documentary experiment ledger','','**Read the [main report](REPORT.md) first.** This catalogue includes group indexes, duplicate-source aliases, designs, invalid runs, engineering, reanalyses and empirical studies. Its length is not an independent experiment count. No model was rerun for this audit.','','Each entry separates execution, result, lesson and limits. A frozen negative is retained even where later diagnostics were more promising.','','## Contents','']
    groups=[('Canonical synthetic programme',lambda r:r['id'].startswith('C-')),('Post-P3.2 discovery D01–D65 and review variants',lambda r:r['id'].startswith('D')),('Track B B-D1–B-D46',lambda r:r['id'].startswith('B-')),('Historical/current directory ledger',lambda r:r['id'].startswith('S')),('Substudies within grouped directories',lambda r:not r['id'].startswith(('C-','D','B-','S')))]
    for title,pred in groups:
        lines+=['### '+title,'']
        for r in records:
            if not pred(r):continue
            lines += [f'<a id="{r["id"]}"></a>',f"#### {r['id']} — {r['title']}",'',f"**Kind:** {r['kind']} · **Outcome:** {r['status']}  ",f"**Lineage:** {r['lineage']}",'',f"- **What ran:** {r['execution']}",f"- **Result:** {r['result']}",f"- **What it tells us:** {r['lesson']}",f"- **Limit:** {r['limit']}",'', '**Sources:** '+', '.join(f'[{Path(p).name}](../../{p})' for p in r['source_paths']), '']
    (HERE/'EXPERIMENT_LEDGER.md').write_text('\n'.join(lines)+'\n')
    coverage=dict(status='PASS_DOCUMENTARY_COVERAGE',catalogue_entries=len(records),entry_kinds=bykind,independent_experiment_count='NOT_IDENTIFIED: nested diagnostics, reruns, shared models/data and historical reporting conventions prevent a valid grand total.',covered_parent_keys=len(entries),current_research_directories=inv['current_research_directories'],archived_historical_reports=inv['historical_report_count'],canonical_snapshot_files=len(origins['files']),discovery_ids=len(drows),track_b_ids=len(read('curated_track_b.tsv')),nested_directory_substudies=len(read('curated_children.tsv')),evidence_source_files=len(manifest),all_source_files_exist_and_hash_match=True,all_parent_keys_have_curated_entries=True,all_56_original_discovery_plan_ids_have_entries=True,model_forward_passes_this_audit=0,training_runs_this_audit=0,paid_api_calls_this_audit=0,protected_checkpoint_or_reserved_example_access=False,raw_model_reproduction=False,scope_limit='This is not an exhaustive search of every private chat, other agent workspace, deleted artifact or unindexed intermediate version. Historical results are reported evidence, not freshly reproduced science.')
    (HERE/'COVERAGE.json').write_text(json.dumps(coverage,indent=2)+'\n')
    (HERE/'EVIDENCE_MANIFEST.json').write_text(json.dumps(dict(files=sorted(manifest.values(),key=lambda v:v['path'])),indent=2)+'\n')
    # Small offline search interface: everything escaped, no network, no hidden API calls.
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['id'],r['kind'],r['status'],r['title'],r['execution'],r['result'],r['lesson'],r['limit']])+'</tr>' for r in records)
    page='''<!doctype html><meta charset="utf-8"><title>Research experiment audit</title><style>body{font:15px system-ui;margin:28px;color:#17212a}input{padding:12px;width:min(90%,680px);font:inherit}table{border-collapse:collapse;width:100%;margin-top:20px}th,td{padding:12px;border:1px solid #ccd2d8;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#edf2f6}td{min-width:160px}td:first-child{min-width:65px}h1{font-size:28px}.wrap{overflow:auto}</style><h1>What every documented investigation told us</h1><p>308 catalogue entries, <b>not 308 independent experiments</b>. Includes plans, invalid runs, shared-data analyses and engineering. No new model inference. Use REPORT.md for conclusions and EXPERIMENT_LEDGER.md/JSON for source links.</p><input id="q" placeholder="Search role, padding, failed, causal, Qwen…" aria-label="Filter research ledger"><p id="n"></p><div class="wrap"><table><thead><tr>'''+''.join('<th>'+h+'</th>' for h in ['ID','Kind','Outcome','Study','What ran','Result','Lesson','Limit'])+'</tr></thead><tbody>'+rows+'''</tbody></table></div><script>const q=document.querySelector('#q'),rr=[...document.querySelectorAll('tbody tr')],n=document.querySelector('#n');function filter(){let k=0;for(const r of rr){const show=r.textContent.toLowerCase().includes(q.value.toLowerCase());r.hidden=!show;k+=show}n.textContent=k+' matching entries'}q.addEventListener('input',filter);filter()</script>'''
    page=page.replace('308 catalogue entries',f'{len(records)} catalogue entries')
    (HERE/'SEARCHABLE_LEDGER.html').write_text(page)
    print(json.dumps(coverage,indent=2))
if __name__=='__main__':main()
