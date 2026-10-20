"""Inventory documentary evidence only. No model imports, inference, or training."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
ARCHIVE=ROOT/'ACL_HANDOFF_2026-10-15_REV2/evidence/history'
EXCLUDE={'complete_experiment_audit_2026-10-20'}
STEMS=('REPORT','RESULT','SUMMARY','DECISION','STOP','ERROR','README','INVENTORY','INDEX','ANALYSIS','AUDIT','GATE','DESIGN','PROPOSAL','VALIDATION')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 entries={};sourcefiles={}
 def add(key,path,kind):
  if not path.is_file():return
  rel=str(path.relative_to(ROOT));e=entries.setdefault(key,dict(key=key,paths=[],kinds=[]))
  if rel not in e['paths']:e['paths'].append(rel)
  if kind not in e['kinds']:e['kinds'].append(kind)
  sourcefiles[rel]=dict(path=rel,bytes=path.stat().st_size,sha256=sha(path))
 index=json.loads((ROOT/'research/paper_audit_2026-10-14/REPORT_INDEX.json').read_text())
 for p in sorted(ARCHIVE.rglob('*.md')):
  if p.name in ['PHASE1_REPORT.md','PHASE1_STRICT_REPORT.md'] and str(p.relative_to(ARCHIVE)) not in {r['path'] for r in index['reports']}:
   index['reports'].append(dict(path=str(p.relative_to(ARCHIVE)),sha256=sha(p)))
 for r in index['reports']:
  p=ARCHIVE/r['path'];assert p.is_file() and sha(p)==r['sha256'],str(p)
  add(str(Path(r['path']).parent),p,'historical_report')
 for folder in sorted((ROOT/'research').iterdir()):
  if not folder.is_dir() or folder.name in EXCLUDE:continue
  key=str(folder.relative_to(ROOT));entries.setdefault(key,dict(key=key,paths=[],kinds=['current_directory']))
  for p in sorted(folder.glob('*')):
   if p.is_file() and p.suffix.lower() in ['.md','.json'] and (p.suffix.lower()=='.md' or any(t in p.name.upper() for t in STEMS) or p.name in ['COLLECTION.json','COMPUTE_ACCOUNTING.json','METADATA_CORRECTION.json','PROTOCOL.md']):
    if p.stat().st_size<3*1024**2:add(key,p,'current_record')
  for sub in ['qwen7b','llama3b','results']:
   for p in sorted((folder/sub).glob('*')):
    if p.is_file() and p.suffix.lower() in ['.md','.json'] and any(t in p.name.upper() for t in STEMS) and p.stat().st_size<2*1024**2:add(key,p,'model_or_results_record')
 foundation=ROOT/'COMPASS_RESEARCH_FOUNDATION_2026-10-15/source'
 for key in ['d5_original','d5_indirect_support','phase3']:
  folder=foundation/key
  if folder.exists():
   entries.setdefault(key,dict(key=key,paths=[],kinds=['early_foundation_group']))
   docs=[p for p in folder.rglob('*.md') if p.is_file()]
   for p in docs:add(key,p,'foundation_record')
 # Some current folders have protocols/results with unusual names.
 for e in entries.values():
  p=ROOT/e['key']
  if p.is_dir() and not e['paths']:
   for f in sorted(p.glob('*.json')):
    if f.stat().st_size<2*1024**2 and f.name not in ['tokenizer.json','vocab.json']:add(e['key'],f,'fallback_record')
 out=dict(scope='All 57 hash-verified historical report records, current research directories excluding this audit, and early foundation groups. Directory/report counts are not experiment counts.',historical_report_count=len(index['reports']),current_research_directories=len([p for p in (ROOT/'research').iterdir() if p.is_dir() and p.name not in EXCLUDE]),entries=list(entries.values()),source_files=list(sourcefiles.values()))
 (HERE/'SOURCE_INVENTORY.json').write_text(json.dumps(out,indent=2)+'\n')
 # Complete report text is kept only as a review aid; the manifest hashes canonical originals.
 docs=[]
 for e in entries.values():
  pp=[ROOT/p for p in e['paths'] if p.endswith('.md')]
  preferred=[p for p in pp if re.search(r'(REPORT|RESULTS|GATE_REPORT|SCIENTIFIC_REPORT|PHASE1).*\.md$',p.name)]
  if not preferred:preferred=pp[:2]
  unique=[];seen=set()
  for p in preferred:
   h=sha(p)
   if h not in seen:unique.append(p);seen.add(h)
  preferred=unique
  text='\n\n'.join('SOURCE: '+str(p.relative_to(ROOT))+'\n'+p.read_text(errors='replace') for p in preferred)
  docs.append(dict(key=e['key'],text=text,chars=len(text)))
 (HERE/'REVIEW_TEXT.json').write_text(json.dumps(docs,indent=2)+'\n')
 print(json.dumps(dict(entries=len(entries),source_files=len(sourcefiles),historical_reports=len(index['reports']),review_characters=sum(x['chars'] for x in docs)),indent=2))
if __name__=='__main__':main()
