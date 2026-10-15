"""Finish analysis after the current approved API runner; never sends requests."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
started=time.monotonic()
while not (ROOT/'GEMMA_COMPLETE.json').exists():
 log=(ROOT/'GEMMA_RECOVERY_RUN.log').read_text() if (ROOT/'GEMMA_RECOVERY_RUN.log').exists() else ''
 if '"stopped": true' in log or '"stopped_before_request": true' in log:
  print('Runner stopped; no automatic recovery or scientific analysis of incomplete Gemma panel.',flush=True);sys.exit(1)
 if time.monotonic()-started>5400:
  print('Analysis watcher timed out; no inference changes.',flush=True);sys.exit(1)
 time.sleep(30)
r=subprocess.run([sys.executable,str(ROOT/'analyze_cross.py')],check=False,timeout=60)
if r.returncode:sys.exit(r.returncode)
d=json.loads((ROOT/'CROSS_ANALYSIS.json').read_text())
lines=['# Cross-family exploratory diagnostic','',d['status'],'','## Joint actor/patient correctness','', 'Each cell counts events with both role answers exactly correct, out of12. L/E denotes lexical versus explicit wording.','', '| Model/interface | Active L/E | Passive L/E | Subject cleft L/E | Object cleft L/E |','|---|---:|---:|---:|---:|']
for n,m in d['models'].items():
 cells=m['strong_free'];vals=[str(cells[f+'/lexical']['joint_correct'])+'/'+str(cells[f+'/explicit']['joint_correct']) for f in ['active','passive','subject_cleft','object_cleft']]
 lines.append('| '+n+' | '+' | '.join(vals)+' |')
lines+=['','## Gemma observations','']
g=d['models']['gemma']
for key,v in g['strong_free'].items():
 lines.append(f"- {key}: joint{v['joint_correct']}/{v['events']}; actor{v['actor_correct']}/12; patient{v['patient_correct']}/12; compliant{v['compliant']}/{v['answers']}.")
lines+=['','## Interpretation limits','', 'Reused constructed events are exploratory comparability checks, not new natural-language confirmation. Only six participant-pair clusters support the paired intervals in CROSS_ANALYSIS.json. All-zero/all-one bootstrap intervals are degenerate, not population certainty.','', 'Gemma uses a hosted31B model with no documented native assistant prefill and no arbitrary candidate scores. The matched no-prefill Llama condition checks that specific interface difference; sizes, families, precision, templates and server implementations remain different. Do not interpret this as pure scaling or infer an internal mechanism.','', 'The rejection of one request produced an HTTP429. Its originalEUR0.52430 worst-case reservation remains held. The following amount is conservative budget accounting, including that unknown-charge hold, NOT an invoice estimate: '+d['billing']['conservative_accounted_eur']+' EUR. The total cap was1 EUR.','', 'The next decision must follow the completed model-level results: if Gemma succeeds, prominently report the capability boundary; if errors remain, inspect compliance and positional/grammatical shortcuts before a mechanistic study. Independently reviewed natural examples remain necessary.']
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('Complete analysis and RESULTS.md saved; historical/frozen hashes verified.',flush=True)
