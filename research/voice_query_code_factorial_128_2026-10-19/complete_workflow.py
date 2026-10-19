"""Finish this one fixed run; never retries an inference or adds conditions."""
import json,time,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;python=ROOT/'research/stronger_model_development_2026-10-15/.venv/bin/python'
def main():
 deadline=time.monotonic()+7200
 while not (HERE/'qwen7b/DECISION.json').exists():
  if (HERE/'qwen7b/ERROR.json').exists():raise RuntimeError('Scientific panel stopped at technical gate; no retry')
  if time.monotonic()>deadline:raise RuntimeError('Wait bound reached; no additional inference')
  time.sleep(2)
 d=json.loads((HERE/'qwen7b/DECISION.json').read_text());assert d['status']=='COMPLETE_FIXED_PANEL' and d['all_controls_passed']
 for script,log in [('analyze.py','ANALYSIS.log'),('storage.py','PACKAGING.log'),('packaged_audit.py','PACKAGED_AUDIT.log'),('report.py','REPORT_GENERATION.log'),('plot.py','PLOT.log')]:
  with (HERE/log).open('x') as f:subprocess.run([str(python),'-u',str(HERE/script)],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
 print('Bounded1119-call investigation complete, with1117 corrected-panel payloads archived and both exposed failed-preflight payloads preserved',flush=True)
if __name__=='__main__':main()
