"""Finish the current one-shot workflow; no retries, no additional panels."""
import json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
python=ROOT/'research/stronger_model_development_2026-10-15/.venv/bin/python'
def main():
 deadline=time.monotonic()+3600
 while not (HERE/'qwen7b/DECISION.json').exists():
  if (HERE/'qwen7b/ERROR.json').exists():raise RuntimeError('Qwen stopped; preserve error, no downstream inference')
  if time.monotonic()>deadline:raise RuntimeError('Qwen wait exceeded fixed one-hour bound; no new inference')
  time.sleep(2)
 decision=json.loads((HERE/'qwen7b/DECISION.json').read_text());assert decision['status']=='COMPLETE_FIXED_PANEL' and decision['all_controls_passed']
 with (HERE/'RUN_llama3b.log').open('x') as f:subprocess.run([str(python),'-u',str(HERE/'run.py'),'llama3b'],check=True,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
 for script,log in [('analyze.py','ANALYSIS.log'),('decompose_selection.py','DECOMPOSITION.log'),('plot.py','PLOT.log'),('archive_payloads.py','ARCHIVE.log')]:
  with (HERE/log).open('x') as f:subprocess.run([str(python),'-u',str(HERE/script)],check=True,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
 print('Both bounded panels, frozen analysis, descriptive decomposition, figure and reconstructable raw archives complete',flush=True)
if __name__=='__main__':main()
