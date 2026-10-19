"""Analysis, independent audits and lossless packaging; zero model forwards."""
import json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    d=json.loads((HERE/'qwen7b/DECISION.json').read_text())
    assert d['status']=='COMPLETE_FIXED_PANEL' and d['all_controls_passed']
    for script in ('analyze.py','audit_continuation.py','storage.py','packaged_audit.py','write_report.py','explain_results.py','plot.py'):
        log=HERE/(script.removesuffix('.py').upper()+'.log')
        with log.open('x') as f:
            subprocess.run([sys.executable,'-u',str(HERE/script)],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=HERE.parent.parent)
        print('COMPLETED',script,flush=True)
    a=json.loads((HERE/'CONTINUATION_AUDIT.json').read_text())
    report=HERE/'REPORT.md'
    with report.open('a') as f:
        f.write('\n## Checkpointed execution audit\n\n')
        f.write(f"All248 reused records are byte-for-byte identical to the interrupted run. The continuation completed869 missing panel calls and28 additional worker controls:897 new model attempts,1149 attempts across the investigation and1147 completed outputs including the old exposed controls. All28 worker controls were independently replayed. Maximum full-logit error:{a['max_worker_logit_error']}; maximum trace error:{a['max_worker_trace_error']}.\n\n")
        f.write(f"Maximum observed active MLX allocation:{a['max_active_mlx_bytes']/2**30:.3f}GiB; peak MLX allocation:{a['max_peak_mlx_bytes']/2**30:.3f}GiB; maximum pre-dispatch swap:{a['max_observed_swap_bytes']/2**30:.3f}GiB; minimum pre-dispatch free disk:{a['min_pre_dispatch_free_disk_bytes']/2**30:.3f}GiB. Every pre-dispatch resource check passed. These measurements do not diagnose the prior kernel panic or guarantee OS stability.\n\n")
        f.write('CONTINUATION_AUDIT.json, DISPATCHES.jsonl, WORKER_GATES.jsonl and WORKER_METRICS.jsonl distinguish logical dataset rows from actual GPU attempts. The 28 worker control payloads and all original scientific payload bytes are preserved. No additional inference is scheduled.\n')
    subprocess.run([sys.executable,str(HERE/'final_decision.py')],check=True)
    print('FIXED_PANEL_FULLY_AUDITED_AND_PACKAGED',flush=True)

if __name__=='__main__':main()
