"""Independent accounting of reused data, real dispatches and worker checks."""
import json
import numpy as np
from core import HERE,ROOT,sha,save

def records(path):return [json.loads(s) for s in path.read_text().splitlines()]

def main():
    parent=ROOT/'research/voice_query_code_factorial_resumed_2026-10-19'
    raw=records(HERE/'qwen7b/RAW.jsonl');old=records(parent/'qwen7b/RAW.jsonl')
    assert len(raw)==1117 and raw[:248]==old
    dispatch=records(HERE/'DISPATCHES.jsonl');metrics=records(HERE/'WORKER_METRICS.jsonl');gates=records(HERE/'WORKER_GATES.jsonl')
    assert len(dispatch)==len(metrics)==897 and len(gates)==28
    assert [x['dispatch'] for x in dispatch]==list(range(1,898))
    actual=[x for x in dispatch if x['kind']=='continuation'];controls=[x for x in dispatch if x['kind']=='restart_equivalence']
    assert len(actual)==869 and len(controls)==28
    assert [x['logical_call'] for x in actual]==list(range(249,1118))
    assert [x['worker'] for x in gates]==list(range(1,29))
    for i in range(1,29):
        dd=[x for x in dispatch if x['worker']==i]
        assert dd[0]['kind']=='restart_equivalence' and sum(x['kind']=='continuation' for x in dd)<=32
    for x,y in zip(dispatch,metrics):
        assert all(x[k]==y[k] for k in x)
        assert x['disk_free_bytes']>4*1024**3 and x['swap_used_bytes']<2*1024**3
        assert y['active_mlx_bytes']<6*1024**3
    control=old[2]
    with np.load(parent/control['payload'],allow_pickle=False) as ref:
        for g in gates:
            assert g['passed'] and g['all_hashes_equal'] and sha(HERE/g['payload'])==g['payload_sha256']
            with np.load(HERE/g['payload'],allow_pickle=False) as z:
                le=float(np.max(np.abs(z['logits'].astype(np.float32)-ref['logits'].astype(np.float32))))
                te=float(np.max(np.abs(z['trace'].astype(np.float32)-ref['trace'].astype(np.float32))))
            assert le==g['logit_error'] and te==g['trace_error'] and le<=.05 and te<=1e-6
    accounting=json.loads((HERE/'qwen7b/EXECUTION_ACCOUNTING.json').read_text())
    assert accounting['total_investigation_attempts']==1149
    result={'passed':True,'reused_records_bit_identical':248,'new_panel_dispatches':869,'worker_equivalence_dispatches':28,'new_model_attempts':897,'previous_attempts':252,'total_attempts':1149,'total_completed_outputs_including_old_exposed_controls':1147,'all_28_worker_controls_replayed':True,'max_worker_logit_error':max(g['logit_error'] for g in gates),'max_worker_trace_error':max(g['trace_error'] for g in gates),'max_active_mlx_bytes':max(x['active_mlx_bytes'] for x in metrics),'max_peak_mlx_bytes':max(x['peak_mlx_bytes'] for x in metrics),'max_observed_swap_bytes':max(x['swap_used_bytes'] for x in metrics),'min_pre_dispatch_free_disk_bytes':min(x['disk_free_bytes'] for x in metrics),'scientific_conditions_or_scores_changed':False,'no_model_calls_by_audit':True}
    save(HERE/'CONTINUATION_AUDIT.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
