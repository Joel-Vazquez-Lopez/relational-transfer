"""Transcription of the external observer executed through a command cell.

Retained after execution for provenance, not part of the ex ante frozen code.
Never run this to resume an experiment; its only action is to interrupt the
specific own controller when the frozen prerequisite is already impossible.
"""
from pathlib import Path
import json, os, signal, subprocess, time

p=Path('research/matched_demand_handoff_2026-10-20')
pattern=r'^research/stronger_model_development_2026-10-15/\.venv/bin/python -u research/matched_demand_handoff_2026-10-20/run\.py$'
for _ in range(180):
    r=[json.loads(l) for l in (p/'NATIVE.jsonl').read_text().splitlines()]
    failures=sum(not x['first_token_correct'] for x in r)
    if 144-failures<.95*144:
        assert not (p/'HYBRIDS.jsonl').exists()
        record=dict(reason='Frozen 95% native first-token prerequisite mathematically unattainable',native_completed_at_stop_check=len(r),first_token_failures_at_stop_check=failures,maximum_possible_correct_out_of_144=144-failures,required_correct_at_least=137,source_code_changed=False,interventions_started=False,stop_rule='Enforce the frozen competence prerequisite before uninformative interventions. Early stop is a documented scheduling deviation from completing all 144 native prompts, not a changed hypothesis, threshold or result.',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        (p/'SCIENTIFIC_STOP.json').write_text(json.dumps(record,indent=2)+'\n')
        ids=subprocess.check_output(['pgrep','-f',pattern],text=True).split()
        assert len(ids)==1,ids
        os.kill(int(ids[0]),signal.SIGINT)
        print(json.dumps(record));break
    time.sleep(1)
else:raise AssertionError('No competence failure reached within bounded observer window')
