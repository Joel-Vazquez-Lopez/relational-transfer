#!/usr/bin/env python3
"""Audit the post hoc candidate-first control against the frozen original arms."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/"role_slot_routing_pilot_2026-10-12"
FORMS=("active","passive","subject_cleft","object_cleft")
STARTS=("fresh1","fresh3")
protocol_hash=hashlib.sha256((ROOT/"PROTOCOL.json").read_bytes()).hexdigest()
base_hash=hashlib.sha256((BASE/"PROTOCOL.json").read_bytes()).hexdigest()
cfg=json.loads((ROOT/"PROTOCOL.json").read_text())
assert cfg["base_protocol_sha256"]==base_hash
out={"protocol_sha256":protocol_hash,"base_protocol_sha256":base_hash,"starts":{}}
for start in STARTS:
    training=json.loads((ROOT/start/"TRAINING.json").read_text())
    ev=json.loads((ROOT/f"EVAL_{start}.json").read_text())
    assert training["protocol_sha256"]==ev["protocol_sha256"]==protocol_hash
    assert training["checkpoint_sha256"]==ev["checkpoint_sha256"]
    assert training["updates"]==96 and training["examples"]==864
    assert training["target_tokens"]==8640
    records=ev["records"]
    assert len(records)==720
    by={(r["event"],r["form"],r["wording"]):r for r in records}
    assert len(by)==720
    events=sorted({r["event"] for r in records})
    assert len(events)==36
    form_result={}
    for form in FORMS:
        form_result[form]={
            "crossed_joint":sum(by[(e,form,"actor_did")]["correct"] and
                                 by[(e,form,"patient_verb")]["correct"] for e in events),
            "novel_joint":sum(by[(e,form,"actor_carried")]["correct"] and
                               by[(e,form,"patient_which")]["correct"] for e in events),
        }
    original={}
    for arm in ("answer_first","slot_first"):
        old=json.loads((BASE/f"EVAL_{start}_{arm}.json").read_text())
        old_by={(r["event"],r["form"],r["wording"]):r for r in old["records"]}
        assert len(old_by)==720
        direct=route=0
        for event in events:
            for form in FORMS[1:]:
                a=old_by[(event,form,"actor_did")]
                p=old_by[(event,form,"patient_verb")]
                direct+=a["correct"] and p["correct"]
                route+=(a["actor"]==a["gold"] and p["patient"]==p["gold"])
        original[arm]={"crossed_direct":direct,"crossed_routed_from_generated_slots":route}
    out["starts"][start]={
        "forms":form_result,
        "crossed_joint":sum(form_result[f]["crossed_joint"] for f in FORMS[1:]),
        "novel_joint":sum(form_result[f]["novel_joint"] for f in FORMS),
        "valid_answers":sum(r["valid"] for r in records),
        "candidate_list_correct":sum(r["candidate_list_correct"] for r in records),
        "target_tokens":training["target_tokens"],
        "original_arms":original,
    }
(ROOT/"ANALYSIS.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
