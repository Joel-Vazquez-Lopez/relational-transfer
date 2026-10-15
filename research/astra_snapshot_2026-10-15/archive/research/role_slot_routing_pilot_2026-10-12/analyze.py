#!/usr/bin/env python3
"""Audit and summarize frozen answer-first versus slot-first pilot."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
STARTS=("fresh1","fresh3")
ARMS=("answer_first","slot_first")
FORMS=("active","passive","subject_cleft","object_cleft")
QUESTIONS=("actor_performed","actor_did","patient_verb","actor_carried","patient_which")
protocol_hash=hashlib.sha256((ROOT/"PROTOCOL.json").read_bytes()).hexdigest()
summary={}
for start in STARTS:
    for arm in ARMS:
        name=f"{start}_{arm}"
        train=json.loads((ROOT/name/"TRAINING.json").read_text())
        ev=json.loads((ROOT/f"EVAL_{name}.json").read_text())
        assert train["protocol_sha256"]==ev["protocol_sha256"]==protocol_hash
        assert train["checkpoint_sha256"]==ev["checkpoint_sha256"]
        assert train["updates"]==96 and train["examples"]==864
        rows=ev["records"]
        assert len(rows)==720
        by={(r["event"],r["form"],r["wording"]):r for r in rows}
        assert len(by)==720
        events=sorted({r["event"] for r in rows})
        assert len(events)==36
        form_summary={}
        for form in FORMS:
            groups=[[by[(e,form,q)] for q in QUESTIONS] for e in events]
            form_summary[form]={
                "crossed_joint":sum(g[1]["correct"] and g[2]["correct"] for g in groups),
                "novel_joint":sum(g[3]["correct"] and g[4]["correct"] for g in groups),
                "all_five":sum(all(r["correct"] for r in g) for g in groups),
                "actor_did":sum(g[1]["correct"] for g in groups),
                "patient_verb":sum(g[2]["correct"] for g in groups),
                "actor_carried":sum(g[3]["correct"] for g in groups),
                "patient_which":sum(g[4]["correct"] for g in groups),
            }
        summary[name]={
            "forms":form_summary,
            "primary_crossed_joint":sum(form_summary[f]["crossed_joint"] for f in FORMS if f!="active"),
            "novel_joint":sum(v["novel_joint"] for v in form_summary.values()),
            "all_five":sum(v["all_five"] for v in form_summary.values()),
            "valid_answers":sum(r["valid"] for r in rows),
            "slot_joint_predictions":sum(r["slot_joint_correct"] for r in rows),
            "answer_consistent_with_slots":sum(r["answer_consistent_with_slots"] for r in rows),
            "target_tokens":train["target_tokens"],
            "elapsed_seconds":train["elapsed_seconds"],
        }
screen={}
for start in STARTS:
    base=summary[start+"_answer_first"]
    treatment=summary[start+"_slot_first"]
    gain=treatment["primary_crossed_joint"]-base["primary_crossed_joint"]
    active=treatment["forms"]["active"]["crossed_joint"]-base["forms"]["active"]["crossed_joint"]
    object_gain=treatment["forms"]["object_cleft"]["crossed_joint"]-base["forms"]["object_cleft"]["crossed_joint"]
    screen[start]={"crossed_joint_gain":gain,"active_joint_gain":active,
                   "object_cleft_joint_gain":object_gain,
                   "pass":gain>=10 and active>=-4 and object_gain>0}
out={"protocol_sha256":protocol_hash,"summary":summary,"screen":screen,
     "positive_pilot":all(v["pass"] for v in screen.values())}
(ROOT/"ANALYSIS.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
