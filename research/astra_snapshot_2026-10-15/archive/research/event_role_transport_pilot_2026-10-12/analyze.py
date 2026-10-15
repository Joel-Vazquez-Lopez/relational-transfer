#!/usr/bin/env python3
"""Audit all frozen predictions and summarize the transport pilot."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ARMS=("slot_sft","transport")
STARTS=("fresh1","fresh3")
FORMS=("active","passive","subject_cleft","object_cleft")
Q=("actor_performed","actor_did","patient_verb","actor_carried","patient_which")
protocol_hash=hashlib.sha256((ROOT/"PROTOCOL.json").read_bytes()).hexdigest()
summary={}
for start in STARTS:
    for arm in ARMS:
        name=f"{start}_{arm}"
        train=json.loads((ROOT/name/"TRAINING.json").read_text())
        data=json.loads((ROOT/f"EVAL_{name}.json").read_text())
        assert train["protocol_sha256"]==data["protocol_sha256"]==protocol_hash
        assert train["checkpoint_sha256"]==data["checkpoint_sha256"]
        assert train["updates"]==48 and train["supervised_rows"]==1152
        records=data["records"]
        assert len(records)==720
        keys=[(x["event"],x["form"],x["wording"]) for x in records]
        assert len(set(keys))==720
        assert all(x["valid"] for x in records)
        by={(r["event"],r["form"],r["wording"]):r for r in records}
        events=sorted({r["event"] for r in records})
        assert len(events)==36
        form_result={}
        for form in FORMS:
            groups=[]
            for event in events:
                rows=[by[(event,form,q)] for q in Q]
                assert len(rows)==5
                groups.append(rows)
            form_result[form]={
                "heldout_joint":sum(g[3]["correct"] and g[4]["correct"] for g in groups),
                "all_five":sum(all(x["correct"] for x in g) for g in groups),
                "actor_carried":sum(g[3]["correct"] for g in groups),
                "patient_which":sum(g[4]["correct"] for g in groups),
                "actor_did":sum(g[1]["correct"] for g in groups),
                "patient_verb":sum(g[2]["correct"] for g in groups),
            }
        summary[name]={
            "forms":form_result,
            "heldout_joint":sum(x["heldout_joint"] for x in form_result.values()),
            "all_five":sum(x["all_five"] for x in form_result.values()),
            "valid_answers":sum(x["valid"] for x in records),
            "training_seconds":train["elapsed_seconds"],
        }
screen={}
for start in STARTS:
    base=summary[f"{start}_slot_sft"]
    treatment=summary[f"{start}_transport"]
    gain=treatment["heldout_joint"]-base["heldout_joint"]
    active=treatment["forms"]["active"]["heldout_joint"]-base["forms"]["active"]["heldout_joint"]
    passive=treatment["forms"]["passive"]["heldout_joint"]-base["forms"]["passive"]["heldout_joint"]
    object_cleft=treatment["forms"]["object_cleft"]["heldout_joint"]-base["forms"]["object_cleft"]["heldout_joint"]
    screen[start]={"joint_gain":gain,"active_gain":active,"passive_gain":passive,
                   "object_cleft_gain":object_cleft,
                   "pass":gain>=8 and active>=-4 and passive>=-4 and object_cleft>0}
out={"protocol_sha256":protocol_hash,"summary":summary,"screen":screen,
     "positive_pilot":all(v["pass"] for v in screen.values())}
(ROOT/"ANALYSIS.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
