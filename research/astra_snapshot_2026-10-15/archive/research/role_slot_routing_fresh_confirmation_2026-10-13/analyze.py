#!/usr/bin/env python3
"""Audit frozen fresh-bank three-arm slot-routing confirmation."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
STARTS=("fresh1","fresh3")
ARMS=("answer_first","candidate_first","slot_first")
FORMS=("active","passive","subject_cleft","object_cleft")
Q=("actor_performed","actor_did","patient_verb","actor_carried","patient_which")
cfg=json.loads((ROOT/"PROTOCOL.json").read_text())
protocol_hash=hashlib.sha256((ROOT/"PROTOCOL.json").read_bytes()).hexdigest()
out={"protocol_sha256":protocol_hash,"arms":{},"screen":{}}
for start in STARTS:
    for arm in ARMS:
        key=f"{start}_{arm}"
        data=json.loads((ROOT/f"EVAL_{key}.json").read_text())
        assert data["protocol_sha256"]==protocol_hash
        assert data["checkpoint_sha256"]==cfg["checkpoint_sha256"][key]
        records=data["records"]
        assert len(records)==960
        by={(r["event"],r["form"],r["wording"]):r for r in records}
        assert len(by)==960
        events=sorted({r["event"] for r in records})
        assert len(events)==48
        forms={};families={}
        for form in FORMS:
            forms[form]={
                "crossed_joint":sum(by[(e,form,"actor_did")]["correct"] and
                                     by[(e,form,"patient_verb")]["correct"] for e in events),
                "novel_joint":sum(by[(e,form,"actor_carried")]["correct"] and
                                   by[(e,form,"patient_which")]["correct"] for e in events),
                "all_five":sum(all(by[(e,form,q)]["correct"] for q in Q) for e in events),
            }
        for pi in range(6):
            subset=[e for e in events if e.startswith(f"p{pi}-")]
            assert len(subset)==8
            families[str(pi)]=sum(by[(e,f,"actor_did")]["correct"] and
                                    by[(e,f,"patient_verb")]["correct"]
                                    for e in subset for f in FORMS[1:])
        result={
            "forms":forms,
            "families_crossed":families,
            "primary_crossed_joint":sum(forms[f]["crossed_joint"] for f in FORMS[1:]),
            "novel_joint":sum(forms[f]["novel_joint"] for f in FORMS),
            "all_five":sum(forms[f]["all_five"] for f in FORMS),
            "valid_answers":sum(r["valid"] for r in records),
        }
        if arm!="candidate_first":
            result["slot_joint_outputs"]=sum(r["slot_joint_correct"] for r in records)
            result["primary_routed_from_slots"]=sum(
                by[(e,f,"actor_did")]["actor"]==by[(e,f,"actor_did")]["gold"] and
                by[(e,f,"patient_verb")]["patient"]==by[(e,f,"patient_verb")]["gold"]
                for e in events for f in FORMS[1:])
        out["arms"][key]=result
for start in STARTS:
    treatment=out["arms"][f"{start}_slot_first"]
    diffs={}
    for arm in ("answer_first","candidate_first"):
        control=out["arms"][f"{start}_{arm}"]
        diffs[arm]={
            "primary_gain":treatment["primary_crossed_joint"]-control["primary_crossed_joint"],
            "active_gain":treatment["forms"]["active"]["crossed_joint"]-control["forms"]["active"]["crossed_joint"],
            "object_cleft_gain":treatment["forms"]["object_cleft"]["crossed_joint"]-control["forms"]["object_cleft"]["crossed_joint"],
            "family_gains":{p:treatment["families_crossed"][p]-control["families_crossed"][p] for p in treatment["families_crossed"]},
        }
    out["screen"][start]={"comparisons":diffs,"pass":all(
        d["primary_gain"]>=12 and d["active_gain"]>=-5 and d["object_cleft_gain"]>=6
        for d in diffs.values())}
out["positive_confirmation"]=all(out["screen"][s]["pass"] for s in STARTS)
(ROOT/"ANALYSIS.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
