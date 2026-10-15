#!/usr/bin/env python3
"""COMPUT-014: independent, source-locked raw-string rescore of Astra adapter panel.
Run: python3 comput014_replay.py /path/to/completed_adapter_diagnostic
No inference, no gold/reference roles in the constructed sentence-only heuristic.
"""
import collections, hashlib, json, re, sys
from pathlib import Path
DIRECT=re.compile(r"^\s*([A-Za-z]+)\.?\s*$")
EXTRACT=re.compile(r"^\s*Actor:\s*([A-Za-z]+);\s*Patient:\s*([A-Za-z]+)\.?\s*$")
ACTIVE=re.compile(r"^([A-Za-z]+) (?:praised|criticized) ([A-Za-z]+)\.$")
PASSIVE=re.compile(r"^([A-Za-z]+) was (?:praised|criticized) by ([A-Za-z]+)\.$")
CLEFT=re.compile(r"^It was ([A-Za-z]+) whom ([A-Za-z]+) (?:praised|criticized)\.$")
WORLDS=("fresh1","fresh2","fresh3")
ARMS=("ordinary","ordered_roles","unordered_pair")
FIELDS=("n","ex_valid","ex_complement","ex_correct","ex_reversed","ex_duplicate",
        "qa_same","qa_different","qa_invalid_pair","qa_both_correct",
        "qa_patient_actor","qa_patient_correct","qa_patient_invalid",
        "qa_reversed","qa_outside","template_correct")
def template(sentence):
    if m:=ACTIVE.fullmatch(sentence):return m.group(1),m.group(2)
    if m:=PASSIVE.fullmatch(sentence):return m.group(2),m.group(1)
    if m:=CLEFT.fullmatch(sentence):return m.group(2),m.group(1)
    return None
def read_checked(root,lock,rel):
    path=root/rel
    if rel not in lock or hashlib.sha256(path.read_bytes()).hexdigest()!=lock[rel]:
        raise ValueError("Source lock failed: "+rel)
    return json.loads(path.read_text(encoding="utf8"))
def classified(record,item):
    def d(k):
        t=record[k];m=DIRECT.fullmatch(t["clean_output"])
        name=m.group(1) if m else None
        if name!=t["parsed"]["name"] or bool(m)!=t["parsed"]["semantic_format_valid"]:
            raise ValueError("Direct parser discrepancy: "+record["id"]+"/"+k)
        return name
    actor=d("direct_actor"); patient=d("direct_patient")
    e=record["extraction"];m=EXTRACT.fullmatch(e["clean_output"])
    ex=(m.group(1),m.group(2)) if m else (None,None)
    p=e["parsed"]
    if ex!=(p["actor"],p["patient"]) or bool(m)!=p["semantic_format_valid"]:
        raise ValueError("Extraction parser discrepancy: "+record["id"])
    names=(item["actor"],item["patient"])
    both=actor is not None and patient is not None
    exok=ex[0] is not None
    return dict(
       n=1,ex_valid=int(exok),
       ex_complement=int(exok and ex[0]!=ex[1] and set(ex)==set(names)),
       ex_correct=int(ex==names),ex_reversed=int(ex==names[::-1]),
       ex_duplicate=int(exok and ex[0]==ex[1]),
       qa_same=int(both and actor==patient),
       qa_different=int(both and actor!=patient),
       qa_invalid_pair=int(not both),
       qa_both_correct=int((actor,patient)==names),
       qa_patient_actor=int(patient==names[0]),
       qa_patient_correct=int(patient==names[1]),
       qa_patient_invalid=int(patient is None),
       qa_reversed=int((actor,patient)==names[::-1]),
       qa_outside=int(both and (actor not in names or patient not in names)),
       template_correct=int(template(item["sentence"])==names))
def replay(folder):
    folder=Path(folder)
    lock=json.loads((folder/"ARTIFACT_SHA256.json").read_text())["files"]
    items=read_checked(folder,lock,"FROZEN_ITEMS.json")
    proto=read_checked(folder,lock,"FROZEN_PROTOCOL.json")
    indexed={r["id"]:r for r in items}
    if len(indexed)!=288 or len(items)!=288:raise ValueError("Item coverage")
    byarm=collections.defaultdict(collections.Counter)
    byform=collections.defaultdict(collections.Counter)
    total=collections.Counter();anomalies=[];files=0
    for world in WORLDS:
        jobs=[(0,"pretrained")]+[(s,a) for s in (830211,830212) for a in ARMS]
        for seed,arm in jobs:
            unit=f"{world}_{seed}_{arm}"
            data=read_checked(folder,lock,"full_outputs/"+unit+".json")
            if (data["world"],data["seed"],data["arm"],len(data["records"]))!=(world,seed,arm,96):
                raise ValueError("Unit metadata mismatch "+unit)
            if data["protocol_sha256"]!=lock["FROZEN_PROTOCOL.json"]:raise ValueError("Protocol mismatch")
            files+=1;seen=set()
            for r in data["records"]:
                if r["id"] in seen or r["id"] not in indexed:raise ValueError("Duplicate/unknown event")
                seen.add(r["id"]);item=indexed[r["id"]]
                if item["world"]!=world:raise ValueError("World mismatch")
                counts=classified(r,item)
                for f in FIELDS:
                    total[f]+=counts[f];byarm[arm][f]+=counts[f];byform[arm+"/"+item["form"]][f]+=counts[f]
                if counts["ex_valid"] and not counts["ex_complement"]:
                    anomalies.append({"unit":unit,"event_id":r["id"],
                                     "output":r["extraction"]["clean_output"]})
    if files!=21 or total["n"]!=2016:raise ValueError("Unexpected coverage")
    return {"status":"Independent archival reanalysis; not model inference",
            "model_units":files,"counts_all":dict(total),
            "by_arm":{k:dict(v) for k,v in sorted(byarm.items())},
            "by_arm_form":{k:dict(v) for k,v in sorted(byform.items())},
            "noncomplementary_valid_outputs":anomalies,
            "limits":["Reference roles only used for scoring",
                      "Closed-grammar router is not a natural-language parser",
                      "Repeated prompts and shared worlds preclude naive independent inference"]}
if __name__=="__main__":
    if len(sys.argv)!=2:raise SystemExit("usage: python3 comput014_replay.py /path/to/completed_adapter_diagnostic")
    print(json.dumps(replay(sys.argv[1]),indent=2))
