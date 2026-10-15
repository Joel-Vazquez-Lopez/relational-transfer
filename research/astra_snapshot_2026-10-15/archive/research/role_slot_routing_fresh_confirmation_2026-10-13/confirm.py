#!/usr/bin/env python3
"""Prospective lexical confirmation of three previously trained role-answer formats."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"role_slot_routing_pilot_2026-10-12"))
sys.path.insert(0,str(ROOT.parent/"role_slot_routing_placebo_2026-10-12"))
import run as base  # noqa: E402
import placebo  # noqa: E402

TEST_PAIRS=(("Robin","Jerry"),("Taylor","Jordan"),("Kim","Jean"),
            ("Jess","Kay"),("Lou","Jim"),("Sean","Scott"))
TEST_VERBS=("nudged","phoned","teased","scolded")
ARMS=("answer_first","candidate_first","slot_first")
STARTS=base.STARTS
FORMS=base.FORMS
QUESTIONS=base.pilot.TEST_Q


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(start,arm):
    if arm=="candidate_first":return placebo.ROOT/start/"adapter.pt"
    return base.ROOT/f"{start}_{arm}"/"adapter.pt"


def test_rows():
    return [base.pilot.question_row(e,f,q,t)
            for e in base.prev.events(TEST_PAIRS,TEST_VERBS)
            for f in FORMS for q,t in QUESTIONS.items()]


def prompt(tok,row,arm):
    return placebo.prompt(tok,row) if arm=="candidate_first" else base.prompt(tok,row,arm)


def parse(raw,arm):
    if arm!="candidate_first":return base.parse_output(raw,arm)
    m=re.match(r"^\s*([A-Za-z]+)\s*;\s*Second:\s*([A-Za-z]+)\s*;\s*Answer:\s*([A-Za-z]+)",raw)
    return (m.group(3),None,None) if m else (None,None,None)


def prepare():
    target=ROOT/"PROTOCOL.json"
    if target.exists():raise FileExistsError(target)
    base.validate();placebo.validate()
    tok,model=base.prev.load("fresh1")
    names=[n for pair in TEST_PAIRS for n in pair]
    old_names={n for pairs in (base.prev.TRAIN_PAIRS,base.prev.TEST_PAIRS,base.pilot.TEST_PAIRS,
                              base.TEST_PAIRS,placebo.base.TEST_PAIRS) for pair in pairs for n in pair}
    old_names|={"Charlie","Dave","Joy","Kyle","Meg","Pat","Ray","Ted","Val","Will","Andy","Carol",
                "Harry","Susan","Alan","Ada","Gary","Albert","Louis","Mark","Ian","Carl","Tony","Tim"}
    assert len(names)==len(set(names))==12 and not set(names)&old_names
    for n in names:
        assert len(tok.encode(n,add_special_tokens=False))==1,n
        assert len(tok.encode(" "+n,add_special_tokens=False))==1,n
    old_verbs=set(base.prev.TRAIN_VERBS)|set(base.prev.TEST_VERBS)|set(base.pilot.TEST_VERBS)|set(base.TEST_VERBS)
    old_verbs|={"grabbed","called","poked","challenged","stopped","encouraged"}
    assert not set(TEST_VERBS)&old_verbs
    del model
    rows=test_rows();assert len(rows)==960
    cfg={
        "date":"2026-10-13","status":"frozen before any inference on this new lexical bank; no additional training",
        "question":"Does the slot-first answer gain over both answer-first and candidate-first replicate on independently chosen names and verbs?",
        "starts":STARTS,"arms":ARMS,"test_pairs":TEST_PAIRS,"test_verbs":TEST_VERBS,
        "forms":FORMS,"questions":QUESTIONS,
        "test_cases":"48 directed events × 4 forms × 5 questions = 960 outputs per checkpoint",
        "primary":"Across passive, subject_cleft and object_cleft, count event-form cases where actor_did and patient_verb are both correct; 48×3=144 cases/checkpoint. One question/form pairing was absent in training for each of these forms.",
        "positive_screen":"Slot-first improves primary by at least +12/144 over BOTH answer-first and candidate-first in EACH start; active same-pair joint regresses by no more than 5/48 versus either; object-cleft primary improves by at least +6/48 over each in both starts. Report all cells and invalids regardless.",
        "secondary":"Novel actor_carried+patient_which both correct across all four forms /192; per-form and per-pair results; source-output slot routing audit. Same 13-token greedy decoding used in pilot.",
        "limits":"Independent lexical test only, not training replication, model-family replication, or natural-language confirmation. Six pair families share one generator; the placebo was designed post hoc on the earlier bank but all three arms are fixed before this bank.",
        "script_sha256":sha(Path(__file__)),
        "base_protocol_sha256":sha(base.ROOT/"PROTOCOL.json"),
        "placebo_protocol_sha256":sha(placebo.ROOT/"PROTOCOL.json"),
        "checkpoint_sha256":{f"{s}_{a}":sha(checkpoint(s,a)) for s in STARTS for a in ARMS},
        "test_rows_sha256":hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
    }
    target.write_text(json.dumps(cfg,indent=2)+"\n")
    print("Frozen",len(rows),"new prompts/checkpoint",flush=True)


def validate():
    cfg=json.loads((ROOT/"PROTOCOL.json").read_text())
    assert cfg["script_sha256"]==sha(Path(__file__))
    assert cfg["base_protocol_sha256"]==sha(base.ROOT/"PROTOCOL.json")
    assert cfg["placebo_protocol_sha256"]==sha(placebo.ROOT/"PROTOCOL.json")
    assert cfg["test_rows_sha256"]==hashlib.sha256(json.dumps(test_rows(),sort_keys=True).encode()).hexdigest()
    for key,h in cfg["checkpoint_sha256"].items():
        s,a=key.split("_",1)
        assert sha(checkpoint(s,a))==h
    return cfg


@torch.inference_mode()
def evaluate(start,arm):
    cfg=validate();target=ROOT/f"EVAL_{start}_{arm}.json"
    if target.exists():raise FileExistsError(target)
    torch.set_num_threads(2)
    tok,model=base.prev.load(start)
    saved=torch.load(checkpoint(start,arm),map_location="cpu",weights_only=True)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert saved.keys()==params.keys()
    for n,p in params.items():p.data.copy_(saved[n])
    model.eval();rows=test_rows();out=[]
    for off in range(0,len(rows),8):
        block=rows[off:off+8]
        inputs=tok([prompt(tok,r,arm) for r in block],padding=True,
                   return_tensors="pt",add_special_tokens=False)
        gen=model.generate(**inputs,max_new_tokens=13,do_sample=False,
                           pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        suffixes=gen[:,inputs["input_ids"].shape[1]:]
        for r,tokens in zip(block,suffixes):
            raw=tok.decode(tokens,skip_special_tokens=True)
            answer,actor,patient=parse(raw,arm)
            out.append({"event":r["id"],"pair_index":r["pair_index"],
                        "form":r["form"],"wording":r["wording"],"role":r["role"],
                        "gold":r["gold"],"raw":raw,"answer":answer,
                        "actor":actor,"patient":patient,
                        "valid":answer in r["pair"],"correct":answer==r["gold"],
                        "slot_joint_correct":actor==r["actor"] and patient==r["patient"] if arm!="candidate_first" else None})
        if (off+8)%240==0:print(start,arm,off+8,"/",len(rows),flush=True)
    assert len(out)==960
    target.write_text(json.dumps({"start":start,"arm":arm,
        "protocol_sha256":sha(ROOT/"PROTOCOL.json"),
        "checkpoint_sha256":cfg["checkpoint_sha256"][f"{start}_{arm}"],
        "records":out},indent=2)+"\n")
    print("Saved",target.name,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("prepare","eval"))
    ap.add_argument("--start",choices=STARTS);ap.add_argument("--arm",choices=ARMS)
    args=ap.parse_args()
    if args.mode=="prepare":prepare()
    else:
        if args.start is None or args.arm is None:ap.error("eval requires start and arm")
        evaluate(args.start,args.arm)
