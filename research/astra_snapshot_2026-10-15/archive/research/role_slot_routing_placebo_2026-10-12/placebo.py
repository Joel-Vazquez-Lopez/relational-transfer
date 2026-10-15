#!/usr/bin/env python3
"""Post hoc candidate-list-before-answer control for the slot-first pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"role_slot_routing_pilot_2026-10-12"))
import run as base  # noqa: E402

STARTS=base.STARTS
EPOCHS=base.EPOCHS
SEED=base.SEED
LR=base.LR


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt(tok,row):
    return tok.apply_chat_template([
        {"role":"system","content":"Give exactly: First: NAME; Second: NAME; Answer: NAME"},
        {"role":"user","content":
         f"Sentence: {row['sentence']}\nCandidates: {', '.join(row['pair'])}\nQuestion: {row['question']}"},
    ],tokenize=False,add_generation_prompt=True)+"First:"


def suffix(row):
    return f" {row['pair'][0]}; Second: {row['pair'][1]}; Answer: {row['gold']}"


def batch(tok,event):
    entries=[]
    for row in base.train_rows(event):
        p=prompt(tok,row);prefix=tok.encode(p,add_special_tokens=False)
        full=tok.encode(p+suffix(row),add_special_tokens=False)
        assert full[:len(prefix)]==prefix
        entries.append((full+[tok.eos_token_id],[-100]*len(prefix)+full[len(prefix):]+[tok.eos_token_id]))
    assert len(entries)==9
    length=max(len(tokens) for tokens,_ in entries)
    ids=[];labels=[];mask=[]
    for tokens,target in entries:
        pad=length-len(tokens)
        ids.append([tok.pad_token_id]*pad+tokens)
        labels.append([-100]*pad+target)
        mask.append([0]*pad+[1]*len(tokens))
    return torch.tensor(ids),torch.tensor(labels),torch.tensor(mask)


def prepare():
    target=ROOT/"PROTOCOL.json"
    if target.exists():raise FileExistsError(target)
    base.validate()
    tok,model=base.prev.load("fresh1")
    batch(tok,base.prev.train_events()[0]);del model
    cfg={
        "date":"2026-10-12","status":"post_hoc_placebo; slot-first and answer-first results on test bank already seen",
        "question":"Does listing both candidate names before the answer explain the slot-first advantage without actor/patient labels?",
        "method":"Same source adapters, training events, question-by-form schedule, 96 updates, optimizer and test items as the frozen slot-routing pilot. Output First: candidate0; Second: candidate1; Answer: gold. Candidate order is fixed by the offered pair and does not encode roles.",
        "primary":"The original pilot's 108 crossed construction/question event-form groups where actor_did and patient_verb are both correct.",
        "interpretation":"If candidate-first approaches slot-first, generic answer position/name priming may explain much of the gain. If it resembles answer-first, role-specific intermediate content remains a plausible explanation, but this post hoc control is not independent confirmation.",
        "base_protocol_sha256":sha(base.ROOT/"PROTOCOL.json"),
        "script_sha256":sha(Path(__file__)),
        "source_sha256":{s:sha(base.prev.source_path(s)) for s in STARTS},
        "train_sha256":hashlib.sha256(json.dumps(base.prev.train_events(),sort_keys=True).encode()).hexdigest(),
        "test_sha256":hashlib.sha256(json.dumps(base.test_rows(),sort_keys=True).encode()).hexdigest(),
    }
    target.write_text(json.dumps(cfg,indent=2)+"\n")
    print("Frozen post hoc candidate-first control",flush=True)


def validate():
    cfg=json.loads((ROOT/"PROTOCOL.json").read_text())
    assert cfg["script_sha256"]==sha(Path(__file__))
    assert cfg["base_protocol_sha256"]==sha(base.ROOT/"PROTOCOL.json")
    assert cfg["train_sha256"]==hashlib.sha256(json.dumps(base.prev.train_events(),sort_keys=True).encode()).hexdigest()
    assert cfg["test_sha256"]==hashlib.sha256(json.dumps(base.test_rows(),sort_keys=True).encode()).hexdigest()
    for s in STARTS:assert cfg["source_sha256"][s]==sha(base.prev.source_path(s))
    return cfg


def train(start):
    cfg=validate();target=ROOT/start
    if target.exists():raise FileExistsError(target)
    torch.set_num_threads(2)
    tok,model=base.prev.load(start)
    torch.manual_seed(SEED)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=LR)
    events=base.prev.train_events();order=list(range(len(events)));rng=random.Random(SEED)
    trace=[];target_tokens=0;began=time.perf_counter();model.train()
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        for idx in order:
            ids,labels,mask=batch(tok,events[idx]);target_tokens+=int((labels!=-100).sum())
            opt.zero_grad(set_to_none=True)
            logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
            losses=F.cross_entropy(logits[:,:-1].contiguous().flatten(0,1),
                                   labels[:,1:].contiguous().flatten(),ignore_index=-100,
                                   reduction="none").view(9,-1)
            counts=(labels[:,1:]!=-100).sum(dim=1)
            loss=(losses.sum(dim=1)/counts).mean()
            if not torch.isfinite(loss):raise FloatingPointError((start,idx))
            loss.backward();opt.step();trace.append(float(loss.detach()))
            if len(trace)%24==0:print(start,len(trace),"/96 elapsed",round(time.perf_counter()-began),flush=True)
    target.mkdir()
    torch.save({n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},target/"adapter.pt")
    (target/"TRAINING.json").write_text(json.dumps({
        "start":start,"protocol_sha256":sha(ROOT/"PROTOCOL.json"),
        "source_sha256":cfg["source_sha256"][start],
        "checkpoint_sha256":sha(target/"adapter.pt"),
        "updates":len(trace),"examples":len(trace)*9,"target_tokens":target_tokens,
        "elapsed_seconds":time.perf_counter()-began,"losses":trace},indent=2)+"\n")
    print("Saved",start,"target_tokens",target_tokens,flush=True)


@torch.inference_mode()
def evaluate(start):
    validate();target=ROOT/f"EVAL_{start}.json"
    if target.exists():raise FileExistsError(target)
    torch.set_num_threads(2)
    tok,model=base.prev.load(start)
    cp=ROOT/start/"adapter.pt"
    saved=torch.load(cp,map_location="cpu",weights_only=True)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert saved.keys()==params.keys()
    for n,p in params.items():p.data.copy_(saved[n])
    model.eval();rows=base.test_rows();out=[]
    for off in range(0,len(rows),8):
        block=rows[off:off+8]
        inputs=tok([prompt(tok,r) for r in block],padding=True,return_tensors="pt",add_special_tokens=False)
        generated=model.generate(**inputs,max_new_tokens=13,do_sample=False,
                                 pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        suffixes=generated[:,inputs["input_ids"].shape[1]:]
        for r,tokens in zip(block,suffixes):
            raw=tok.decode(tokens,skip_special_tokens=True)
            m=re.match(r"^\s*([A-Za-z]+)\s*;\s*Second:\s*([A-Za-z]+)\s*;\s*Answer:\s*([A-Za-z]+)",raw)
            first,second,answer=(m.group(1),m.group(2),m.group(3)) if m else (None,None,None)
            out.append({"event":r["id"],"form":r["form"],"wording":r["wording"],
                        "role":r["role"],"gold":r["gold"],"raw":raw,
                        "first":first,"second":second,"answer":answer,
                        "correct":answer==r["gold"],"valid":answer in r["pair"],
                        "candidate_list_correct":first==r["pair"][0] and second==r["pair"][1]})
        if (off+8)%180==0:print(start,off+8,"/",len(rows),flush=True)
    assert len(out)==720
    target.write_text(json.dumps({"start":start,"protocol_sha256":sha(ROOT/"PROTOCOL.json"),
                                   "checkpoint_sha256":sha(cp),"records":out},indent=2)+"\n")
    print("Saved",target.name,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("prepare","train","eval"))
    ap.add_argument("--start",choices=STARTS);args=ap.parse_args()
    if args.mode=="prepare":prepare()
    elif args.start is None:ap.error("train/eval need start")
    elif args.mode=="train":train(args.start)
    else:evaluate(args.start)
