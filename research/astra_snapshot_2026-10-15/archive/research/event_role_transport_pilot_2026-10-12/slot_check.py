#!/usr/bin/env python3
"""Post hoc held-out slot diagnostic; no training or sealed data."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run  # noqa: E402


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def cases():
    return [(e,f) for e in run.prev.events(run.PAIRS,run.VERBS) for f in run.TEST_FORMS]


def prefix(tok,event,form,present):
    sentence=run.prev.sentence(event,form) if present else "[omitted]"
    return tok.apply_chat_template([
        {"role":"system","content":"Give the two event roles exactly as Actor: NAME; Patient: NAME"},
        {"role":"user","content":
         f"Sentence: {sentence}\nCandidates: {', '.join(event['pair'])}\nIdentify the event roles."},
    ],tokenize=False,add_generation_prompt=True)+"Actor:"


def prepare():
    target=ROOT/"SLOT_CHECK_PROTOCOL.json"
    if target.exists(): raise FileExistsError(target)
    run.validate()
    cfg={
        "date":"2026-10-12","status":"post_hoc_diagnostic_frozen_before_slot_test_inference; direct_QA_results_seen",
        "question":"Does the same checkpoint prefer the correct ordered actor/patient tuple on held-out lexical events, even where direct role QA fails?",
        "arms":run.ARMS,"starts":run.STARTS,"forms":run.TEST_FORMS,
        "cases":len(cases()),
        "metric":"Teacher-forced correct-versus-reversed tuple log-likelihood under the same slot prompt used in training, plus sentence-omitted name-order control. Positive margin favors the correct tuple. This is output behavior, not proof of an internal slot representation.",
        "limitations":"Post hoc on the already inspected lexical QA bank; prompts match training format; candidate-relative score may exceed open-generation accuracy.",
        "script_sha256":sha(Path(__file__)),"main_protocol_sha256":sha(ROOT/"PROTOCOL.json"),
        "checkpoints":{f"{s}_{a}":sha(ROOT/f"{s}_{a}"/"adapter.pt")
                       for s in run.STARTS for a in run.ARMS},
    }
    target.write_text(json.dumps(cfg,indent=2)+"\n")
    print("Frozen slot check",len(cases()),"event-forms",flush=True)


def validate():
    cfg=json.loads((ROOT/"SLOT_CHECK_PROTOCOL.json").read_text())
    assert cfg["script_sha256"]==sha(Path(__file__))
    assert cfg["main_protocol_sha256"]==sha(ROOT/"PROTOCOL.json")
    for key,h in cfg["checkpoints"].items():
        assert sha(ROOT/key/"adapter.pt")==h
    return cfg


def score_entries(tok,event,form,present):
    p=prefix(tok,event,form,present)
    pids=tok.encode(p,add_special_tokens=False)
    out=[]
    for actor,patient in ((event["actor"],event["patient"]),
                          (event["patient"],event["actor"])):
        full=tok.encode(p+f" {actor}; Patient: {patient}",add_special_tokens=False)
        assert full[:len(pids)]==pids
        out.append((full,[-100]*len(pids)+full[len(pids):]))
    assert len(out[0][0])==len(out[1][0])
    return out


@torch.inference_mode()
def evaluate(start,arm):
    cfg=validate(); target=ROOT/f"SLOT_{start}_{arm}.json"
    if target.exists(): raise FileExistsError(target)
    torch.set_num_threads(2)
    tok,model=run.prev.load(start)
    saved=torch.load(ROOT/f"{start}_{arm}"/"adapter.pt",map_location="cpu",weights_only=True)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert saved.keys()==params.keys()
    for n,p in params.items(): p.data.copy_(saved[n])
    model.eval(); records=[]
    for i,(event,form) in enumerate(cases()):
        for present in (True,False):
            entries=score_entries(tok,event,form,present)
            length=max(len(x[0]) for x in entries)
            ids=torch.tensor([[tok.pad_token_id]*(length-len(x[0]))+x[0] for x in entries])
            labels=torch.tensor([[-100]*(length-len(x[0]))+x[1] for x in entries])
            mask=torch.tensor([[0]*(length-len(x[0]))+[1]*len(x[0]) for x in entries])
            logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
            losses=F.cross_entropy(logits[:,:-1].contiguous().flatten(0,1),
                                   labels[:,1:].contiguous().flatten(),ignore_index=-100,
                                   reduction="none").view(2,-1)
            scores=(-losses.sum(dim=1)).tolist()
            records.append({"event":event["id"],"form":form,"sentence_present":present,
                            "margin":scores[0]-scores[1],
                            "correct_preferred":scores[0]>scores[1]})
        if (i+1)%48==0: print(start,arm,i+1,"/",len(cases()),flush=True)
    target.write_text(json.dumps({"start":start,"arm":arm,
        "protocol_sha256":sha(ROOT/"SLOT_CHECK_PROTOCOL.json"),
        "checkpoint_sha256":cfg["checkpoints"][f"{start}_{arm}"],
        "records":records},indent=2)+"\n")
    print("Saved",target.name,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("prepare","eval"))
    ap.add_argument("--start",choices=run.STARTS);ap.add_argument("--arm",choices=run.ARMS)
    args=ap.parse_args()
    if args.mode=="prepare":prepare()
    else:
        if args.start is None or args.arm is None:ap.error("eval requires start and arm")
        evaluate(args.start,args.arm)
