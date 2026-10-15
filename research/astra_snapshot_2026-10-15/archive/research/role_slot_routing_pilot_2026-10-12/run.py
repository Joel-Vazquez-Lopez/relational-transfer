#!/usr/bin/env python3
"""Answer-first versus slot-first training with identical role and answer labels."""
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
sys.path.insert(0,str(ROOT.parent/"role_prompt_equivariance_pilot_2026-10-12"))
sys.path.insert(0,str(ROOT.parent/"role_subject_decoupling_pilot_2026-10-12"))
import pilot  # noqa: E402
import experiment as prev  # noqa: E402

TEST_PAIRS=(("Aaron","Ruby"),("Matt","Kelly"),("Dan","Hope"),
            ("Ryan","Jessica"),("Tyler","Jennifer"),("Grant","Michelle"))
TEST_VERBS=("lifted","tapped","emailed")
FORMS=pilot.FORMS
STARTS=prev.STARTS
ARMS=("answer_first","slot_first")
EPOCHS=2
LR=prev.LR
SEED=prev.SEED
TRAIN_Q_BY_FORM={
    "active":("actor_performed","actor_did","patient_verb"),
    "passive":("actor_performed","actor_did"),
    "subject_cleft":("actor_performed","patient_verb"),
    "object_cleft":("actor_performed","patient_verb"),
}
PRIMARY_CELLS=(("passive","actor_did","patient_verb"),
               ("subject_cleft","actor_did","patient_verb"),
               ("object_cleft","actor_did","patient_verb"))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def train_rows(event):
    return [pilot.question_row(event,form,q,pilot.TRAIN_Q[q])
            for form in FORMS for q in TRAIN_Q_BY_FORM[form]]


def test_rows():
    return [pilot.question_row(e,form,q,text)
            for e in prev.events(TEST_PAIRS,TEST_VERBS)
            for form in FORMS for q,text in pilot.TEST_Q.items()]


def prompt(tok,row,arm):
    if arm=="answer_first":
        system="Give exactly: Answer: NAME; Actor: NAME; Patient: NAME"
        opening="Answer:"
    else:
        system="Give exactly: Actor: NAME; Patient: NAME; Answer: NAME"
        opening="Actor:"
    p=tok.apply_chat_template([
        {"role":"system","content":system},
        {"role":"user","content":f"Sentence: {row['sentence']}\nCandidates: {', '.join(row['pair'])}\nQuestion: {row['question']}"},
    ],tokenize=False,add_generation_prompt=True)
    return p+opening


def suffix(row,arm):
    if arm=="answer_first":
        return f" {row['gold']}; Actor: {row['actor']}; Patient: {row['patient']}"
    return f" {row['actor']}; Patient: {row['patient']}; Answer: {row['gold']}"


def batch(tok,event,arm):
    entries=[]
    for row in train_rows(event):
        p=prompt(tok,row,arm)
        pids=tok.encode(p,add_special_tokens=False)
        full=tok.encode(p+suffix(row,arm),add_special_tokens=False)
        assert full[:len(pids)]==pids
        entries.append((full+[tok.eos_token_id],[-100]*len(pids)+full[len(pids):]+[tok.eos_token_id]))
    assert len(entries)==9
    length=max(len(x[0]) for x in entries)
    ids=[];labels=[];masks=[]
    for tokens,target in entries:
        pad=length-len(tokens)
        ids.append([tok.pad_token_id]*pad+tokens)
        labels.append([-100]*pad+target)
        masks.append([0]*pad+[1]*len(tokens))
    return torch.tensor(ids),torch.tensor(labels),torch.tensor(masks)


def prepare():
    target=ROOT/"PROTOCOL.json"
    if target.exists():raise FileExistsError(target)
    tok,model=prev.load("fresh1")
    prior_names={n for pair in prev.TRAIN_PAIRS+prev.TEST_PAIRS+pilot.TEST_PAIRS for n in pair}
    prior_names|={"Charlie","Dave","Joy","Kyle","Meg","Pat","Ray","Ted","Val","Will","Andy","Carol"}
    test_names=[n for pair in TEST_PAIRS for n in pair]
    assert len(test_names)==len(set(test_names))==12 and not set(test_names)&prior_names
    for n in test_names:
        assert len(tok.encode(n,add_special_tokens=False))==1,n
        assert len(tok.encode(" "+n,add_special_tokens=False))==1,n
    assert not set(TEST_VERBS)&(set(prev.TRAIN_VERBS)|set(prev.TEST_VERBS)|set(pilot.TEST_VERBS)|{"grabbed","called","poked"})
    for arm in ARMS:
        batch(tok,prev.train_events()[0],arm)
    del model
    cfg={
        "date":"2026-10-12","status":"exploratory; frozen before training and any inference on this new lexical bank",
        "hypothesis":"When given the same actor, patient, and answer labels, generating role slots before the answer improves transfer to crossed construction/question combinations versus giving the answer before the slots.",
        "arms":ARMS,"starts":STARTS,"train_question_sets":TRAIN_Q_BY_FORM,
        "test_questions":pilot.TEST_Q,"forms":FORMS,"test_pairs":TEST_PAIRS,"test_verbs":TEST_VERBS,
        "epochs":EPOCHS,"train_events":len(prev.train_events()),"updates":len(prev.train_events())*EPOCHS,
        "rows_per_update":9,"learning_rate":LR,
        "matched":"Same natural-role source adapter within each start, 48 directed train events, 9 sentence/question prompts per event, same role and answer labels, same train item order, 96 updates, optimizer, learning rate, LoRA parameters. The output order and format instruction differ. Compare logged target-token totals.",
        "primary":"On 36 new lexical events, sum event-form cases where both actor_did and patient_verb are correct for passive, subject_cleft, and object_cleft (108 cases). In each of these cells one question/form combination was absent in training.",
        "primary_cells":PRIMARY_CELLS,
        "positive_screen":"Slot-first exceeds answer-first by at least 10/108 crossed-form joint cases in EACH start, loses no more than 4/36 active joint cases, and improves object-cleft joint in each start. Report all cells, novel wording accuracy, tuple accuracy, invalid outputs and target-token totals regardless.",
        "limits":"A positive result would still be format/order-specific, one Qwen0.5B family, two related starts, constructed English. It would require a control for output position, independent natural data, and a second model family before a general training-repair claim. No human-review data accessed.",
        "source_sha256":{s:sha(prev.source_path(s)) for s in STARTS},
        "script_sha256":sha(Path(__file__)),
        "train_sha256":hashlib.sha256(json.dumps(prev.train_events(),sort_keys=True).encode()).hexdigest(),
        "test_sha256":hashlib.sha256(json.dumps(test_rows(),sort_keys=True).encode()).hexdigest(),
    }
    target.write_text(json.dumps(cfg,indent=2)+"\n")
    print("Frozen",len(prev.train_events()),"train events and",len(test_rows()),"test prompts",flush=True)


def validate():
    cfg=json.loads((ROOT/"PROTOCOL.json").read_text())
    assert cfg["script_sha256"]==sha(Path(__file__))
    assert cfg["train_sha256"]==hashlib.sha256(json.dumps(prev.train_events(),sort_keys=True).encode()).hexdigest()
    assert cfg["test_sha256"]==hashlib.sha256(json.dumps(test_rows(),sort_keys=True).encode()).hexdigest()
    for s in STARTS:assert cfg["source_sha256"][s]==sha(prev.source_path(s))
    return cfg


def train(start,arm):
    cfg=validate();target=ROOT/f"{start}_{arm}"
    if target.exists():raise FileExistsError(target)
    torch.set_num_threads(2)
    tok,model=prev.load(start)
    torch.manual_seed(SEED)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=LR)
    events=prev.train_events();order=list(range(len(events)));rng=random.Random(SEED)
    trace=[];target_tokens=0;began=time.perf_counter();model.train()
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        for idx in order:
            ids,labels,mask=batch(tok,events[idx],arm)
            target_tokens+=int((labels!=-100).sum())
            opt.zero_grad(set_to_none=True)
            logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
            losses=F.cross_entropy(logits[:,:-1].contiguous().flatten(0,1),
                                   labels[:,1:].contiguous().flatten(),
                                   ignore_index=-100,reduction="none").view(9,-1)
            counts=(labels[:,1:]!=-100).sum(dim=1)
            loss=(losses.sum(dim=1)/counts).mean()
            if not torch.isfinite(loss):raise FloatingPointError((start,arm,idx))
            loss.backward();opt.step();trace.append(float(loss.detach()))
            if len(trace)%24==0:
                print(start,arm,len(trace),"/96 elapsed",round(time.perf_counter()-began),flush=True)
    target.mkdir()
    torch.save({n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},target/"adapter.pt")
    (target/"TRAINING.json").write_text(json.dumps({
        "start":start,"arm":arm,"protocol_sha256":sha(ROOT/"PROTOCOL.json"),
        "source_sha256":cfg["source_sha256"][start],
        "checkpoint_sha256":sha(target/"adapter.pt"),
        "updates":len(trace),"examples":len(trace)*9,"target_tokens":target_tokens,
        "elapsed_seconds":time.perf_counter()-began,"losses":trace},indent=2)+"\n")
    print("Saved",target.name,"target_tokens",target_tokens,flush=True)


def parse_output(raw,arm):
    if arm=="answer_first":
        m=re.match(r"^\s*([A-Za-z]+)\s*;\s*Actor:\s*([A-Za-z]+)\s*;\s*Patient:\s*([A-Za-z]+)",raw)
        return (m.group(1),m.group(2),m.group(3)) if m else (None,None,None)
    m=re.match(r"^\s*([A-Za-z]+)\s*;\s*Patient:\s*([A-Za-z]+)\s*;\s*Answer:\s*([A-Za-z]+)",raw)
    return (m.group(3),m.group(1),m.group(2)) if m else (None,None,None)


@torch.inference_mode()
def evaluate(start,arm):
    cfg=validate();target=ROOT/f"EVAL_{start}_{arm}.json"
    if target.exists():raise FileExistsError(target)
    torch.set_num_threads(2)
    tok,model=prev.load(start)
    cp=ROOT/f"{start}_{arm}"/"adapter.pt"
    saved=torch.load(cp,map_location="cpu",weights_only=True)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert saved.keys()==params.keys()
    for n,p in params.items():p.data.copy_(saved[n])
    model.eval();rows=test_rows();out=[]
    for off in range(0,len(rows),8):
        block=rows[off:off+8]
        batch_in=tok([prompt(tok,r,arm) for r in block],padding=True,
                     return_tensors="pt",add_special_tokens=False)
        generated=model.generate(**batch_in,max_new_tokens=13,do_sample=False,
                                 pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        suffixes=generated[:,batch_in["input_ids"].shape[1]:]
        for r,tokens in zip(block,suffixes):
            raw=tok.decode(tokens,skip_special_tokens=True)
            answer,actor,patient=parse_output(raw,arm)
            out.append({"event":r["id"],"form":r["form"],"wording":r["wording"],
                        "role":r["role"],"gold":r["gold"],"raw":raw,
                        "answer":answer,"actor":actor,"patient":patient,
                        "valid":answer in r["pair"],"correct":answer==r["gold"],
                        "slot_joint_correct":actor==r["actor"] and patient==r["patient"],
                        "answer_consistent_with_slots":answer==(actor if r["role"]=="actor" else patient)})
        if (off+8)%180==0:print(start,arm,off+8,"/",len(rows),flush=True)
    assert len(out)==720
    target.write_text(json.dumps({"start":start,"arm":arm,
        "protocol_sha256":sha(ROOT/"PROTOCOL.json"),
        "checkpoint_sha256":sha(cp),"records":out},indent=2)+"\n")
    print("Saved",target.name,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("prepare","train","eval"))
    ap.add_argument("--start",choices=STARTS);ap.add_argument("--arm",choices=ARMS)
    args=ap.parse_args()
    if args.mode=="prepare":prepare()
    else:
        if args.start is None or args.arm is None:ap.error("train/eval need start and arm")
        if args.mode=="train":train(args.start,args.arm)
        else:evaluate(args.start,args.arm)
