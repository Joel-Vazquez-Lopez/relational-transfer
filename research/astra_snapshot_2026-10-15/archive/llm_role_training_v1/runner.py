"""Matched local Qwen role-supervision pilot. See PROTOCOL.json."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import string
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parent.parent
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from llm_role_training_v0.feasibility import add_adapters  # noqa: E402
MANIFEST = json.loads((PROJECT / "role_transfer_phase01" / "role_aux_confirmation_v1"
                       / "GRAPH_MANIFEST.json").read_text())
MODEL_PATH = ("/Users/joelvazquezlopez/.cache/huggingface/hub/"
              "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
              "7ae557604adf67be50417f59c2c2f167def9a775")
NAMES = ("Mira", "Owen", "Leila", "Victor", "Nadia", "Felix",
         "Iris", "Amir", "Clara", "Hugo")
VERBS = ("chased", "helped", "followed")
ARMS = ("pretrained_no_finetuning", "unordered_identities", "ordered_roles")
SEED = 820107


def edges() -> set[tuple[int, int]]:
    result = {tuple(e) for e in MANIFEST["worlds"]["fresh0"]["labeled_edges"]}
    assert len(result) == 20 and all(a < b for a, b in result)
    return result


def sentence(actor: int, patient: int, verb: str, voice: str) -> str:
    if voice == "active":
        return f"{NAMES[actor]} {verb} {NAMES[patient]}."
    return f"{NAMES[patient]} was {verb} by {NAMES[actor]}."


def prompt_text(tokenizer, text: str, question: str) -> str:
    return tokenizer.apply_chat_template([
        {"role": "system", "content": "Answer the question with exactly one name from the sentence. Output only the name."},
        {"role": "user", "content": f"Sentence: {text}\nQuestion: {question}"},
    ], tokenize=False, add_generation_prompt=True)


def training_rows() -> list[dict]:
    rows = []
    for a, b in sorted(edges()):
        for actor, patient in ((a, b), (b, a)):
            for verb, voice, role in itertools.product(VERBS, ("active", "passive"),
                                                       ("actor", "patient")):
                question = ("Identify the actor (the person doing the action)." if role == "actor"
                            else "Identify the patient (the person acted on).")
                rows.append({"actor":actor,"patient":patient,"verb":verb,"voice":voice,
                             "role":role,"sentence":sentence(actor,patient,verb,voice),
                             "question":question,
                             "answer":NAMES[actor if role=="actor" else patient]})
    assert len(rows)==480
    order = np.random.default_rng(SEED).permutation(len(rows))
    return [rows[int(i)] for i in order]


def test_rows() -> list[dict]:
    seen = edges()
    rows=[]
    for a,b in itertools.combinations(range(10),2):
        for actor,patient in ((a,b),(b,a)):
            event=f"{a}:{b}:{actor}"
            for verb,voice,role,wording in itertools.product(
                VERBS,("active","passive"),("actor","patient"),("generic","verb_specific")):
                if role=="actor":
                    question=("Who did the action?" if wording=="generic"
                              else f"Who {verb} the other person?")
                else:
                    question=("Who was acted on?" if wording=="generic"
                              else f"Who was {verb}?")
                rows.append({"pair":[a,b],"seen":(a,b) in seen,"event":event,
                             "actor":actor,"patient":patient,"verb":verb,"voice":voice,
                             "role":role,"wording":wording,
                             "sentence":sentence(actor,patient,verb,voice),
                             "question":question,
                             "gold":NAMES[actor if role=="actor" else patient]})
    assert len(rows)==2160
    assert sum(r["seen"] for r in rows)==960
    return rows


def load(arm: str):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    tokenizer.padding_side="left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL_PATH,local_files_only=True,
                                               dtype=torch.float32)
    heads=None
    if arm!="pretrained_no_finetuning":
        add_adapters(model)
        heads=nn.ModuleList([nn.Linear(model.config.hidden_size,10),
                             nn.Linear(model.config.hidden_size,10)])
    model.config.use_cache=False
    return tokenizer,model,heads


def initial_trainable_hash(model, heads) -> str:
    state={name: p.detach().cpu() for name,p in model.named_parameters() if p.requires_grad}
    state.update({f"head.{name}": p.detach().cpu() for name,p in heads.named_parameters()})
    h=hashlib.sha256()
    for name,p in sorted(state.items()):
        h.update(name.encode());h.update(p.numpy().tobytes())
    return h.hexdigest()


def train(arm: str, tokenizer, model, heads) -> dict:
    assert heads is not None
    rows=training_rows()
    trainable=[p for p in model.parameters() if p.requires_grad]+list(heads.parameters())
    optimizer=torch.optim.AdamW(trainable,lr=1e-4)
    start_hash=initial_trainable_hash(model,heads)
    diagnostics=[]
    model.train();heads.train()
    for step,row in enumerate(rows,1):
        prefix=tokenizer.encode(prompt_text(tokenizer,row["sentence"],row["question"]),
                                add_special_tokens=False)
        answer=tokenizer.encode(row["answer"],add_special_tokens=False)
        tokens=torch.tensor([prefix+answer+[tokenizer.eos_token_id]],dtype=torch.long)
        labels=torch.tensor([[-100]*len(prefix)+answer+[tokenizer.eos_token_id]],
                            dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        result=model(input_ids=tokens,labels=labels,output_hidden_states=True,use_cache=False)
        state=result.hidden_states[-1][:,len(prefix)-1,:]
        left,right=(row["actor"],row["patient"]) if arm=="ordered_roles" else (
            min(row["actor"],row["patient"]),max(row["actor"],row["patient"]))
        auxiliary=(F.cross_entropy(heads[0](state),torch.tensor([left]))
                   +F.cross_entropy(heads[1](state),torch.tensor([right])))/2
        loss=result.loss+.5*auxiliary
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        loss.backward()
        optimizer.step()
        diagnostics.append({"step":step,"language_loss":float(result.loss.detach()),
                            "auxiliary_loss":float(auxiliary.detach()),
                            "total_loss":float(loss.detach())})
        if step%80==0:
            recent=diagnostics[-40:]
            print(f"{arm} step={step}/480 language={np.mean([x['language_loss'] for x in recent]):.3f} auxiliary={np.mean([x['auxiliary_loss'] for x in recent]):.3f}",flush=True)
    return {"examples":len(rows),"updates":len(diagnostics),
            "initial_trainable_sha256":start_hash,
            "last_40":{key:float(np.mean([r[key] for r in diagnostics[-40:]]))
                       for key in ("language_loss","auxiliary_loss","total_loss")},
            "diagnostics":diagnostics}


def evaluate(arm: str, tokenizer, model) -> dict:
    rows=test_rows()
    model.eval()
    model.config.use_cache=True
    for start in range(0,len(rows),8):
        batch=rows[start:start+8]
        texts=[prompt_text(tokenizer,row["sentence"],row["question"]) for row in batch]
        tokenized=tokenizer(texts,padding=True,return_tensors="pt")
        with torch.inference_mode():
            generated=model.generate(**tokenized,max_new_tokens=8,do_sample=False,
                                     use_cache=True,pad_token_id=tokenizer.pad_token_id)
        outputs=tokenizer.batch_decode(generated[:,tokenized["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
        for row,raw in zip(batch,outputs):
            answer=raw.strip().strip(string.punctuation+" ")
            row.update({"raw":raw,"answer":answer,"valid":answer in (NAMES[row["actor"]],
                                                                       NAMES[row["patient"]]),
                        "correct":answer==row["gold"]})
        if (start+len(batch))%216==0:
            print(f"{arm} evaluated {start+len(batch)}/{len(rows)}",flush=True)
    groups=defaultdict(list)
    for row in rows:
        for key in (("overall",),(row["seen"],),(row["seen"],row["voice"]),
                    (row["seen"],row["role"]),(row["seen"],row["wording"]),
                    (row["seen"],row["voice"],row["role"],row["wording"])):
            groups[str(key)].append(row)
    summary={key:{"correct":sum(r["correct"] for r in items),
                  "valid":sum(r["valid"] for r in items),"total":len(items),
                  "accuracy":sum(r["correct"] for r in items)/len(items)}
             for key,items in groups.items()}
    return {"summary":summary,"rows":rows}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--arm",choices=ARMS,required=True)
    args=parser.parse_args()
    destination=ROOT/args.arm
    if destination.exists():
        raise FileExistsError(destination)
    tokenizer,model,heads=load(args.arm)
    training=None
    if heads is not None:
        training=train(args.arm,tokenizer,model,heads)
    evaluation=evaluate(args.arm,tokenizer,model)
    destination.mkdir(parents=True)
    if heads is not None:
        payload={"model":{name:p.detach().cpu() for name,p in model.named_parameters()
                          if p.requires_grad},"heads":heads.state_dict()}
        torch.save(payload,destination/"adapter.pt")
        adapter_hash=hashlib.sha256((destination/"adapter.pt").read_bytes()).hexdigest()
    else:
        adapter_hash=None
    result={"arm":args.arm,"model":"Qwen/Qwen2.5-0.5B-Instruct",
            "protocol_sha256":hashlib.sha256((ROOT/"PROTOCOL.json").read_bytes()).hexdigest(),
            "adapter_sha256":adapter_hash,"training":training,"evaluation":evaluation}
    (destination/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"arm":args.arm,"seen":evaluation["summary"][str((True,))],
                      "unseen":evaluation["summary"][str((False,))],
                      "initial_trainable_sha256":training["initial_trainable_sha256"] if training else None},indent=2),flush=True)


if __name__=="__main__":
    main()
