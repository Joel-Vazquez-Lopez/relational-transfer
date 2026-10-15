"""Matched role-transformation training study on local Qwen2.5-0.5B."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_role_training_v0.feasibility import add_adapters


HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_repair_v2"
MODEL = Path("/Users/joelvazquezlopez/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775")
GRAPH = HERE / "role_transfer_phase01/role_aux_confirmation_v1/GRAPH_MANIFEST.json"
NAMES = ("Stephen", "Alexander", "Sean", "Alan", "Allen", "Bruce", "Carl", "Louis", "Roy", "Juan")
TRAIN_VERBS = ("chased", "helped", "followed")
TEST_VERBS = ("coached", "interviewed")
QUESTION_TRAIN = "Who performed the action?"
QUESTIONS = {
    "performed": QUESTION_TRAIN,
    "did": "Who did the action?",
    "carried_out": "Which person carried out the action described in the sentence?",
}
FORMS = ("active", "passive", "actor_cleft", "patient_cleft")
ARMS = ("pretrained", "ordinary", "generic_consistency", "ordered_roles", "ordered_plus_transformations")
SEED = 830110
EPOCHS = 2
LR = 1e-4
ORDERED_WEIGHT = 0.5
VOICE_WEIGHT = 0.5
SWAP_WEIGHT = 0.5


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edges() -> set[tuple[int, int]]:
    data = json.loads(GRAPH.read_text())
    seen = {tuple(e) for e in data["worlds"]["fresh0"]["labeled_edges"]}
    assert len(seen) == 20 and all(a < b for a, b in seen)
    return seen


def sentence(actor: int, patient: int, verb: str, form: str) -> str:
    a, p = NAMES[actor], NAMES[patient]
    if form == "active":
        return f"{a} {verb} {p}."
    if form == "passive":
        return f"{p} was {verb} by {a}."
    if form == "actor_cleft":
        return f"It was {a} who {verb} {p}."
    if form == "patient_cleft":
        return f"It was {p} whom {a} {verb}."
    raise ValueError(form)


def train_groups() -> list[list[dict]]:
    groups = []
    for left, right in sorted(edges()):
        for verb in TRAIN_VERBS:
            group = []
            for actor, patient in ((left, right), (right, left)):
                for form in ("active", "passive"):
                    group.append({"pair": [left, right], "actor": actor,
                                  "patient": patient, "verb": verb, "form": form,
                                  "sentence": sentence(actor, patient, verb, form),
                                  "question": QUESTION_TRAIN,
                                  "gold": NAMES[actor]})
            groups.append(group)
    assert len(groups) == 60 and all(len(g) == 4 for g in groups)
    return groups


def test_rows() -> list[dict]:
    seen = edges()
    rows = []
    for left, right in itertools.combinations(range(len(NAMES)), 2):
        for actor, patient in ((left, right), (right, left)):
            for verb in TRAIN_VERBS + TEST_VERBS:
                event = f"{left}:{right}:{actor}:{verb}"
                for form, wording in itertools.product(FORMS, QUESTIONS):
                    rows.append({"pair": [left, right], "seen_pair": (left, right) in seen,
                                 "seen_verb": verb in TRAIN_VERBS,
                                 "event": event, "actor": actor, "patient": patient,
                                 "verb": verb, "form": form, "wording": wording,
                                 "sentence": sentence(actor, patient, verb, form),
                                 "question": QUESTIONS[wording], "gold": NAMES[actor]})
    assert len(rows) == 45 * 2 * 5 * 4 * 3
    return rows


def prompt(tokenizer, row: dict) -> str:
    return tokenizer.apply_chat_template([
        {"role": "system", "content": "Answer with exactly one of the two names in the sentence. Output only the name."},
        {"role": "user", "content": f"Sentence: {row['sentence']}\nQuestion: {row['question']}"},
    ], tokenize=False, add_generation_prompt=True)


def protocol() -> dict:
    return {
        "status": "fixed_before_training", "date": "2026-10-10",
        "purpose": "Pilot whether ordered semantic-role supervision and transformation-aware consistency improve correct role transfer in a pretrained decoder model.",
        "model": "Qwen2.5-0.5B-Instruct local unmodified snapshot before LoRA",
        "model_weights_sha256": sha(MODEL / "model.safetensors"),
        "source_sha256": sha(HERE / "llm_role_repair_v2.py"),
        "graph_sha256": sha(GRAPH),
        "design": "One 10-name graph, 20 observed and 25 withheld unordered pairs. All arms train on identical 240 examples: 20 observed pairs x 3 verbs x 2 directions x 2 voices. Only actor question 'Who performed the action?' is in training. Four matched training groups plus unmodified reference.",
        "arms": ARMS,
        "training": {
            "ordinary": "Supervised answer loss alone.",
            "generic_consistency": "Same supervised loss plus Jensen-Shannon agreement of two-name actor distributions between active/passive renderings of each directed event.",
            "ordered_roles": "Same supervised loss plus ordered actor and patient ID cross-entropy from the final prompt state.",
            "ordered_plus_transformations": "Same supervised loss and ordered heads, plus voice agreement and Jensen-Shannon equivariance when the two participant roles are reversed.",
            "matched": "Same group ordering, all four sentence variants per group, LoRA architecture and initialization, two epochs (120 updates), learning rate 1e-4, no test-based tuning. Each additional objective uses the same four training examples.",
            "weights": {"ordered": ORDERED_WEIGHT, "voice": VOICE_WEIGHT, "role_swap": SWAP_WEIGHT},
        },
        "evaluation": "All 45 name pairs x two directions x five verbs (three seen and two unseen) x four forms (two seen and two unseen clefts) x three fixed actor questions (one seen and two unseen) = 5400 prompts per arm. Greedy first token; exact actor name counts correct; invalid token wrong. Primary pilot subset is 25 withheld pairs x 2 directions x 3 seen verbs x 2 seen forms x 3 questions (900 prompts). Also report unseen verbs, unseen forms, all-variant event correctness, invalid rate, per-pair results and familiar-pair retention. No post-hoc exclusions.",
        "interpretation": "One graph, one training start and one small model family constitute a feasibility and direction pilot. A repair claim requires fresh graph, independent starts and model families with frozen replication criteria.",
    }


def prepare() -> None:
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "PROTOCOL.json").exists():
        raise FileExistsError(ROOT / "PROTOCOL.json")
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    for name in NAMES:
        assert len(tok.encode(name, add_special_tokens=False)) == 1, name
        assert len(tok.encode(" " + name, add_special_tokens=False)) == 1, name
    (ROOT / "PROTOCOL.json").write_text(json.dumps(protocol(), indent=2) + "\n")
    (ROOT / "TRAIN_GROUPS.json").write_text(json.dumps(train_groups(), indent=2) + "\n")
    (ROOT / "TEST_ROWS.json").write_text(json.dumps(test_rows(), indent=2) + "\n")
    print("Frozen 60 training groups and 5400 evaluation prompts", flush=True)


def validate_fixed() -> tuple[list[list[dict]], list[dict]]:
    p = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert sha(HERE / "llm_role_repair_v2.py") == p["source_sha256"]
    assert sha(GRAPH) == p["graph_sha256"]
    groups = json.loads((ROOT / "TRAIN_GROUPS.json").read_text())
    rows = json.loads((ROOT / "TEST_ROWS.json").read_text())
    assert groups == train_groups() and rows == test_rows()
    return groups, rows


def load(arm: str):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
                                                dtype=torch.float32)
    heads = None
    if arm != "pretrained":
        add_adapters(model)
        if arm in ("ordered_roles", "ordered_plus_transformations"):
            heads = nn.ModuleList([nn.Linear(model.config.hidden_size, len(NAMES)),
                                   nn.Linear(model.config.hidden_size, len(NAMES))])
    model.config.use_cache = False
    return tokenizer, model, heads


def adapter_hash(model) -> str:
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        if param.requires_grad:
            h.update(name.encode())
            h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def name_logit(logits, tokenizer, name: str):
    ids = [tokenizer.encode(name, add_special_tokens=False)[0],
           tokenizer.encode(" " + name, add_special_tokens=False)[0]]
    return torch.logsumexp(logits[..., ids], dim=-1)


def js(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = (p + q) / 2
    return (F.kl_div(m.log(), p, reduction="batchmean") +
            F.kl_div(m.log(), q, reduction="batchmean")) / 2


def encode_group(tokenizer, group: list[dict]):
    data = []
    for row in group:
        prefix = tokenizer.encode(prompt(tokenizer, row), add_special_tokens=False)
        answer = tokenizer.encode(row["gold"], add_special_tokens=False)
        assert len(answer) == 1
        data.append((prefix, answer))
    length = max(len(prefix) + 2 for prefix, _ in data)
    tokens, labels = [], []
    for prefix, answer in data:
        pad = length - len(prefix) - 2
        tokens.append([tokenizer.pad_token_id] * pad + prefix + answer + [tokenizer.eos_token_id])
        labels.append([-100] * (pad + len(prefix)) + answer + [tokenizer.eos_token_id])
    return torch.tensor(tokens), torch.tensor(labels), length - 3


def train(arm: str, tokenizer, model, heads, groups: list[list[dict]]) -> dict:
    parameters = [p for p in model.parameters() if p.requires_grad]
    if heads is not None:
        parameters += list(heads.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=LR)
    initial_hash = adapter_hash(model)
    head_hash = None
    if heads is not None:
        h = hashlib.sha256()
        for name, param in heads.named_parameters():
            h.update(name.encode()); h.update(param.detach().numpy().tobytes())
        head_hash = h.hexdigest()
    rng = np.random.default_rng(SEED)
    order = [int(i) for _ in range(EPOCHS) for i in rng.permutation(len(groups))]
    diagnostics = []
    model.train()
    if heads is not None:
        heads.train()
    started = time.perf_counter()
    for step, group_id in enumerate(order, 1):
        group = groups[group_id]
        tokens, labels, prompt_idx = encode_group(tokenizer, group)
        optimizer.zero_grad(set_to_none=True)
        result = model(input_ids=tokens, labels=labels, output_hidden_states=heads is not None,
                       use_cache=False)
        language = result.loss
        auxiliary = torch.zeros((), dtype=language.dtype)
        voice = torch.zeros((), dtype=language.dtype)
        swap = torch.zeros((), dtype=language.dtype)
        if heads is not None:
            states = result.hidden_states[-1][:, prompt_idx, :]
            actor_ids = torch.tensor([row["actor"] for row in group])
            patient_ids = torch.tensor([row["patient"] for row in group])
            auxiliary = (F.cross_entropy(heads[0](states), actor_ids) +
                         F.cross_entropy(heads[1](states), patient_ids)) / 2
        if arm in ("generic_consistency", "ordered_plus_transformations"):
            left, right = [NAMES[i] for i in group[0]["pair"]]
            final_logits = result.logits[:, prompt_idx, :]
            pair_logits = torch.stack((name_logit(final_logits, tokenizer, left),
                                       name_logit(final_logits, tokenizer, right)), dim=-1)
            probs = F.softmax(pair_logits, dim=-1)
            voice = (js(probs[0:1], probs[1:2]) + js(probs[2:3], probs[3:4])) / 2
            if arm == "ordered_plus_transformations":
                swap = (js(probs[0:1], probs[2:3].flip(-1)) +
                        js(probs[1:2], probs[3:4].flip(-1))) / 2
        loss = language + ORDERED_WEIGHT * auxiliary + VOICE_WEIGHT * voice + SWAP_WEIGHT * swap
        if not torch.isfinite(loss):
            raise FloatingPointError((arm, step))
        loss.backward()
        optimizer.step()
        diagnostics.append({"step": step, "group": group_id,
                            "language": float(language.detach()),
                            "ordered": float(auxiliary.detach()),
                            "voice": float(voice.detach()), "swap": float(swap.detach()),
                            "total": float(loss.detach())})
        if step % 20 == 0:
            print(f"{arm}: {step}/{len(order)} updates, elapsed {time.perf_counter()-started:.0f}s, recent loss {np.mean([d['total'] for d in diagnostics[-20:]]):.3f}", flush=True)
    return {"updates": len(order), "train_examples": len(groups) * 4,
            "initial_adapter_sha256": initial_hash,
            "initial_head_sha256": head_hash,
            "elapsed_seconds": time.perf_counter() - started,
            "diagnostics": diagnostics}


@torch.inference_mode()
def evaluate(arm: str, tokenizer, model, rows: list[dict]) -> list[dict]:
    model.eval()
    records = []
    started = time.perf_counter()
    for start in range(0, len(rows), 16):
        batch = rows[start:start+16]
        inputs = tokenizer([prompt(tokenizer, row) for row in batch], padding=True,
                           return_tensors="pt")
        logits = model(**inputs, use_cache=False).logits[:, -1]
        for row, score in zip(batch, logits):
            answer = tokenizer.decode([int(score.argmax())]).strip()
            records.append({**row, "answer": answer,
                            "valid": answer in (NAMES[row["actor"]], NAMES[row["patient"]]),
                            "correct": answer == row["gold"]})
        if (start + len(batch)) % 600 == 0:
            print(f"{arm}: evaluated {start+len(batch)}/{len(rows)}, elapsed {time.perf_counter()-started:.0f}s", flush=True)
    return records


def summarize(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in records:
        groups["all"].append(row)
        groups[f"pair_seen={row['seen_pair']}"].append(row)
        groups[f"pair_seen={row['seen_pair']},verb_seen={row['seen_verb']},form={row['form']},wording={row['wording']}"].append(row)
        groups[f"pair_seen={row['seen_pair']},verb_seen={row['seen_verb']},form={row['form']}"].append(row)
    summary = {key: {"correct": sum(row["correct"] for row in group),
                     "valid": sum(row["valid"] for row in group),
                     "total": len(group)} for key, group in groups.items()}
    for label, condition in (("heldout_pairs_seen_forms_seen_verbs",
                              lambda row: not row["seen_pair"] and row["seen_verb"] and row["form"] in ("active", "passive")),
                             ("heldout_pairs_unseen_verbs",
                              lambda row: not row["seen_pair"] and not row["seen_verb"]),
                             ("heldout_pairs_unseen_forms",
                              lambda row: not row["seen_pair"] and row["form"] in ("actor_cleft", "patient_cleft"))):
        subset = [row for row in records if condition(row)]
        summary[label] = {"correct": sum(row["correct"] for row in subset),
                          "valid": sum(row["valid"] for row in subset),
                          "total": len(subset)}
        by_event = defaultdict(list)
        for row in subset:
            by_event[row["event"]].append(row)
        summary[label]["all_variants_correct"] = sum(all(row["correct"] for row in group)
                                                      for group in by_event.values())
        summary[label]["events"] = len(by_event)
    return summary


def run(arm: str) -> None:
    groups, rows = validate_fixed()
    output = ROOT / arm
    if output.exists():
        raise FileExistsError(output)
    tokenizer, model, heads = load(arm)
    training = None if arm == "pretrained" else train(arm, tokenizer, model, heads, groups)
    records = evaluate(arm, tokenizer, model, rows)
    output.mkdir()
    if arm != "pretrained":
        torch.save({"adapter": {name: p.detach().cpu() for name, p in model.named_parameters()
                                if p.requires_grad},
                    "heads": heads.state_dict() if heads is not None else None}, output / "adapter.pt")
    result = {"arm": arm, "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
              "training": training, "summary": summarize(records), "records": records}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"arm": arm,
                      "primary": result["summary"]["heldout_pairs_seen_forms_seen_verbs"],
                      "unseen_verbs": result["summary"]["heldout_pairs_unseen_verbs"],
                      "unseen_forms": result["summary"]["heldout_pairs_unseen_forms"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        if args.arm is None:
            parser.error("--arm is required for run")
        run(args.arm)
