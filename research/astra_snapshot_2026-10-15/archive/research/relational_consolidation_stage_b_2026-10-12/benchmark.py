#!/usr/bin/env python3
"""Deterministic Stage B generator and source-only benchmark audit.

No model weights are read or updated. No reserved natural-language data are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORMS = ("canonical", "passive", "object_cleft", "subject_cleft", "role_reversal")
NAMES = (
    "Ari", "Bela", "Cora", "Davi", "Eli", "Fara", "Gino", "Hana", "Ivo",
    "Juna", "Kira", "Lori", "Mara", "Niko", "Olin", "Pavi", "Rina", "Sora",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def pair(a: int, b: int) -> tuple[int, int]:
    return (min(a, b), max(a, b))


def make_graph(n: int, degree: int, seed: int, forbidden: tuple[int, int]) -> list[tuple[int, int]]:
    assert degree % 2 == 0 and degree < n
    rng = random.Random(seed)
    base = {pair(i, (i + offset) % n) for i in range(n) for offset in range(1, degree // 2 + 1)}
    nodes = list(range(n))
    for _ in range(1000):
        rng.shuffle(nodes)
        edges = {pair(nodes[a], nodes[b]) for a, b in base}
        if forbidden not in edges:
            break
    else:
        raise RuntimeError("could not keep focal pair out of training graph")
    # Degree-preserving double-edge switches vary topology beyond relabelling.
    edge_list = sorted(edges)
    accepted = 0
    for _ in range(10000):
        i, j = rng.sample(range(len(edge_list)), 2)
        a, b = edge_list[i]
        c, d = edge_list[j]
        if rng.randrange(2):
            c, d = d, c
        x, y = pair(a, c), pair(b, d)
        if a == c or b == d or x == y or forbidden in (x, y):
            continue
        if x in edges or y in edges:
            continue
        edges.remove(pair(a, b))
        edges.remove(pair(c, d))
        edges.update((x, y))
        edge_list = sorted(edges)
        accepted += 1
        if accepted >= 500:
            break
    if accepted < 100:
        raise RuntimeError(f"insufficient graph switches: {accepted}")
    deg = Counter(v for e in edges for v in e)
    assert len(edges) == n * degree // 2
    assert all(deg[i] == degree for i in range(n))
    return sorted(edges)


def event(family: int, split: str, a: int, b: int, verb: str) -> dict:
    return {
        "world_id": f"world-{family}",
        "event_id": f"f{family}-{split}-{a:02d}-{b:02d}-{verb}",
        "split": split,
        "predicate_id": verb,
        "actor_id": a,
        "patient_id": b,
        "other_role_type": "semantic_patient",
        "truth_source": "sentence_schema",
        "unordered_pair": list(pair(a, b)),
    }


def render(ev: dict, form: str) -> dict:
    a, b, verb = ev["actor_id"], ev["patient_id"], ev["predicate_id"]
    actor, patient = NAMES[a], NAMES[b]
    if form == "symbolic":
        sentence, mention, gold = f"ACT[{actor},{patient},{verb}]", [a, b], [a, b]
    elif form == "canonical":
        sentence, mention, gold = f"{actor} {verb} {patient}.", [a, b], [a, b]
    elif form == "passive":
        sentence, mention, gold = f"{patient} was {verb} by {actor}.", [b, a], [a, b]
    elif form == "object_cleft":
        sentence, mention, gold = f"It was {patient} that {actor} {verb}.", [b, a], [a, b]
    elif form == "subject_cleft":
        sentence, mention, gold = f"It was {actor} who {verb} {patient}.", [a, b], [a, b]
    elif form == "role_reversal":
        sentence, mention, gold = f"{patient} {verb} {actor}.", [b, a], [b, a]
    else:
        raise ValueError(form)
    return {
        "event_id": ev["event_id"],
        "split": ev["split"],
        "construction_id": form,
        "sentence": sentence,
        "mention_order": mention,
        "gold_actor_id": gold[0],
        "gold_patient_id": gold[1],
        "role_map_source": ev["event_id"] if form != "role_reversal" else "schema_swap",
    }


def family_data(cfg: dict, family: int) -> dict:
    n = cfg["entities_per_family"]
    seed = cfg["seed"] + 1009 * family
    focal = tuple(cfg["focal_withheld_pair"])
    train_pairs = make_graph(n, cfg["degree_per_entity"], seed, focal)
    train_set = set(train_pairs)
    unused = [pair(a, b) for a in range(n) for b in range(a + 1, n) if pair(a, b) not in train_set and pair(a, b) != focal]
    rng = random.Random(seed + 709)
    rng.shuffle(unused)
    k = cfg["withheld_pairs_per_split"]
    dev_pairs = sorted(unused[:k])
    test_pairs = sorted([focal] + unused[k:k + k - 1])
    if set(dev_pairs) & set(test_pairs) or set(dev_pairs + test_pairs) & train_set:
        raise AssertionError("pair leakage")

    def events(pairs: list[tuple[int, int]], split: str) -> list[dict]:
        return [event(family, split, a, b, verb)
                for a, b in pairs for verb in cfg["verbs"] for a, b in ((a, b), (b, a))]

    return {
        "family": family,
        "graph_edges": [list(p) for p in train_pairs],
        "development_pairs": [list(p) for p in dev_pairs],
        "confirmation_pairs": [list(p) for p in test_pairs],
        "train_events": events(train_pairs, "train"),
        "development_events": events(dev_pairs, "development"),
        "confirmation_events": events(test_pairs, "confirmation"),
    }


def baseline_predictions(row: dict, most_common_actor: int) -> dict[str, int]:
    first, last = row["mention_order"]
    form = row["construction_id"]
    return {
        "first_participant": first,
        "last_participant": last,
        # The first named participant is the overt subject of canonical/passive
        # clauses; in clefts this is only a simple surface proxy.
        "grammatical_subject_proxy": first,
        "by_phrase_else_first": last if form == "passive" else first,
        "lexical_memorization_fallback": most_common_actor,
        # Transparent ceiling: this finite grammar has a perfect hand-coded parser.
        "construction_aware_oracle": last if form in ("passive", "object_cleft") else first,
    }


def audit(data: list[dict], cfg: dict) -> dict:
    all_ids = set()
    simple = ("first_participant", "last_participant", "grammatical_subject_proxy",
              "by_phrase_else_first", "lexical_memorization_fallback")
    counts: dict[str, Counter] = {name: Counter() for name in (*simple, "construction_aware_oracle")}
    family_summaries = []
    for fam in data:
        train = fam["train_events"]
        development = fam["development_events"]
        confirmation = fam["confirmation_events"]
        train_keys = {tuple(ev["unordered_pair"]) for ev in train}
        dev_keys = {tuple(ev["unordered_pair"]) for ev in development}
        test_keys = {tuple(ev["unordered_pair"]) for ev in confirmation}
        assert not (train_keys & dev_keys or train_keys & test_keys or dev_keys & test_keys)
        assert tuple(cfg["focal_withheld_pair"]) in test_keys
        assert len(train_keys) == 54 and len(dev_keys) == len(test_keys) == 15
        assert len(train) == 324 and len(development) == len(confirmation) == 90
        assert {ev["predicate_id"] for ev in train} == set(cfg["verbs"])
        train_derived = s2_pairs_from_train(fam)
        assert {tuple(row["source_unordered_pair"]) for row in train_derived} == train_keys
        def rendered_sentences(events: list[dict]) -> set[str]:
            return {render(ev, form)["sentence"] for ev in events for form in FORMS}
        train_rendered = rendered_sentences(train)
        dev_rendered = rendered_sentences(development)
        test_rendered = rendered_sentences(confirmation)
        assert not (train_rendered & dev_rendered or train_rendered & test_rendered or dev_rendered & test_rendered)
        assert not ({(tuple(ev["unordered_pair"]), ev["predicate_id"]) for ev in train}
                    & {(tuple(ev["unordered_pair"]), ev["predicate_id"]) for ev in confirmation})
        train_actor_count = Counter(ev["actor_id"] for ev in train)
        assert len(set(train_actor_count.values())) == 1
        most_common = sorted(train_actor_count)[0]
        train_sentences = {render(ev, "canonical")["sentence"] for ev in train}
        actor_positions = Counter()
        gold_actor_count = Counter()
        for ev in confirmation:
            if ev["event_id"] in all_ids:
                raise AssertionError("duplicate event ID")
            all_ids.add(ev["event_id"])
            rows = [render(ev, form) for form in FORMS]
            assert all(row["gold_actor_id"] == ev["actor_id"] for row in rows[:-1])
            assert rows[-1]["gold_actor_id"] == ev["patient_id"]
            assert all(row["gold_actor_id"] != row["gold_patient_id"] for row in rows)
            assert all(row["sentence"] not in train_sentences for row in rows)
            by_name = {name: [] for name in counts}
            for row in rows:
                actor_positions["first" if row["mention_order"][0] == row["gold_actor_id"] else "second"] += 1
                gold_actor_count[row["gold_actor_id"]] += 1
                pred = baseline_predictions(row, most_common)
                for name in counts:
                    correct = int(pred[name] == row["gold_actor_id"])
                    counts[name]["correct_actor"] += correct
                    counts[name]["total_rows"] += 1
                    by_name[name].append(correct)
            for name, flags in by_name.items():
                counts[name]["joint_events"] += int(all(flags))
                counts[name]["total_events"] += 1
        assert actor_positions["first"] == 3 * len(confirmation)
        assert actor_positions["second"] == 2 * len(confirmation)
        expected_actor_count = Counter({k: v * 5 for k, v in Counter(e["actor_id"] for e in confirmation).items()})
        assert gold_actor_count == expected_actor_count
        edges = set(map(tuple, fam["graph_edges"]))
        triangle_count = sum(
            pair(a, c) in edges and pair(a, d) in edges and pair(c, d) in edges
            for a in range(cfg["entities_per_family"])
            for c in range(a + 1, cfg["entities_per_family"])
            for d in range(c + 1, cfg["entities_per_family"])
        )
        focal_a, focal_b = cfg["focal_withheld_pair"]
        common_neighbors = sum(pair(focal_a, v) in edges and pair(focal_b, v) in edges
                               for v in range(cfg["entities_per_family"]) if v not in (focal_a, focal_b))
        family_summaries.append({
            "family": fam["family"],
            "graph_edges": len(fam["graph_edges"]),
            "train_events": len(train),
            "development_events": len(development),
            "confirmation_events": len(confirmation),
            "confirmation_actor_first_rows": actor_positions["first"],
            "confirmation_actor_second_rows": actor_positions["second"],
            "train_actor_count_per_entity": min(train_actor_count.values()),
            "triangles": triangle_count,
            "focal_pair_common_neighbors": common_neighbors,
            "train_derived_s2_pairs": len(train_derived),
            "cross_split_exact_sentence_overlap": 0,
        })
    scores = {
        name: {
            "actor_accuracy": round(c["correct_actor"] / c["total_rows"], 6),
            "joint_event_accuracy": round(c["joint_events"] / c["total_events"], 6),
            **dict(c),
        }
        for name, c in counts.items()
    }
    gates = {
        "split_and_semantics_checks": "pass",
        "simple_heuristic_actor_gate": "pass" if all(scores[n]["actor_accuracy"] < cfg["simple_heuristic_actor_accuracy_fail_at_or_above"] for n in simple) else "fail",
        "simple_heuristic_joint_gate": "pass" if all(scores[n]["joint_event_accuracy"] < cfg["simple_heuristic_joint_event_accuracy_fail_at_or_above"] for n in simple) else "fail",
        "human_grammar_review": "pending",
        "model_training": "not_started",
    }
    return {"families": family_summaries, "heuristics": scores, "gates": gates}


def jsonl_bytes(records: list[dict]) -> bytes:
    return b"".join(stable_json(r) for r in records)


def s2_pairs_from_train(family: dict) -> list[dict]:
    """Only training events may supply supervision for consolidation pairs."""
    allowed = {ev["event_id"] for ev in family["train_events"]}
    result = []
    for ev in family["train_events"]:
        if ev["split"] != "train" or ev["event_id"] not in allowed:
            raise ValueError("non-training event offered to S2 generator")
        base = render(ev, "canonical")
        for form in ("passive", "object_cleft", "subject_cleft", "role_reversal"):
            other = render(ev, form)
            result.append({
                "source_event_id": ev["event_id"],
                "source_unordered_pair": ev["unordered_pair"],
                "source": base["sentence"],
                "transformed": other["sentence"],
                "transformation": form,
                "role_map": "swap" if form == "role_reversal" else "identity",
                "gold_actor_id": other["gold_actor_id"],
                "gold_patient_id": other["gold_patient_id"],
            })
    forbidden = {tuple(ev["unordered_pair"]) for split in ("development_events", "confirmation_events") for ev in family[split]}
    if any(tuple(row["source_unordered_pair"]) in forbidden for row in result):
        raise AssertionError("withheld pair leaked into S2 pairs")
    return result


def build() -> None:
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    data = [family_data(cfg, f) for f in range(cfg["families"])]
    report = audit(data, cfg)
    out = ROOT / "generated"
    out.mkdir(exist_ok=True)
    hashes = {}
    for fam in data:
        fdir = out / f"family_{fam['family']:02d}"
        fdir.mkdir(exist_ok=True)
        for key in ("graph_edges", "development_pairs", "confirmation_pairs", "train_events", "development_events", "confirmation_events"):
            payload = jsonl_bytes(fam[key]) if key.endswith("events") else stable_json(fam[key])
            path = fdir / (key.upper() + (".jsonl" if key.endswith("events") else ".json"))
            path.write_bytes(payload)
            hashes[str(path.relative_to(ROOT))] = digest(payload)
        s2 = jsonl_bytes(s2_pairs_from_train(fam))
        s2_path = fdir / "S2_TRAIN_DERIVATIONS.jsonl"
        s2_path.write_bytes(s2)
        hashes[str(s2_path.relative_to(ROOT))] = digest(s2)
    report["generator_version"] = cfg["generator_version"]
    report["protocol_sha256"] = digest((ROOT / "PROTOCOL.json").read_bytes())
    report["generator_source_sha256"] = digest(Path(__file__).read_bytes())
    report["artifact_sha256"] = hashes
    (ROOT / "AUDIT.json").write_bytes(stable_json(report))
    print(json.dumps({"gates": report["gates"], "heuristics": report["heuristics"]}, indent=2))


def verify() -> None:
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    data = [family_data(cfg, f) for f in range(cfg["families"])]
    expected = audit(data, cfg)
    actual = json.loads((ROOT / "AUDIT.json").read_text())
    assert actual["families"] == expected["families"]
    assert actual["heuristics"] == expected["heuristics"]
    assert actual["gates"] == expected["gates"]
    assert actual["protocol_sha256"] == digest((ROOT / "PROTOCOL.json").read_bytes())
    assert actual["generator_source_sha256"] == digest(Path(__file__).read_bytes())
    regenerated_hashes = {}
    for fam in data:
        fdir = Path("generated") / f"family_{fam['family']:02d}"
        for key in ("graph_edges", "development_pairs", "confirmation_pairs", "train_events", "development_events", "confirmation_events"):
            payload = jsonl_bytes(fam[key]) if key.endswith("events") else stable_json(fam[key])
            rel = str(fdir / (key.upper() + (".jsonl" if key.endswith("events") else ".json")))
            regenerated_hashes[rel] = digest(payload)
        regenerated_hashes[str(fdir / "S2_TRAIN_DERIVATIONS.jsonl")] = digest(jsonl_bytes(s2_pairs_from_train(fam)))
    assert actual["artifact_sha256"] == regenerated_hashes
    for rel, expected_hash in regenerated_hashes.items():
        assert digest((ROOT / rel).read_bytes()) == expected_hash, rel
    print("verified deterministic artifacts and audit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "verify"))
    args = parser.parse_args()
    globals()[args.action]()
