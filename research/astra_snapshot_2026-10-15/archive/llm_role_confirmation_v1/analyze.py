"""Summarize the fixed 120-event Qwen confirmation without treating prompts as independent."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def signflip_p(effects: list[float]) -> float:
    observed = abs(mean(effects))
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=len(effects)):
        if abs(mean(s * x for s, x in zip(signs, effects))) >= observed - 1e-12:
            extreme += 1
    return extreme / (2 ** len(effects))


def main() -> None:
    obj = json.loads((ROOT / "RESULT.json").read_text())
    assert obj["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
    rows = obj["records"]
    assert len(rows) == 480
    event_rows = defaultdict(dict)
    pair_events = defaultdict(list)
    cells = defaultdict(list)
    for row in rows:
        key = (row["form"], row["wording"])
        assert key not in event_rows[row["event"]]
        event_rows[row["event"]][key] = row
        cells[(row["set"], *key)].append(row)
    assert len(event_rows) == 120
    events = []
    for event, group in event_rows.items():
        assert len(group) == 4
        sample = next(iter(group.values()))
        active_gap = int(group["active", "performed"]["correct"]) - int(group["active", "generic"]["correct"])
        passive_gap = int(group["passive", "performed"]["correct"]) - int(group["passive", "generic"]["correct"])
        entry = {"event": event, "set": sample["set"], "pair": sample["pair"],
                 "active_wording_effect": active_gap,
                 "passive_wording_effect": passive_gap,
                 "interaction": active_gap - passive_gap,
                 "all_four_correct": all(r["correct"] for r in group.values()),
                 "active_both_correct": all(group["active", w]["correct"] for w in ("generic", "performed")),
                 "passive_both_correct": all(group["passive", w]["correct"] for w in ("generic", "performed")),
                 "both_voices_correct_generic": all(group[f, "generic"]["correct"] for f in ("active", "passive")),
                 "both_voices_correct_performed": all(group[f, "performed"]["correct"] for f in ("active", "passive")),
                 "wording_agreement_active": group["active", "generic"]["answer"] == group["active", "performed"]["answer"],
                 "voice_agreement_generic": group["active", "generic"]["answer"] == group["passive", "generic"]["answer"]}
        events.append(entry)
        pair_events[sample["pair"]].append(entry)
    assert len(pair_events) == 15
    pair_effects = {pair: {metric: mean(e[metric] for e in group)
                           for metric in ("active_wording_effect", "passive_wording_effect", "interaction")}
                    for pair, group in pair_events.items()}
    metrics = ("active_wording_effect", "passive_wording_effect", "interaction")
    report = {
        "n_events": 120, "n_prompts": 480, "n_name_pairs": 15,
        "cells": {f"set{s}/{form}/{wording}": {
            "n": len(cells[s, form, wording]),
            "correct": sum(r["correct"] for r in cells[s, form, wording]),
            "valid": sum(r["valid"] for r in cells[s, form, wording]),
            "mean_actor_margin": mean(r["actor_margin"] for r in cells[s, form, wording]),
        } for s, form, wording in itertools.product(range(3), ("active", "passive"), ("generic", "performed"))},
        "overall_effects": {metric: {"event_mean": mean(e[metric] for e in events),
                                      "pair_mean": mean(p[metric] for p in pair_effects.values()),
                                      "pair_signflip_p_two_sided": signflip_p([p[metric] for p in pair_effects.values()]),
                                      "by_set": [mean(e[metric] for e in events if e["set"] == s) for s in range(3)]}
                            for metric in metrics},
        "event_metrics": {metric: {"count": sum(e[metric] for e in events), "total": 120}
                          for metric in ("all_four_correct", "active_both_correct", "passive_both_correct",
                                         "both_voices_correct_generic", "both_voices_correct_performed",
                                         "wording_agreement_active", "voice_agreement_generic")},
        "pair_effects": pair_effects,
    }
    (ROOT / "SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
