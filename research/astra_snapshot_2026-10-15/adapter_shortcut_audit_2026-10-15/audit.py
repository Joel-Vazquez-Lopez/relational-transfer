"""Read-only positional-shortcut audit of frozen adapter development outputs."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "adapter_readout_diagnostic_2026-10-15"
OUT = HERE / "AUDIT.json"
ARMS = ("pretrained", "ordinary", "ordered_roles", "unordered_pair")
FORMS = ("active", "passive", "patient_cleft")
INTERFACES = ("archived", "direct_actor", "direct_patient", "extraction")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_rate(n: int, d: int):
    return n / d if d else None


def names_in_sentence(item: dict) -> tuple[str, str]:
    matches = [(m.start(), name) for name in (item["actor"], item["patient"])
               for m in re.finditer(r"\b" + re.escape(name) + r"\b", item["sentence"])]
    if len(matches) != 2:
        raise ValueError("Expected each participant exactly once")
    return tuple(name for _, name in sorted(matches))


def observed_map(item: dict, record: dict, interface: str) -> str:
    first, last = names_in_sentence(item)
    parsed = record[interface]["parsed"]
    if not parsed["semantic_format_valid"]:
        return "invalid"
    if interface == "extraction":
        actor, patient = parsed["actor"], parsed["patient"]
        if (actor, patient) == (first, last):
            return "first_actor_last_patient"
        if (actor, patient) == (last, first):
            return "last_actor_first_patient"
        if (actor, patient) == (first, first):
            return "both_first"
        if (actor, patient) == (last, last):
            return "both_last"
        return "other_valid"
    name = parsed["name"]
    return "first" if name == first else "last" if name == last else "other_valid"


def score(item: dict, record: dict, interface: str) -> dict:
    mapping = observed_map(item, record, interface)
    first, last = names_in_sentence(item)
    subject = item["patient"] if item["form"] == "passive" else item["actor"]
    # "Subject" means the lexical subject of the event clause: actor in the
    # active/cleft templates, patient in the passive. Cleft matrix "It" is ignored.
    if interface == "extraction":
        parsed = record[interface]["parsed"]
        actor, patient = parsed["actor"], parsed["patient"]
        actor_ok, patient_ok = actor == item["actor"], patient == item["patient"]
        rule_matches = {"first": (actor, patient) == (first, last),
                        "last": (actor, patient) == (last, first),
                        "event_subject": (actor, patient) == (subject, item["patient"] if subject == item["actor"] else item["actor"])}
        return {"map": mapping, "valid": parsed["semantic_format_valid"],
                "actor_correct": actor_ok, "patient_correct": patient_ok,
                "both_correct": actor_ok and patient_ok,
                "rules": rule_matches}
    parsed = record[interface]["parsed"]
    name = parsed["name"]
    target = item["patient"] if interface == "direct_patient" else item["actor"]
    return {"map": mapping, "valid": parsed["semantic_format_valid"],
            "correct": name == target, "is_gold_actor": name == item["actor"],
            "is_gold_patient": name == item["patient"],
            "rules": {"first": name == first, "last": name == last,
                      "event_subject": name == subject}}


def summarize(rows: list[dict], interface: str) -> dict:
    n = len(rows)
    if not n:
        raise ValueError("Empty summary")
    maps = Counter(row["scores"][interface]["map"] for row in rows)
    valid = sum(row["scores"][interface]["valid"] for row in rows)
    result = {"n": n, "valid_n": valid, "invalid_n": n-valid,
              "map_counts": dict(maps), "map_rates_all": {key: count/n for key, count in maps.items()},
              "map_rates_valid": {key: count/valid for key, count in maps.items() if key != "invalid"} if valid else {},
              "fixed_rule_matches_all": {}, "fixed_rule_matches_valid": {}}
    for rule in ("first", "last", "event_subject"):
        count = sum(row["scores"][interface]["rules"][rule] for row in rows)
        result["fixed_rule_matches_all"][rule] = {"count": count, "rate": count/n}
        result["fixed_rule_matches_valid"][rule] = {"count": count, "rate": safe_rate(count,valid)}
    if interface == "extraction":
        for role in ("actor", "patient", "both"):
            count = sum(row["scores"][interface][f"{role}_correct"] for row in rows)
            result[f"{role}_correct_all"] = {"count": count, "rate": count/n}
            result[f"{role}_correct_valid"] = {"count": count, "rate": safe_rate(count,valid)}
    else:
        for key in ("correct", "is_gold_actor", "is_gold_patient"):
            count = sum(row["scores"][interface][key] for row in rows)
            result[f"{key}_all"] = {"count": count, "rate": count/n}
            result[f"{key}_valid"] = {"count": count, "rate": safe_rate(count,valid)}
    return result


def paired_counts(rows: list[dict], direct: str, extract_role: str) -> dict:
    def value(row, interface, role):
        if interface == "extraction":
            return row["scores"][interface][f"{role}_correct"]
        return row["scores"][interface]["correct"]
    if extract_role == "both":
        direct_values = [value(r,"direct_actor","actor") and value(r,"direct_patient","patient") for r in rows]
    else:
        direct_values = [value(r,direct,extract_role) for r in rows]
    extract_values = [value(r,"extraction",extract_role) for r in rows]
    return {"n":len(rows), "direct_correct":sum(direct_values), "extract_correct":sum(extract_values),
            "extraction_gain":sum(e and not d for d,e in zip(direct_values,extract_values)),
            "extraction_loss":sum(d and not e for d,e in zip(direct_values,extract_values))}


def base_change(rows: list[dict], base_by_id: dict, interface: str, role: str) -> dict:
    def outcome(r):
        s=r["scores"][interface]
        return s[f"{role}_correct"] if interface=="extraction" else s["correct"]
    paired=[(outcome(base_by_id[r["id"]]),outcome(r)) for r in rows]
    return {"n":len(paired),"base_correct":sum(b for b,_ in paired),
            "adapter_correct":sum(a for _,a in paired),
            "newly_correct_adapter":sum(a and not b for b,a in paired),
            "newly_incorrect_adapter":sum(b and not a for b,a in paired)}


def sensitivity(rows: list[dict], interface: str) -> dict:
    malformed=[r for r in rows if not r["scores"][interface]["valid"]]
    unique_correct=unique_wrong=ambiguous=none=0
    examples=[]
    for row in malformed:
        item=row["item"]
        clean=row["record"][interface].get("clean_output",row["record"][interface]["raw_output"])
        found=[name for name in (item["actor"],item["patient"])
               if re.search(r"\b"+re.escape(name)+r"\b",clean)]
        if len(found)==1:
            target=item["patient"] if interface=="direct_patient" else item["actor"]
            if found[0]==target: unique_correct+=1
            else: unique_wrong+=1
            label="one_pair_name"
        elif len(found)==2:
            ambiguous+=1; label="both_pair_names"
        else:
            none+=1; label="no_pair_name"
        if len(examples)<6:
            examples.append({"id":row["id"],"raw":row["record"][interface]["raw_output"],
                             "category":label,"actor":item["actor"],"patient":item["patient"]})
    return {"malformed_n":len(malformed),"unique_pair_name_correct":unique_correct,
            "unique_pair_name_wrong":unique_wrong,"ambiguous_both_names":ambiguous,
            "no_pair_name":none,"examples":examples,
            "note":"Sensitivity inventory only; frozen primary metrics remain unchanged."}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    items=json.loads((SOURCE/"FROZEN_ITEMS.json").read_text())
    lookup={r["id"]:r for r in items}
    if len(items)!=288 or len(lookup)!=288:
        raise ValueError("Frozen item coverage")
    protocol=json.loads((SOURCE/"FROZEN_PROTOCOL.json").read_text())
    if protocol["item_sha256"]!=sha(SOURCE/"FROZEN_ITEMS.json"):
        raise ValueError("Frozen item hash")
    paths=sorted((SOURCE/"full_outputs").glob("*.json"))
    if len(paths)!=21:
        raise ValueError("Expected 21 raw unit files")
    rows=[]
    hashes={"audit_source":sha(Path(__file__)),
            "frozen_protocol":sha(SOURCE/"FROZEN_PROTOCOL.json"),
            "frozen_items":sha(SOURCE/"FROZEN_ITEMS.json"),
            "frozen_analysis":sha(SOURCE/"ANALYSIS.json"),
            "raw_units":{p.name:sha(p) for p in paths}}
    for path in paths:
        data=json.loads(path.read_text())
        if data["protocol_sha256"]!=hashes["frozen_protocol"] or len(data["records"])!=96:
            raise ValueError(f"Bad unit {path}")
        for record in data["records"]:
            item=lookup[record["id"]]
            if item["world"]!=data["world"]:
                raise ValueError("World mismatch")
            rows.append({"id":record["id"],"world":data["world"],"seed":data["seed"],
                         "arm":data["arm"],"form":item["form"],"pair":item["pair"],
                         "event":(data["world"],data["seed"],tuple(item["pair"]),item["actor_id"],item["verb"]),
                         "item":item,"record":record,
                         "scores":{face:score(item,record,face) for face in INTERFACES}})
    if len(rows)!=2016:
        raise ValueError("Raw coverage")
    aggregate={}; units={}; event_vectors={}; paired={}; changes={}; sensitivity_results={}
    for arm in ARMS:
        arm_rows=[r for r in rows if r["arm"]==arm]
        aggregate[arm]={}
        for form in FORMS:
            subset=[r for r in arm_rows if r["form"]==form]
            aggregate[arm][form]={face:summarize(subset,face) for face in INTERFACES}
            paired[f"{arm}_{form}"]={role:paired_counts(subset,
                "direct_actor" if role=="actor" else "direct_patient",role)
                for role in ("actor","patient","both")}
        sensitivity_results[arm]={face:sensitivity(arm_rows,face) for face in INTERFACES}
    for world in protocol["worlds"]:
        for seed in (0,)+tuple(protocol["seeds"]):
            arms=("pretrained",) if seed==0 else ARMS[1:]
            for arm in arms:
                key=f"{world}_{seed}_{arm}"
                subset=[r for r in rows if (r["world"],r["seed"],r["arm"])==(world,seed,arm)]
                if len(subset)!=96: raise ValueError(key)
                units[key]={form:{face:summarize([r for r in subset if r["form"]==form],face)
                                  for face in INTERFACES} for form in FORMS}
                events=defaultdict(dict)
                for r in subset:
                    events[r["event"]][r["form"]]=r["scores"]["extraction"]["map"]
                if len(events)!=32 or any(len(v)!=3 for v in events.values()):
                    raise ValueError("Event coverage")
                code={"first_actor_last_patient":"F","last_actor_first_patient":"L",
                      "invalid":"X","both_first":"D","both_last":"D","other_valid":"O"}
                vectors=Counter("".join(code[ev[f]] for f in FORMS) for ev in events.values())
                event_vectors[key]={"n_events":32,"vectors_active_passive_cleft":dict(vectors),
                                    "all_semantic_FLL":vectors["FLL"],
                                    "all_last_position_LLL":vectors["LLL"]}
    for world in protocol["worlds"]:
        base={r["id"]:r for r in rows if r["world"]==world and r["arm"]=="pretrained"}
        if len(base)!=96: raise ValueError("Base coverage")
        for seed in protocol["seeds"]:
            for arm in ARMS[1:]:
                for form in FORMS:
                    subset=[r for r in rows if (r["world"],r["seed"],r["arm"],r["form"])==(world,seed,arm,form)]
                    changes[f"{world}_{seed}_{arm}_{form}"]={
                        "archived_actor":base_change(subset,base,"archived","actor"),
                        "direct_actor":base_change(subset,base,"direct_actor","actor"),
                        "direct_patient":base_change(subset,base,"direct_patient","patient"),
                        "extraction_actor":base_change(subset,base,"extraction","actor"),
                        "extraction_patient":base_change(subset,base,"extraction","patient"),
                        "extraction_both":base_change(subset,base,"extraction","both")}
    report={"status":"read_only_shortcut_audit","source_hashes":hashes,
            "first_last_predictions":{"active":{"first_actor":True,"last_actor":False,"event_subject_actor":True},
                                      "passive":{"first_actor":False,"last_actor":True,"event_subject_actor":False},
                                      "patient_cleft":{"first_actor":False,"last_actor":True,"event_subject_actor":True}},
            "subject_definition":"Event-clause lexical subject: actor in active and patient-cleft; patient in passive. Cleft matrix expletive It is excluded.",
            "aggregate":aggregate,"per_unit":units,"event_vectors":event_vectors,
            "paired_interface":paired,"adapter_vs_world_base":changes,
            "malformed_sensitivity":sensitivity_results,
            "direct_patient_question":protocol["patient_question"]}
    OUT.write_text(json.dumps(report,indent=2)+"\n")
    print(f"Saved {OUT}; rows={len(rows)}; units={len(units)}")


if __name__=="__main__":
    main()
