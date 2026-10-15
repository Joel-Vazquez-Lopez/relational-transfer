#!/usr/bin/env python3
"""Read-only integrity audit for explicitly authorized synthetic research archives.

Does not extract, execute ZIP members, load PyTorch pickle, or call any LLM/API.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

EXPECTED = {
    "mira_v4": "50451ea957fb5e81a03ea7b637e34fd976906e9968186683f5118cd4c66b58d7",
    "brain_e038": "14a6cdaa2b2c7403b00fb6fadb46eb553d1c25abb5985a6d6a59f865b8b7cf86",
    "nova_cal": "dceb5086e27cf87c855e7f738407dabcff815c60a94f63ad50f7028373e6f17c",
}
NOVA_SOURCE_SHA = {
    "model-f580e8708a63": "20511d7cc8210c8d4eb3d31fb76636a08f0b5dd3e188960010819081b627054a",
    "model-a6c42d028e1e": "104a6b28655445c3bac59c3c7b22027b8a4c2e6c8870d2e0c3ed9b6d65a0df2f",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def jsonl(z, member):
    return [json.loads(line) for line in z.read(member).splitlines() if line.strip()]


def audit_archive(path, key):
    data = Path(path).read_bytes()
    actual = sha256(data)
    if actual != EXPECTED[key]:
        raise ValueError(f"{key}: archive SHA mismatch: {actual}")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"{key}: duplicate archive member path")
        if any(n.startswith("/") or ".." in Path(n).parts for n in names):
            raise ValueError(f"{key}: unsafe archive path")
        bad = z.testzip()
        if bad:
            raise ValueError(f"{key}: ZIP CRC failure on {bad}")
        if key == "mira_v4":
            root = "MIRA_V3_ATTACK_AND_V4/"
            internal_manifest = json.loads(z.read(root + "SHA256.json"))
            for rel, digest in internal_manifest.items():
                if sha256(z.read(root + rel)) != digest:
                    raise ValueError(f"mira_v4: member SHA mismatch: {rel}")
            inp = jsonl(z, root + "model_inputs_v4.jsonl")
            gold = jsonl(z, root + "gold_v4.jsonl")
            if len(inp) != 2592 or len(gold) != 2592:
                raise ValueError("mira_v4: unexpected question count")
            input_ids = [r["id"] for r in inp]
            gold_ids = [r["id"] for r in gold]
            if len(set(input_ids)) != len(inp) or set(input_ids) != set(gold_ids):
                raise ValueError("mira_v4: duplicate/unmatched row IDs")
            if not all(isinstance(r["passage"], str) and isinstance(r["question"], str) for r in inp):
                raise ValueError("mira_v4: missing text prompts")
            families = {r["family_id"] for r in gold}
            if len(families) != 60:
                raise ValueError(f"mira_v4: unexpected family count {len(families)}")
            print("MIRA: 2592 input IDs, 2592 gold IDs, 60 families; internal member SHA and archive verified.")
            print("NOTE: model prompts must contain ONLY passage and question; never id, split, gold or generator_edges.")
        elif key == "brain_e038":
            root = "BRAIN_O_E037_038/"
            record = json.loads(z.read(root + "MANIFEST_SHA256.json"))
            for rel, digest in record["files"].items():
                if sha256(z.read(root + rel)) != digest:
                    raise ValueError(f"brain_e038: member SHA mismatch: {rel}")
            print(f"BRAIN-O: {len(record['files'])} manifest hashes verified; checkpoints were NOT loaded.")
        else:
            records = {}
            for label, source_hash in NOVA_SOURCE_SHA.items():
                member = f"nova004_cal01/{label}/predictions.jsonl"
                raw = z.read(member)
                if sha256(raw) != source_hash:
                    raise ValueError(f"nova_cal: source predictions hash mismatch for {label}")
                rows = jsonl(z, member)
                ids = [r["id"] for r in rows]
                if len(rows) != 24 or len(set(ids)) != 24 or not all(isinstance(r["raw"],str) for r in rows):
                    raise ValueError(f"nova_cal: invalid row count/format for {label}")
                records[label] = set(ids)
            if len(set(map(frozenset, records.values()))) != 1:
                raise ValueError("nova_cal: model prediction ID sets differ")
            print("NOVA: raw SHA verified; 24+24 matching output IDs; no model rerun.")
    print(f"PASS {key}: {actual}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mira-v4")
    p.add_argument("--brain-e038")
    p.add_argument("--nova-cal")
    a = p.parse_args()
    selected = [(k, v) for k, v in vars(a).items() if v]
    if not selected:
        p.error("Provide at least one --mira-v4 / --brain-e038 / --nova-cal path.")
    try:
        for kind, path in selected:
            audit_archive(path, kind)
    except (KeyError, OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
