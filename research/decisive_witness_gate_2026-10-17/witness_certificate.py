"""Deterministic pre-inference identifiability check; no model or corpus access."""

import json
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent


def predictions():
    # Donor: E1 occurs first and bears marker A. Its question-label state is patched
    # into recipients with independent order, marker assignment, and role.
    rows = []
    for first_event, marker_a_event, role in product(("E1", "E2"), ("E1", "E2"), ("Agent", "Patient")):
        row = {
            "recipient_first_event": first_event,
            "recipient_marker_A_event": marker_a_event,
            "role": role,
            "semantic_event_identity": f"E1.{role}",
            "ordinal_index": f"{first_event}.{role}",
            "strong_compound_key": f"E1.{role}",
        }
        rows.append(row)
    return rows


def main():
    rows = predictions()
    assert len(rows) == 8
    assert any(r["semantic_event_identity"] != r["ordinal_index"] for r in rows)
    assert all(r["semantic_event_identity"] == r["strong_compound_key"] for r in rows)
    out = {
        "status": "NO_DISCRIMINATING_WITNESS_BEFORE_MODEL_INFERENCE",
        "donor": "E1 first, A attached to E1; fixed question-label patch",
        "rival_definitions": {
            "semantic_event_identity": "The donor state selects the same source-defined E1 event identity in the recipient, then retrieves the queried role.",
            "ordinal_index": "The donor state selects the first recipient occurrence, then retrieves the queried role.",
            "strong_compound_key": "The donor state conveys an E1 syntax/content key; a query-aware recipient parser matches that source record after order and marker changes, then copies the queried role.",
        },
        "rows": rows,
        "certificate": "EI and OI disagree after order reversal, but the competent compound-key copier and EI agree in every order × marker × role cell. Identity, adjacent-token, sham, and cross-document controls cannot break this equivalence without independent restrictions on the copier's accessible key or native route. No unique event-selection mechanism is identifiable from the proposed answer outcomes.",
    }
    path = HERE / "PREDICTIONS.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{len(rows)} factorial cells; EI/OI disagree in {sum(r['semantic_event_identity'] != r['ordinal_index'] for r in rows)}; EI/CK disagree in {sum(r['semantic_event_identity'] != r['strong_compound_key'] for r in rows)}")


if __name__ == "__main__":
    main()
