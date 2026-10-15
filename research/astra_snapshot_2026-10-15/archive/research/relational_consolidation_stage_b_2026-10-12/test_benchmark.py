#!/usr/bin/env python3
"""Structural and semantic checks for the Stage B generator."""
import json
import unittest
from pathlib import Path

import benchmark as b


CFG = json.loads((Path(__file__).resolve().parent / "PROTOCOL.json").read_text())


class BenchmarkTests(unittest.TestCase):
    def test_reproducible_world(self):
        self.assertEqual(b.stable_json(b.family_data(CFG, 2)), b.stable_json(b.family_data(CFG, 2)))
        self.assertNotEqual(b.family_data(CFG, 2)["graph_edges"], b.family_data(CFG, 3)["graph_edges"])

    def test_disjoint_pairs_and_fixed_degree(self):
        for i in range(CFG["families"]):
            f = b.family_data(CFG, i)
            a = set(map(tuple, f["graph_edges"]))
            d = set(map(tuple, f["development_pairs"]))
            t = set(map(tuple, f["confirmation_pairs"]))
            self.assertFalse(a & d or a & t or d & t)
            self.assertEqual(len(a), 54)
            self.assertEqual(len(d), 15)
            self.assertEqual(len(t), 15)
            self.assertIn((0, 1), t)
            self.assertEqual(set(dict.fromkeys(CFG["verbs"])), set(CFG["verbs"]))

    def test_semantic_role_maps(self):
        f = b.family_data(CFG, 0)
        for ev in f["train_events"] + f["development_events"] + f["confirmation_events"]:
            rows = [b.render(ev, form) for form in b.FORMS]
            for r in rows[:4]:
                self.assertEqual((r["gold_actor_id"], r["gold_patient_id"]), (ev["actor_id"], ev["patient_id"]))
            self.assertEqual((rows[4]["gold_actor_id"], rows[4]["gold_patient_id"]), (ev["patient_id"], ev["actor_id"]))
            self.assertEqual(set(rows[0]["mention_order"]), {ev["actor_id"], ev["patient_id"]})

    def test_s2_uses_only_training_events(self):
        f = b.family_data(CFG, 0)
        rows = b.s2_pairs_from_train(f)
        self.assertEqual(len(rows), 4 * len(f["train_events"]))
        self.assertEqual({x["source_event_id"] for x in rows}, {x["event_id"] for x in f["train_events"]})
        self.assertFalse({tuple(x["source_unordered_pair"]) for x in rows} & set(map(tuple, f["confirmation_pairs"])))
        contaminated = dict(f)
        contaminated["train_events"] = f["train_events"] + [f["confirmation_events"][0]]
        with self.assertRaises(ValueError):
            b.s2_pairs_from_train(contaminated)

    def test_all_families_pass_structural_audit(self):
        report = b.audit([b.family_data(CFG, i) for i in range(CFG["families"])], CFG)
        self.assertEqual(report["gates"]["split_and_semantics_checks"], "pass")
        self.assertEqual(report["gates"]["simple_heuristic_actor_gate"], "pass")
        self.assertEqual(report["gates"]["simple_heuristic_joint_gate"], "pass")
        self.assertEqual(report["heuristics"]["construction_aware_oracle"]["joint_event_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
