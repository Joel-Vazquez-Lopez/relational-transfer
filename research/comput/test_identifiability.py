import unittest
from identifiability import rules, audit

def rows(subjects):
    return [{"event_id": f"{s}:{q}", "participant_first":"A", "participant_last":"B",
             "grammatical_subject":s, "question_role":q}
            for s in subjects for q in ("actor", "patient")]

class TestIdentifiability(unittest.TestCase):
    def test_confounded_design(self):
        classes = audit(rows(["A"]))["equivalence_classes"]
        self.assertIn(["first", "subject"], classes)
        self.assertIn(["position_by_question", "subject_by_question"], classes)
    def test_counterbalanced_design(self):
        result = audit(rows(["A", "B"]))
        self.assertEqual(result["identifiable_classes"], 7)
        self.assertEqual(len(result["greedy_discriminating_row_indices"]), 3)
    def test_no_gold_leakage(self):
        r = rows(["B"])[0]
        self.assertEqual(rules(r), rules(dict(r, gold_actor="B", gold_patient="A")))
    def test_selected_rows_separate_all_distinguishable_pairs(self):
        original = rows(["A", "B"])
        subset = [original[i] for i in audit(original)["greedy_discriminating_row_indices"]]
        self.assertEqual(audit(original)["identifiable_classes"], audit(subset)["identifiable_classes"])

if __name__ == "__main__":
    unittest.main()
