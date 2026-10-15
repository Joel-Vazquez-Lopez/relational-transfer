"""RELATE-001 deterministic hypothesis-signature tests, not learned-model tests."""
import re
import unittest
from collections import Counter
from context_binding_challenge import generate, summarize, KINDS, NAMES, HYPOTHESES

class SameQueryControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=generate()

    def test_complete_coverage(self):
        self.assertEqual(len(self.rows),768)
        self.assertEqual(Counter(x["intervention"] for x in self.rows),
                         {k:192 for k in KINDS})
        self.assertEqual(len({x["family"] for x in self.rows}),24)

    def test_participants_once_and_fixed_query_fields(self):
        for row in self.rows:
            for name in NAMES:
                self.assertEqual(row["recipient"].count(name),1)
                self.assertEqual(row["donor"].count(name),1)
            self.assertIn(row["query_verb"],("praised","criticized"))
            self.assertIn(row["query_role"],("actor","patient"))

    def test_semantic_and_slot_counterfactuals(self):
        for row in self.rows:
            kind=row["intervention"]
            self.assertEqual(row["recipient_gold"]==row["donor_gold"],
                             kind in ("event_order_swap","voice_flip"))
            self.assertEqual(row["recipient_slot"]==row["donor_slot"],
                             kind in ("participant_swap","voice_flip"))

    def test_lexical_and_punctuation_bags_not_bpe_token_bags(self):
        for row in self.rows:
            if row["intervention"]!="voice_flip":
                words=lambda text: sorted(re.findall(r"[A-Za-z]+|[^\w\s]",text))
                self.assertEqual(words(row["recipient"]),words(row["donor"]))

    def test_signatures_identify_three_idealized_hypotheses(self):
        sig=summarize(self.rows)
        def count(kind,h):return sig[kind][h].get("change",0)
        for kind,counts in {
            "predicate_swap":(0,192,192),
            "participant_swap":(0,0,192),
            "event_order_swap":(0,192,0),
            "voice_flip":(0,0,0)
        }.items():
            self.assertEqual(tuple(count(kind,h) for h in HYPOTHESES),counts)

if __name__=="__main__":unittest.main()
