"""Offline, dependency-free relational interference fixture gate.

Synthetic toy events only; this does NOT establish matched natural-language
semantics, source-text rights, human gold, or model behavior.
"""
from dataclasses import dataclass
import unittest

@dataclass(frozen=True)
class Case:
    condition: str
    text: str
    target_verb: str
    answer: str
    target_verb_occurrence: int = 0

def token_span(tokens, sequence, occurrence=0):
    needle = sequence.split()
    matches = [i for i in range(len(tokens)-len(needle)+1)
               if tokens[i:i+len(needle)] == needle]
    if occurrence < 0 or occurrence >= len(matches):
        raise ValueError("requested token occurrence not found")
    start = matches[occurrence]
    return start, start + len(needle)

def verify(case):
    tokens = case.text.split()
    vstart, vend = token_span(tokens, case.target_verb,
                             case.target_verb_occurrence)
    astart, aend = token_span(tokens, case.answer)
    assert vend > vstart and aend > astart
    assert " ".join(tokens[astart:aend]) == case.answer
    return {"condition": case.condition, "predicate_idx": vstart,
            "answer_range": [astart, aend], "target": case.target_verb}

# Invented fixtures, not QA-SRL examples. The target is always the first
# 'praised' event except C5, where it is the second 'praised' event.
CASES = [
    Case("C0", "Ada praised Bea", "praised", "Bea"),
    Case("C1", "Ada praised Bea and Cy greeted Dee", "praised", "Bea"),
    Case("C2", "Ada praised Bea and Cy praised Dee", "praised", "Bea"),
    Case("C3", "Ada praised Bea and Cy greeted Ada", "praised", "Bea"),
    Case("C4", "Ada praised Bea and Cy praised Ada", "praised", "Bea"),
    Case("C5", "Cy praised Ada and Ada praised Bea", "praised", "Bea", 1),
]

class TestFixtures(unittest.TestCase):
    def test_all_conditions_have_valid_target_and_answer_spans(self):
        self.assertEqual(len({c.condition for c in CASES}), 6)
        for case in CASES:
            with self.subTest(condition=case.condition):
                result = verify(case)
                self.assertEqual(
                    " ".join(case.text.split()[slice(*result["answer_range"])]),
                    "Bea")

    def test_order_reversal_reindexes_target(self):
        self.assertEqual(verify(CASES[4])["predicate_idx"], 1)
        self.assertEqual(verify(CASES[5])["predicate_idx"], 5)

    def test_missing_answer_rejected(self):
        with self.assertRaises(ValueError):
            verify(Case("bad", "Ada praised Bea", "praised", "Zoe"))

    def test_invalid_predicate_occurrence_rejected(self):
        with self.assertRaises(ValueError):
            verify(Case("bad", "Ada praised Bea", "praised", "Bea", 2))

    def test_duplicate_answer_surface_requires_occurrence_disambiguation(self):
        tokens = "Ada praised Bea and Bea praised Ada".split()
        self.assertEqual(token_span(tokens, "Bea", 0), (2, 3))
        self.assertEqual(token_span(tokens, "Bea", 1), (4, 5))

if __name__ == "__main__":
    unittest.main()
