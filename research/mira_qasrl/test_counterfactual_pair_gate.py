"""Mira counterfactual fixture gate. Structural checks ONLY, never semantic proof."""
from dataclasses import dataclass
from collections import Counter
import unittest

@dataclass(frozen=True)
class Trial:
    tokens: tuple[str, ...]
    target_span: tuple[int, int]
    answer_span: tuple[int, int]
    event_spans: tuple[tuple[int, int], ...]
    target_event: int
    semantic_review: str = "PENDING"
    source_id: str = ""

def validate_trial(t):
    errors = []
    n = len(t.tokens)
    def valid(s):
        return len(s) == 2 and 0 <= s[0] < s[1] <= n
    if not valid(t.target_span): errors.append("TARGET_SPAN_INVALID")
    if not valid(t.answer_span): errors.append("ANSWER_SPAN_INVALID")
    if not t.event_spans or any(not valid(s) for s in t.event_spans):
        errors.append("EVENT_SPANS_INVALID")
    if not 0 <= t.target_event < len(t.event_spans):
        errors.append("TARGET_EVENT_INVALID")
    elif valid(t.target_span) and valid(t.event_spans[t.target_event]):
        a,b=t.event_spans[t.target_event]
        if not (a <= t.target_span[0] and t.target_span[1] <= b):
            errors.append("TARGET_OUTSIDE_EVENT")
    if t.semantic_review != "APPROVED": errors.append("SEMANTICS_NOT_APPROVED")
    if not t.source_id: errors.append("SOURCE_ID_MISSING")
    return sorted(set(errors))

def validate_pair(low, high):
    errors = validate_trial(low) + validate_trial(high)
    if low.tokens[slice(*low.target_span)] != high.tokens[slice(*high.target_span)]:
        errors.append("TARGET_SURFACE_CHANGED")
    if low.tokens[slice(*low.answer_span)] != high.tokens[slice(*high.answer_span)]:
        errors.append("ANSWER_SURFACE_CHANGED")
    if len(low.tokens) != len(high.tokens): errors.append("TOKEN_LENGTH_UNMATCHED")
    if len(low.event_spans) != len(high.event_spans): errors.append("EVENT_COUNT_UNMATCHED")
    if low.source_id != high.source_id: errors.append("SOURCE_CHANGED")
    if Counter(low.tokens) != Counter(high.tokens): errors.append("LEXICAL_MULTISET_UNMATCHED")
    for t in (low, high):
        if not (0 <= t.answer_span[0] < t.answer_span[1] <= len(t.tokens)): continue
        ans=t.tokens[slice(*t.answer_span)]
        hits=sum(t.tokens[i:i+len(ans)] == ans for i in range(len(t.tokens)-len(ans)+1))
        if hits > 1: errors.append("ANSWER_OCCURRENCE_AMBIGUOUS")
    return sorted(set(errors))

class TestCounterfactualValidator(unittest.TestCase):
    def trial(self, tokens=("Ada","praised","Bea","and","Cy","praised","Dee"),
              target=(1,2), answer=(0,1), event=0, review="APPROVED", source="fictional"):
        return Trial(tuple(tokens),target,answer,((0,3),(4,7)),event,review,source)
    def test_self_pair(self):
        t=self.trial()
        self.assertEqual(validate_pair(t,t),[])
    def test_pending_review_fails(self):
        self.assertIn("SEMANTICS_NOT_APPROVED",validate_trial(self.trial(review="PENDING")))
    def test_wrong_event_fails(self):
        self.assertIn("TARGET_OUTSIDE_EVENT",validate_trial(self.trial(event=1)))
    def test_repeated_answer_fails(self):
        t=self.trial(tokens=("Ada","praised","Bea","and","Cy","praised","Ada"))
        self.assertIn("ANSWER_OCCURRENCE_AMBIGUOUS",validate_pair(t,t))
    def test_length_fails(self):
        t=self.trial()
        u=self.trial(tokens=t.tokens+("today",))
        self.assertIn("TOKEN_LENGTH_UNMATCHED",validate_pair(t,u))
    def test_lexical_fails(self):
        t=self.trial()
        u=self.trial(tokens=("Ada","praised","Bea","and","Cy","greeted","Dee"))
        self.assertIn("LEXICAL_MULTISET_UNMATCHED",validate_pair(t,u))
    def test_source_fails(self):
        self.assertIn("SOURCE_CHANGED",validate_pair(self.trial(),self.trial(source="other")))
    def test_target_surface_fails(self):
        t=self.trial()
        u=self.trial(tokens=("Ada","greeted","Bea","and","Cy","praised","Dee"))
        self.assertIn("TARGET_SURFACE_CHANGED",validate_pair(t,u))

if __name__ == "__main__":
    unittest.main()
