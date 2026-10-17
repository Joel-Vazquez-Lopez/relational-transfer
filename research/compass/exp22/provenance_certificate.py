#!/usr/bin/env python3
"""Compass 22: finite symbolic, model-free event-source identifiability certificate.

No new LLM outputs, prompts, licensed text, model training, or gold labels.
All referents are symbolic, distinct and unambiguous by assumption.
"""
from dataclasses import dataclass
from itertools import permutations
import json

@dataclass(frozen=True)
class Case:
    order: tuple[str, ...]
    target: str
    named_competitor: str | None = None

    def validate(self):
        assert len(self.order) == len(set(self.order))
        assert self.target in self.order
        if self.named_competitor is not None:
            assert self.named_competitor in self.order
            assert self.named_competitor != self.target


def first_fallback(case: Case):
    case.validate()
    return case.order[0]


def other_event(case: Case):
    """Only defined when there are exactly two events."""
    case.validate()
    assert len(case.order) == 2
    return next(e for e in case.order if e != case.target)


def immediate_predecessor(case: Case):
    """Only use cases with target after the first event."""
    case.validate()
    index = case.order.index(case.target)
    assert index > 0
    return case.order[index - 1]


def designated_competitor(case: Case):
    case.validate()
    assert case.named_competitor is not None
    return case.named_competitor


def gold(case: Case):
    case.validate()
    return case.target


def two_event_table():
    rows=[]
    for order in permutations(('E1', 'E2')):
        for target in order:
            c=Case(order,target)
            rows.append(dict(order=list(order),target=target,gold=gold(c),
                             first_fallback=first_fallback(c),other_event=other_event(c),
                             rivals_disagree=first_fallback(c)!=other_event(c)))
    assert len(rows)==4
    assert sum(r['rivals_disagree'] for r in rows)==2
    assert not any(r['rivals_disagree'] for r in rows if r['target']==r['order'][1])
    return rows


def three_event_table():
    """Three events make first-event and immediately previous source separable."""
    rows=[]
    for order in permutations(('E1','E2','E3')):
        for pos in (1,2):
            target=order[pos]
            c=Case(order,target)
            rows.append(dict(order=list(order),target=target,target_index=pos,
                             first_fallback=first_fallback(c),
                             immediate_predecessor=immediate_predecessor(c),
                             rivals_disagree=first_fallback(c)!=immediate_predecessor(c)))
    assert len(rows)==12
    assert all(r['rivals_disagree']==(r['target_index']==2) for r in rows)
    return rows


def conditional_provenance(a: tuple[str,...], target: int, predicted: str):
    """Label only for unique candidate referents; ambiguous aliases rejected."""
    assert len(set(a)) == len(a)
    assert 0 <= target < len(a)
    if predicted == a[target]:
        return 'TARGET'
    matches=[i for i,x in enumerate(a) if x==predicted]
    if not matches:
        return 'OUTSIDE_ROLE_CANDIDATES'
    return 'OTHER_EVENT_'+str(matches[0]+1)


def audit():
    assert conditional_provenance(('A','B'),1,'A')=='OTHER_EVENT_1'
    assert conditional_provenance(('A','B'),1,'B')=='TARGET'
    assert conditional_provenance(('A','B'),1,'C')=='OUTSIDE_ROLE_CANDIDATES'
    # For any two distinct possible source answers, an incorrect answer in that
    # candidate set is uniquely and necessarily the other event's answer.
    for ordered in permutations(('A','B')):
        for target in (0,1):
            for response in ordered:
                tag=conditional_provenance(ordered,target,response)
                assert (tag.startswith('OTHER_EVENT_')) == (response != ordered[target])
    two=two_event_table();three=three_event_table()
    return {
        'source':'COMPASS_22_MODEL_FREE_CERTIFICATE',
        'observed_local_evidence_used':'none; old reported seven errors are not replayed',
        'two_event_predictions':two,
        'three_event_predictions':three,
        'two_event_disagree_count':sum(r['rivals_disagree'] for r in two),
        'two_event_second_target_disagree_count':sum(r['rivals_disagree'] for r in two if r['target']==r['order'][1]),
        'three_event_late_target_first_vs_predecessor_disagree_count':sum(r['rivals_disagree'] for r in three if r['target_index']==2),
        'formal_bound':'For k=2 unique role fillers, conditional on a wrong response belonging to their two-element set, its source is determined exactly by target and error indicator.',
        'caveat':'Three-event agreement/disagreement is between restricted symbolic fallback policies, not a distinguisher of arbitrary native neural mechanisms or competent syntax/content copy algorithms.'
    }

if __name__=='__main__':
    print(json.dumps(audit(),indent=2))
