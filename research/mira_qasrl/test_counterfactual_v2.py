"""Independent structural counterfactual admissibility checks (no semantic oracle)."""
from dataclasses import dataclass
from collections import Counter
import unittest

@dataclass(frozen=True)
class Event:
    event_id: str
    start: int
    end: int
    predicate_start: int
    predicate_end: int
    roles: tuple[tuple[str, str], ...]

@dataclass(frozen=True)
class Case:
    tokens: tuple[str, ...]
    events: tuple[Event, ...]
    target_id: str
    answer_role: str
    source_id: str
    reviewer_ids: tuple[str, ...] = ()
    review_status: str = "PENDING"

@dataclass(frozen=True)
class Comparison:
    left: Case
    right: Case
    intended_change: str  # ORDER or DISTRACTOR


def validate(c):
    errs=[]
    n=len(c.tokens)
    if not c.source_id: errs.append('SOURCE_ID_MISSING')
    if c.review_status != 'APPROVED' or len(set(c.reviewer_ids)) < 2:
        errs.append('INDEPENDENT_SEMANTIC_REVIEW_MISSING')
    ids=[e.event_id for e in c.events]
    if len(ids)!=len(set(ids)): errs.append('DUPLICATE_EVENT_ID')
    if c.target_id not in ids: errs.append('TARGET_EVENT_MISSING')
    for e in c.events:
        if not 0 <= e.start < e.end <= n: errs.append('EVENT_SPAN_INVALID'); continue
        if not e.start <= e.predicate_start < e.predicate_end <= e.end:
            errs.append('PREDICATE_OUTSIDE_EVENT')
        roles=dict(e.roles)
        if len(roles)!=len(e.roles): errs.append('DUPLICATE_ROLE')
        if e.event_id == c.target_id and c.answer_role not in roles:
            errs.append('TARGET_ROLE_MISSING')
    return sorted(set(errs))


def compare(p):
    a,b=p.left,p.right
    errs=validate(a)+validate(b)
    if a.source_id!=b.source_id: errs.append('SOURCE_CHANGED')
    if a.target_id!=b.target_id: errs.append('TARGET_CHANGED')
    if a.answer_role!=b.answer_role: errs.append('QUESTION_ROLE_CHANGED')
    am={e.event_id:e for e in a.events}; bm={e.event_id:e for e in b.events}
    if set(am)!=set(bm): errs.append('EVENT_ID_SET_CHANGED')
    for eid in set(am)&set(bm):
        if dict(am[eid].roles)!=dict(bm[eid].roles): errs.append('ROLE_BINDING_CHANGED')
    if p.intended_change=='ORDER':
        if Counter(a.tokens)!=Counter(b.tokens): errs.append('TOKEN_MULTISET_CHANGED')
        if set(am)==set(bm):
            order_a=[e.event_id for e in sorted(a.events,key=lambda e:e.start)]
            order_b=[e.event_id for e in sorted(b.events,key=lambda e:e.start)]
            if order_a==order_b: errs.append('EVENT_ORDER_NOT_CHANGED')
            for eid in am:
                ea,eb=am[eid],bm[eid]
                if a.tokens[ea.start:ea.end]!=b.tokens[eb.start:eb.end]:
                    errs.append('EVENT_TEXT_CHANGED')
    elif p.intended_change=='DISTRACTOR':
        for eid in set(am)&set(bm):
            ea,eb=am[eid],bm[eid]
            if a.tokens[ea.start:ea.end]!=b.tokens[eb.start:eb.end]:
                errs.append('EVENT_TEXT_CHANGED')
        if len(a.tokens)==len(b.tokens):
            errs.append('NO_TOKEN_LOAD_CHANGE')
    else: errs.append('UNKNOWN_INTERVENTION')
    return sorted(set(errs))

class TestGate(unittest.TestCase):
    def base(self, order=False, review='APPROVED', reviewers=('r1','r2')):
        clauses=[('e1',('Ada','praised','Bea'),(('agent','Ada'),('patient','Bea'))),
                 ('e2',('Cy','praised','Dee'),(('agent','Cy'),('patient','Dee')))]
        if order: clauses.reverse()
        tokens=[]; events=[]
        for eid,words,roles in clauses:
            s=len(tokens); tokens.extend(words)
            events.append(Event(eid,s,len(tokens),s+1,s+2,roles))
        return Case(tuple(tokens),tuple(events),'e1','agent','fictional',reviewers,review)
    def test_order_swap(self):
        self.assertEqual(compare(Comparison(self.base(),self.base(True),'ORDER')),[])
    def test_unreviewed_rejected(self):
        self.assertIn('INDEPENDENT_SEMANTIC_REVIEW_MISSING',validate(self.base(review='PENDING')))
    def test_one_reviewer_rejected(self):
        self.assertIn('INDEPENDENT_SEMANTIC_REVIEW_MISSING',validate(self.base(reviewers=('r1',))))
    def test_unmodified_order_rejected(self):
        self.assertIn('EVENT_ORDER_NOT_CHANGED',compare(Comparison(self.base(),self.base(),'ORDER')))
    def test_changed_role_rejected(self):
        b=self.base(True); es=list(b.events); e=es[0]
        es[0]=Event(e.event_id,e.start,e.end,e.predicate_start,e.predicate_end,(('agent','Dee'),('patient','Cy')))
        b=Case(b.tokens,tuple(es),b.target_id,b.answer_role,b.source_id,b.reviewer_ids,b.review_status)
        self.assertIn('ROLE_BINDING_CHANGED',compare(Comparison(self.base(),b,'ORDER')))
    def test_word_scramble_rejected(self):
        b=self.base(True); toks=list(b.tokens); toks[0],toks[1]=toks[1],toks[0]
        b=Case(tuple(toks),b.events,b.target_id,b.answer_role,b.source_id,b.reviewer_ids,b.review_status)
        self.assertIn('EVENT_TEXT_CHANGED',compare(Comparison(self.base(),b,'ORDER')))
    def test_unknown_change_rejected(self):
        self.assertIn('UNKNOWN_INTERVENTION',compare(Comparison(self.base(),self.base(),'OTHER')))
    def test_distractor_append(self):
        a=self.base(); b=Case(a.tokens+('Meanwhile','Nina','waited'),a.events,a.target_id,a.answer_role,a.source_id,a.reviewer_ids,a.review_status)
        self.assertEqual(compare(Comparison(a,b,'DISTRACTOR')),[])
    def test_distractor_same_length_rejected(self):
        a=self.base()
        self.assertIn('NO_TOKEN_LOAD_CHANGE',compare(Comparison(a,a,'DISTRACTOR')))

if __name__=='__main__': unittest.main()
