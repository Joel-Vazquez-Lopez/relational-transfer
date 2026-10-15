import unittest
from heldout_rules import evaluate
class TestHeldout(unittest.TestCase):
 def cases(self):
  return [dict(group_id=g,event_id=f'{g}{q}',participant_first='A',participant_last='B',grammatical_subject='A',question_role=q,observed_answer='A' if q=='actor' else 'B') for g in ('g1','g2','g3') for q in ('actor','patient')]
 def test_grouping(self):
  r=evaluate(self.cases()); self.assertEqual(r['groups'],3); self.assertEqual(r['unanimous_correct'],6)
 def test_gold_label_noninterference(self):
  data=self.cases(); a=evaluate(data); [r.update(gold_actor='B',gold_patient='A') for r in data]; self.assertEqual(a,evaluate(data))
 def test_single_group_rejected(self):
  with self.assertRaises(ValueError): evaluate(self.cases()[:2])
 def test_unexpected_response_rejected(self):
  x=self.cases();x[0]['observed_answer']='OUT_OF_SET'
  with self.assertRaises(ValueError):evaluate(x)
if __name__=='__main__':unittest.main()
