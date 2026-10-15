import unittest
from rule_transfer_audit import audit
class TestTransfer(unittest.TestCase):
 def test_audit(self):
  cells={}
  for form in ("active","passive","object_cleft","subject_cleft"):
   for w in ("lexical","explicit"):
    cells[form+"/"+w]={"n":2,"agreement":{"always_first":2,"always_last":0,"first_as_actor":1,"last_as_actor":1,"always_actor":2}}
  d=audit({"models":{"toy":{"heuristic_prediction_agreement":cells}}})["toy"]
  self.assertEqual(d["transfer"]["wording"]["min_agreement"],16)
  self.assertEqual(d["transfer"]["construction"]["min_agreement"],16)
  self.assertEqual(d["overall_in_sample"]["always_first"],16)
  self.assertNotIn("always_actor",d["overall_in_sample"])
 def test_tied_winners(self):
  cells={form+"/"+w:{"n":2,"agreement":{"always_first":2 if form=="active" else 0,"always_last":0 if form=="active" else 2,"first_as_actor":1,"last_as_actor":1}} for form in ("active","passive","object_cleft","subject_cleft") for w in ("lexical","explicit")}
  d=audit({"models":{"toy":{"heuristic_prediction_agreement":cells}}})["toy"]
  self.assertLessEqual(d["transfer"]["construction"]["min_agreement"],d["transfer"]["construction"]["max_agreement"])
if __name__=="__main__": unittest.main()
