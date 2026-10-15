import unittest
from cross_family_rule_audit import analyze
class Tests(unittest.TestCase):
 def test_single_rule_ceiling_and_oracle(self):
  cells={f"cell_{i}":{"n":2,"agreement":{"first":2 if i%2==0 else 0,"last":0 if i%2==0 else 2}} for i in range(8)}
  v=analyze({"models":{"toy":{"heuristic_prediction_agreement":cells}}})["toy"]
  self.assertEqual(v["single_rule_in_sample_best"],8)
  self.assertEqual(v["cellwise_oracle_in_sample"],16)
 def test_no_label_fields_needed(self):
  cells={f"c{i}":{"n":2,"agreement":{"first":1,"last":1}} for i in range(8)}
  r=analyze({"models":{"x":{"heuristic_prediction_agreement":cells}}})
  self.assertEqual(r["x"]["best_rules"],["first","last"])
if __name__=="__main__": unittest.main()
