import unittest
from comput009 import generate,result,PREDICTORS,VARIANTS,FORMS,construction,extract_mentions
from selection import run
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.rows=generate();cls.summary=result(cls.rows);cls.stages=run(cls.rows)["stages"]
 def test_size(self):
  self.assertEqual(len(self.rows),2304)
  self.assertEqual(self.summary["n_source_events"],192)
 def test_cell_balance(self):
  for variant in VARIANTS:
   subset=[r for r in self.rows if r["variant"]==variant]
   self.assertEqual(len(subset),384)
   for f in FORMS:self.assertEqual(sum(r["form"]==f for r in subset),96)
 def test_global(self):
  for v in VARIANTS:
   expected=192 if v.startswith(("as_for","regarding")) else 384
   self.assertEqual(self.summary["per_variant"][v]["global_order"]["correct"],expected)
 def test_comma(self):
  for v in VARIANTS:
   expected=0 if v=="parenthetical" else 384
   self.assertEqual(self.summary["per_variant"][v]["after_first_comma"]["correct"],expected)
 def test_grammar(self):
  for v in VARIANTS:self.assertEqual(self.summary["per_variant"][v]["fixed_surface_grammar"]["correct"],384)
 def test_gold_independence(self):
  for r in self.rows:
   modified={**r,"gold":"SPOOF"}
   for func in PREDICTORS.values():
    self.assertEqual(func(r["sentence"],r["question"]),func(modified["sentence"],modified["question"]))
 def test_flip(self):
  for key in ("as_for","regarding"):
   row=self.summary["counterfactuals"][key]
   self.assertEqual(row["comparisons"],384)
   self.assertEqual(row["global_order_flips"],384)
   self.assertEqual(row["fixed_surface_grammar_stable"],384)
 def test_parser(self):
  for r in self.rows:
   self.assertEqual(construction(r["sentence"]),r["form"])
   self.assertEqual(len(set(extract_mentions(r["sentence"]))),2)
 def test_abstention(self):
  self.assertEqual(self.summary["per_variant"]["parenthetical"]["after_first_comma"]["abstain"],384)
 def test_winners(self):
  self.assertEqual(len(self.stages[0]["train_tied_winners"]),3)
  self.assertEqual(self.stages[0]["heldout_agreement_range"],[384,768])
  self.assertEqual(self.stages[1]["heldout_agreement_range"],[768,768])
  self.assertEqual(self.stages[2]["heldout_agreement_range"],[0,384])
if __name__=="__main__":unittest.main()
