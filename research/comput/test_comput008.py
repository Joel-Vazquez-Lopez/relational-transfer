import unittest
from comput008 import generate,predict,audit,FORMS,RULES
class TestComput008(unittest.TestCase):
 def setUp(self):self.rows=generate()
 def test_size(self):self.assertEqual(len(self.rows),384)
 def test_first_last_balanced(self):
  for form in FORMS:
   for rule in ("first","last"):
    rr=[r for r in self.rows if r["form"]==form]
    self.assertEqual(sum(predict(r,rule)==r["gold"] for r in rr),48)
 def test_construction_shortcut(self):
  for r in self.rows:self.assertEqual(predict(r,"construction_position"),r["gold"])
 def test_gold_is_never_feature(self):
  for r in self.rows:
   for rule in RULES:self.assertEqual(predict(r,rule),predict(dict(r,gold="SPOOF"),rule))
 def test_surface_position(self):
  for r in self.rows:self.assertLess(r["sentence"].index(r["first"]),r["sentence"].index(r["last"]))
 def test_question_flip(self):
  for r in self.rows:
   key=(r["pair_id"],r["verb"],r["direction"],r["form"])
   other=[z for z in self.rows if (z["pair_id"],z["verb"],z["direction"],z["form"])==key and z["question"]!=r["question"]]
   self.assertEqual(len(other),1);self.assertNotEqual(other[0]["gold"],r["gold"])
 def test_cell_counts(self):
  for form,d in audit(self.rows)["cells"].items():
   self.assertEqual(d["construction_position"]["correct"],96)
if __name__=="__main__":unittest.main()
