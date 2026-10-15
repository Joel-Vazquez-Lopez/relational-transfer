import unittest
from route_falsification import evaluate,worlds,swap
class TestWitness(unittest.TestCase):
 def test_count(self):
  self.assertEqual(len(list(worlds())),24)
 def test_valid_swaps(self):
  for w in worlds():
   t=swap(w,"praised")
   self.assertEqual(sorted(w["praised"].values()),sorted(t["praised"].values()))
   self.assertEqual(w["criticized"],t["criticized"])
 def test_route_effects(self):
  x=evaluate()
  self.assertEqual(x["cache"],{"n":96,"cache_effect":96,"source_effect":0})
  self.assertEqual(x["reconstruction"],{"n":96,"cache_effect":0,"source_effect":96})
  self.assertEqual(x["shortcut"],x["reconstruction"])
  self.assertEqual(x["hybrid"],{"n":96,"cache_effect":48,"source_effect":48})
