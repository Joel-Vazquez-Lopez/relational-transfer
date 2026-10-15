import unittest
from cluster_margin_bounds import bounds, sign_p
class Tests(unittest.TestCase):
 def test_bounds(self):
  self.assertEqual(bounds(34,26)["p_max"],1.)
  self.assertEqual(bounds(26,7)["p_min"], 0.00048828125)
  self.assertAlmostEqual(bounds(42,18)["p_max"],0.548828125)
 def test_small_exhaustive(self):
  from itertools import product
  for a in range(5):
   for b in range(5):
    feasible=set()
    for av in product((0,1),repeat=4):
     if sum(av)!=a:continue
     for bv in product((0,1),repeat=4):
      if sum(bv)!=b:continue
      x=sum(av[:2]); y=sum(bv[:2]);u=sum(av[2:]);v=sum(bv[2:])
      pos=(x>y)+(u>v); neg=(x<y)+(u<v)
      feasible.add(sign_p(pos,neg))
    r=bounds(a,b,2,2)
    self.assertEqual((r["p_min"],r["p_max"]),(min(feasible),max(feasible)))
if __name__=="__main__":unittest.main()
