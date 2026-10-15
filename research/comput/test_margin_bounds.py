import unittest
from margin_bounds import tables,summarize
class TestMargins(unittest.TestCase):
 def test_counts(self):
  for r in tables(48,34,26):
   self.assertEqual(r["both"]+r["a_only"],34)
   self.assertEqual(r["both"]+r["b_only"],26)
   self.assertEqual(sum(r[k] for k in ("both","a_only","b_only","neither")),48)
 def test_bounds(self):
  self.assertEqual(summarize(48,34,26)["minimum_discordant"],8)
  self.assertEqual(summarize(48,26,7)["minimum_discordant"],19)
  self.assertEqual(summarize(48,42,18)["minimum_discordant"],24)
 def test_identical_margins(self):
  self.assertEqual(summarize(10,4,4)["possible_exact_p_range"],[1.,1.])
 def test_reject_invalid(self):
  with self.assertRaises(ValueError):tables(10,11,0)
if __name__=="__main__":unittest.main()
