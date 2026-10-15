import unittest
from paired_audit import run, sign_p, feasible_clusters
class PairedAuditTests(unittest.TestCase):
    def test_event_sign(self):
        self.assertAlmostEqual(sign_p(7,2),0.1796875)
        self.assertAlmostEqual(sign_p(0,8),0.0078125)
    def test_phi_cluster_allocations(self):
        self.assertEqual(feasible_clusters(0,8),[(0,4,2),(0,5,1),(0,6,0)])
    def test_conservation(self):
        for s in run().values():
            self.assertEqual(s['gains']+s['losses']+s['unchanged'],12)
            self.assertEqual(s['after']-s['before'],s['gains']-s['losses'])
if __name__=='__main__': unittest.main()
