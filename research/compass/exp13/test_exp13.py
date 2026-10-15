"""Deterministic structural checks for Compass Investigation 13."""
import itertools, unittest
from collections import Counter
from exp13_curriculum import dataset, PAIRS, final_state

class TestGenerator(unittest.TestCase):
    def test_sizes_and_unseen_fourcycles(self):
        rows=dataset()
        self.assertEqual(len(rows),1372)
        self.assertEqual(Counter(r['length'] for r in rows),{0:4,1:72,2:432,3:864})
        self.assertEqual(sum(r['cycle_type']=='4' and r['length']==3 for r in rows),384)
        self.assertEqual(sum(r['cycle_type']=='4' and r['length']<3 for r in rows),0)
        self.assertEqual(len(set(r['world'] for r in rows if r['length']==3 and r['cycle_type']=='4')),6)
    def test_order_and_inverse(self):
        for p,q in itertools.product(PAIRS,repeat=2):
            state=final_state([p,q])
            self.assertEqual(tuple(sorted(state)),(0,1,2,3))
            self.assertEqual(final_state([p,p]),(0,1,2,3))
            if set(p).isdisjoint(q):
                self.assertEqual(state,final_state([q,p]))
            elif p!=q:
                self.assertNotEqual(state,final_state([q,p]))
    def test_gold_is_question_independent(self):
        rows=dataset()
        for i in range(0,len(rows),4):
            group=rows[i:i+4]
            self.assertEqual([r['q'] for r in group],[0,1,2,3])
            self.assertEqual(tuple(r['y'] for r in group),group[0]['world'])
            self.assertEqual(len(set(tuple(r['x']) for r in group)),1)
    def test_entity_relabeling(self):
        for p,q in itertools.product(PAIRS,repeat=2):
            s=final_state([p,q])
            for perm in itertools.permutations(range(4)):
                rp=tuple(perm[x] for x in p)
                rq=tuple(perm[x] for x in q)
                shifted=final_state([rp,rq])
                for k in range(4):
                    self.assertEqual(shifted[perm[k]],perm[s[k]])

if __name__=='__main__':
    unittest.main()
