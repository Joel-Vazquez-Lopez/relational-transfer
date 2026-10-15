import unittest
from collections import Counter
from build_pairs_v2 import make_rows, audit, bag, surface_lookup,sorted_name_policy


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=make_rows(16)
        cls.result=audit(cls.rows)

    def test_pairs(self):
        self.assertEqual(len(self.rows),512)
        self.assertEqual(len(set(r['pair_id'] for r in self.rows)),512)
        self.assertEqual(Counter(r['condition'] for r in self.rows), {k:128 for k in ('identity','target_role_swap','distractor_role_swap','predicate_swap')})

    def test_token_bag_and_query(self):
        for r in self.rows:
            self.assertEqual(bag(r['base_context']),bag(r['donor_context']))
            self.assertEqual(r['informative'],r['base_gold']!=r['donor_gold'])

    def test_surface_lookup(self):
        self.assertEqual(self.result['surface_lookup_correct'],1024)
        for r in self.rows:
            self.assertEqual(surface_lookup(r['base_context'],r['query']),r['base_gold'])
            self.assertEqual(surface_lookup(r['donor_context'],r['query']),r['donor_gold'])

    def test_balanced_gold_per_role(self):
        for role in ('actor','patient'):
            for condition in ('identity','target_role_swap','distractor_role_swap','predicate_swap'):
                subset=[r for r in self.rows if r['role']==role and r['condition']==condition]
                self.assertEqual(len(subset),64)
                for prefix in ('base','donor'):
                    match=sum(sorted_name_policy(r[f'{prefix}_context'],r['query'])==r[f'{prefix}_gold'] for r in subset)
                    self.assertEqual(match,16,(role,condition,prefix))

    def test_family_split(self):
        splits={}
        for r in self.rows: splits.setdefault(r['family_id'],set()).add(r['split'])
        self.assertEqual(len(splits),16)
        self.assertTrue(all(len(s)==1 for s in splits.values()))


if __name__=='__main__': unittest.main()
