import unittest
from collections import Counter
from build_pairs import bag, check, make_rows


class PairDesignTests(unittest.TestCase):
    def setUp(self):
        self.rows = make_rows(16)

    def test_cardinality_and_grouping(self):
        self.assertEqual(len(self.rows), 512)
        self.assertEqual(check(self.rows), 16)
        self.assertEqual(Counter(r['condition'] for r in self.rows),
                         {'identity': 128, 'target_role_swap': 128,
                          'distractor_role_swap': 128, 'predicate_swap': 128})

    def test_same_question_and_bag(self):
        for r in self.rows:
            self.assertEqual(bag(r['base_context']), bag(r['donor_context']))
            self.assertIn("For the event '", r['query'])

    def test_selective_gold_flip(self):
        for r in self.rows:
            flip = r['base_gold'] != r['donor_gold']
            self.assertEqual(flip, r['condition'] in ('target_role_swap', 'predicate_swap'))

    def test_independent_families_and_splits(self):
        identities = {}
        splits = {}
        for r in self.rows:
            identities.setdefault(r['family_id'], set()).update(
                (r['base_gold'], r['donor_gold']))
            splits.setdefault(r['family_id'], set()).add(r['split'])
        self.assertTrue(all(len(v) == 1 for v in splits.values()))
        self.assertEqual(len(set.union(*identities.values())), 16 * 4)


if __name__ == '__main__':
    unittest.main()
