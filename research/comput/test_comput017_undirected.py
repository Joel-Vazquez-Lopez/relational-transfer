import unittest
from comput017_undirected import predict, undirected_graph

class TestUndirectedV5(unittest.TestCase):
    def test_two_hop_ignores_grammatical_voice_and_order(self):
        q='Who thanked the person who warned Bob?'
        for p in ('Alice warned Bob. Dana thanked Alice.',
                  'Bob was warned by Alice. Alice was thanked by Dana.',
                  'Dana thanked Alice; Alice warned Bob.',
                  'Dana thanked Alice. Then, Alice warned Bob.'):
            self.assertEqual(predict(p,q),'Dana')
    def test_and_intersection(self):
        p='Alice warned Bob. Alice alerted Carol. Beth warned Bob. Claire alerted Carol. Donna thanked Alice. Emma thanked Beth. Fran thanked Claire.'
        self.assertEqual(predict(p,'Who thanked the person who both warned Bob and alerted Carol?'),'Donna')
    def test_directional_collision(self):
        p='Alice warned Bob. Bob warned Carol. Dana thanked Alice.'
        self.assertIsNone(predict(p,'Who thanked the person who warned Bob?'))
    def test_graph_invariant_to_edge_direction(self):
        a='Alice warned Bob. Dana thanked Alice.'
        b='Bob warned Alice. Alice thanked Dana.'
        clean=lambda g:{v:{k:frozenset(w) for k,w in d.items()} for v,d in g.items()}
        self.assertEqual(clean(undirected_graph(a)),clean(undirected_graph(b)))
    def test_fail_closed(self):
        with self.assertRaises(ValueError):
            predict('Alice warned Bob and Carol.','Who thanked the person who warned Bob?')
if __name__=='__main__':unittest.main()
