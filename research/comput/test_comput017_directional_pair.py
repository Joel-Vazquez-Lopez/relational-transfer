"""Constructed counterfactual: same tokens and undirected graph, different directed answers."""
import collections, re, unittest
from comput017_undirected import predict, undirected_graph
A='Alice warned Bob. Bob warned Carol. Dana thanked Alice. Evan thanked Carol.'
B='Bob warned Alice. Carol warned Bob. Dana thanked Alice. Evan thanked Carol.'
QUESTION='Who thanked the person who warned Bob?'
def directed_source(text,verb,patient):
    hits=[]
    for clause in text.split('. '):
        m=re.fullmatch(r'([A-Z][a-z]+) ([a-z]+) ([A-Z][a-z]+)\.?',clause)
        if m and m.group(2)==verb and m.group(3)==patient:hits.append(m.group(1))
    if len(hits)!=1:raise ValueError(hits)
    return hits[0]
class DirectionPair(unittest.TestCase):
    def test_same_word_multiset(self):
        bag=lambda s:collections.Counter(re.findall(r'[a-z]+',s.lower()))
        self.assertEqual(bag(A),bag(B))
    def test_same_undirected_graph(self):
        canonical=lambda s:{v:{k:frozenset(xs) for k,xs in d.items()} for v,d in undirected_graph(s).items()}
        self.assertEqual(canonical(A),canonical(B))
    def test_different_unique_directed_gold(self):
        self.assertEqual(directed_source(A,'thanked',directed_source(A,'warned','Bob')),'Dana')
        self.assertEqual(directed_source(B,'thanked',directed_source(B,'warned','Bob')),'Evan')
    def test_undirected_abstains(self):
        self.assertIsNone(predict(A,QUESTION))
        self.assertIsNone(predict(B,QUESTION))
if __name__=='__main__':unittest.main()
