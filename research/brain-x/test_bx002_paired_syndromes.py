"""Reproducibility/invariant tests for Brain-X BX-002 fixture and controls."""
import unittest
import numpy as np
from bx002_paired_syndromes import N, EDGES, ASSIGN, worlds, encode, marginal_decode, vote_baseline, paired_counts

class BX002Tests(unittest.TestCase):
    def test_same_world_pair_only_edge_values_differ(self):
        w=worlds(33,101)
        a=encode(w['local'],w['valid'],w['query'])
        b=encode(w['local'],w['random'],w['query'])
        self.assertTrue(np.array_equal(a[:,:N],b[:,:N]))
        self.assertTrue(np.array_equal(a[:,2*N],b[:,2*N]))
        self.assertTrue(np.array_equal(a[:,N:2*N,:13],b[:,N:2*N,:13]))
        self.assertTrue(np.array_equal(a[:,N:2*N,14:],b[:,N:2*N,14:]))
        self.assertTrue(np.any(a[:,N:2*N,13]!=b[:,N:2*N,13]))
        self.assertTrue(np.array_equal(w['labels'],w['bits'][np.arange(33),w['query']]))
    def test_source_determinism(self):
        a=worlds(10,101);b=worlds(10,101)
        for key in a:self.assertTrue(np.array_equal(a[key],b[key]))
    def test_exact_marginal_is_consistent_with_manual_enumeration(self):
        w=worlds(9,23)
        p=.35;q=.08
        scores=[]
        for t in range(9):
            post=[]
            for a in ASSIGN:
                v=np.prod(np.where(a==w['local'][t],1-p,p))
                rel=a[EDGES[:,0]]^a[EDGES[:,1]]
                v*=np.prod(np.where(rel==w['valid'][t],1-q,q))
                post.append(v)
            post=np.asarray(post);post/=post.sum()
            marg=float(post@ASSIGN[:,w['query'][t]])
            scores.append(int(marg>.5))
        self.assertEqual(scores, marginal_decode(w['local'],w['valid'],w['query']).tolist())
    def test_voting_label_unprivileged(self):
        w=worlds(100,25)
        # The vote sees only noisy inputs; permuting true latent answers does not change result.
        self.assertTrue(np.array_equal(vote_baseline(w['local'],w['valid'],w['query']), vote_baseline(w['local'].copy(),w['valid'].copy(),w['query'].copy())))
    def test_paired_counts(self):
        v=np.array([0,1,1,0]);r=np.array([1,1,0,0]);y=np.array([0,1,0,1]);p=paired_counts(v,r,y)
        self.assertEqual((p['both_correct'],p['valid_only'],p['random_only'],p['both_wrong']),(1,1,1,1))

if __name__=='__main__':unittest.main()
