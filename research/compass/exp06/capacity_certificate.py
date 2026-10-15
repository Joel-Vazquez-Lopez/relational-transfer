#!/usr/bin/env python3
"""Compass 06 exact, exhaustive one-bit caching certificate (no training data).
Requires NumPy. Four independent binary relations, 16 worlds, four queries.
Enumerates all 2^16 world->single-bit encoders, optimizes binary answers
within message clusters, and certifies guaranteed-answerable queries.
"""
import itertools
import json
import numpy as np

def certify():
    world=np.asarray(list(itertools.product((0,1),repeat=4)),dtype=np.int16)
    assignments=((np.arange(65536,dtype=np.uint32)[:,None] >>
                  np.arange(16,dtype=np.uint32))&1).astype(np.int16)
    n1=assignments.sum(1)[:,None]
    n0=16-n1
    ones1=assignments@world
    ones0=world.sum(0)[None,:]-ones1
    correct=(np.maximum(ones1,n1-ones1).sum(1)+
             np.maximum(ones0,n0-ones0).sum(1))
    safe=(np.where((ones1==0)|(ones1==n1),n1,0).sum(1)+
          np.where((ones0==0)|(ones0==n0),n0,0).sum(1))
    optimal=correct==correct.max()
    coordinate=world[:,0].astype(np.int16)
    coord_idx=int(np.where(np.all(assignments==coordinate,axis=1))[0][0])
    result={
      "num_encoders":65536, "num_world_query_pairs":64,
      "max_correct":int(correct.max()),
      "number_accuracy_optimal":int(optimal.sum()),
      "max_safe_among_accuracy_optimal":int(safe[optimal].max()),
      "coordinate_cache_correct":int(correct[coord_idx]),
      "coordinate_cache_safe":int(safe[coord_idx]),
      "max_correct_with_at_least_16_safe":int(correct[safe>=16].max())
    }
    assert result=={
      "num_encoders":65536,"num_world_query_pairs":64,
      "max_correct":44,"number_accuracy_optimal":992,
      "max_safe_among_accuracy_optimal":0,
      "coordinate_cache_correct":40,"coordinate_cache_safe":16,
      "max_correct_with_at_least_16_safe":40
    }
    return result

if __name__=="__main__":
    print(json.dumps(certify(),indent=2))
