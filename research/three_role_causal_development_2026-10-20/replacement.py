"""Exact residual-site replacement; no semantic-role alignment or learned map."""
def replace(mx, value, positions, vectors):
    assert len(positions)==len(vectors)==3 and len(set(positions))==3
    pieces=[];left=0
    for pos,vector in sorted(zip(positions,vectors),key=lambda x:x[0]):
        assert left<=pos<value.shape[1]
        pieces.extend([value[:,left:pos],mx.array(vector).astype(value.dtype).reshape(1,1,-1)])
        left=pos+1
    pieces.append(value[:,left:])
    return mx.concatenate(pieces,axis=1)
