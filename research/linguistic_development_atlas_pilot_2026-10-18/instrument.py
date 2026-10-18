"""Native operation capture only. No patches, model load, or inference at import."""
import contextlib
import mlx.nn as nn


class Operation(nn.Module):
    def __init__(self, original, record, key):
        super().__init__()
        self.original, self.record, self.key = original, record, key

    def __call__(self, *args, **kwargs):
        y = self.original(*args, **kwargs)
        self.record[self.key] = y
        return y


class Block(nn.Module):
    def __init__(self, original, record):
        super().__init__()
        self.original, self.record = original, record
        self.use_sliding = getattr(original, 'use_sliding', False)

    def __call__(self, x, *args, **kwargs):
        self.record['incoming'] = x
        y = self.original(x, *args, **kwargs)
        self.record['outgoing'] = y
        return y


@contextlib.contextmanager
def instrument(model):
    blocks = model.model.layers
    old, records = [], {}
    for li, block in enumerate(blocks):
        record = {}; records[li] = record
        attention, mlp = block.self_attn, block.mlp
        block.self_attn = Operation(attention, record, 'attention')
        block.mlp = Operation(mlp, record, 'mlp')
        blocks[li] = Block(block, record)
        old.append((li, block, attention, mlp))
    try:
        yield records
    finally:
        for li, block, attention, mlp in old:
            block.self_attn, block.mlp = attention, mlp
            blocks[li] = block
