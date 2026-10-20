"""Sequential native Qwen layers; fixed shape; caller-supplied boundary callback.

Known layer-offloading technique, not a new linguistic mechanism. No weights or
native layer operations are changed. Evaluated modules are freed immediately.
"""
import gc
from pathlib import Path

def forward(ids,model_path,tap=None,check=None):
    import mlx.core as mx
    from mlx_lm.utils import load_model
    from mlx_lm.models.base import create_attention_mask
    model,config=load_model(Path(model_path),lazy=True)
    assert not config.get('tie_word_embeddings',False) and len(model.model.layers)==28
    h=model.model.embed_tokens(mx.array(ids));mx.eval(h)
    model.model.embed_tokens=None;mx.clear_cache()
    if tap:h=tap(0,h)
    if check:check(0)
    mask=create_attention_mask(h,None)
    for i in range(28):
        block=model.model.layers[i];h=block(h,mask,cache=None);mx.eval(h)
        model.model.layers[i]=None;del block;mx.clear_cache()
        if tap:h=tap(i+1,h)
        if check:check(i+1)
    h=model.model.norm(h);mx.eval(h)
    logits=model.lm_head(h);mx.eval(logits)
    del model,h;gc.collect();mx.clear_cache()
    return logits
