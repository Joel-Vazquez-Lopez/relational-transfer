"""Technical smoke test: read and alter one Qwen transformer block output."""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = os.environ["ROLE_PILOT_MODEL"]
torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.float32).eval()
prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "Answer questions about sentences. Give only the person's name."},
        {"role": "user", "content": "Sentence: Bob was chased by Alice.\nWho performed the action? Reply with the name only."},
    ],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt")
layer = model.model.layers[len(model.model.layers) // 2]
captured = []


def capture(_module, _input, output):
    captured.append(tuple(output.shape))


with torch.inference_mode():
    baseline = model(**inputs).logits[:, -1, :]
handle = layer.register_forward_hook(capture)
with torch.inference_mode():
    repeated = model(**inputs).logits[:, -1, :]
handle.remove()
assert torch.allclose(baseline, repeated)


def replace_last_token(_module, _input, output):
    hidden = output.clone()
    hidden[:, -1, :] = 0
    return hidden


handle = layer.register_forward_hook(replace_last_token)
with torch.inference_mode():
    intervened = model(**inputs).logits[:, -1, :]
handle.remove()
print("layers", len(model.model.layers))
print("captured_middle_layer_shape", captured)
print("maximum_absolute_logit_change", (intervened - baseline).abs().max().item())
print("baseline_first_token", tokenizer.decode(baseline.argmax(dim=-1)))
print("intervened_first_token", tokenizer.decode(intervened.argmax(dim=-1)))
