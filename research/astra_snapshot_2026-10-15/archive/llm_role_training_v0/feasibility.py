"""Five-step local CPU LoRA feasibility gate, not an LLM result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import time

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = ROOT / "FEASIBILITY.json"
NAMES = ("Mira", "Owen", "Leila", "Victor", "Nadia", "Felix",
         "Iris", "Amir", "Clara", "Hugo")


class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int = 4, alpha: int = 8) -> None:
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, original.in_features))
        self.b = nn.Parameter(torch.zeros(original.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5 ** .5)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def add_adapters(model: nn.Module) -> int:
    model.requires_grad_(False)
    count = 0
    for layer in model.model.layers:
        for name in ("q_proj", "v_proj"):
            original = getattr(layer.self_attn, name)
            assert isinstance(original, nn.Linear)
            setattr(layer.self_attn, name, LoRALinear(original))
            count += 1
    return count


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.manual_seed(810107)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
                                                 dtype=torch.float32)
    model.config.use_cache = False
    adapter_count = add_adapters(model)
    hidden = model.config.hidden_size
    first_head = nn.Linear(hidden, len(NAMES))
    second_head = nn.Linear(hidden, len(NAMES))
    prompt_text = tokenizer.apply_chat_template([
        {"role": "system", "content": "Answer with the name only."},
        {"role": "user", "content": "Sentence: Mira chased Owen.\nWho performed the action?"},
    ], tokenize=False, add_generation_prompt=True)
    prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    answer = tokenizer.encode("Mira", add_special_tokens=False)
    tokens = torch.tensor([prompt + answer + [tokenizer.eos_token_id]], dtype=torch.long)
    labels = torch.tensor([[-100] * len(prompt) + answer + [tokenizer.eos_token_id]],
                          dtype=torch.long)
    parameters = [p for p in model.parameters() if p.requires_grad]
    parameters += list(first_head.parameters()) + list(second_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    diagnostics = []
    for step in range(5):
        start = time.perf_counter()
        model.train()
        first_head.train()
        second_head.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=tokens, labels=labels, output_hidden_states=True,
                       use_cache=False)
        state = output.hidden_states[-1][:, len(prompt) - 1, :]
        actor = torch.tensor([NAMES.index("Mira")])
        patient = torch.tensor([NAMES.index("Owen")])
        auxiliary = (F.cross_entropy(first_head(state), actor)
                     + F.cross_entropy(second_head(state), patient)) / 2
        loss = output.loss + .5 * auxiliary
        loss.backward()
        lora_grad = sum(float(p.grad.norm()) for p in model.parameters()
                        if p.requires_grad and p.grad is not None)
        head_grad = sum(float(p.grad.norm()) for p in
                        list(first_head.parameters()) + list(second_head.parameters())
                        if p.grad is not None)
        assert torch.isfinite(loss) and lora_grad > 0 and head_grad > 0
        optimizer.step()
        elapsed = time.perf_counter() - start
        record = {"step": step + 1, "seconds": elapsed,
                  "total_loss": float(loss.detach()),
                  "language_loss": float(output.loss.detach()),
                  "auxiliary_loss": float(auxiliary.detach()),
                  "lora_gradient_norm_sum": lora_grad,
                  "head_gradient_norm_sum": head_grad}
        diagnostics.append(record)
        print(json.dumps(record), flush=True)
    result = {"model": MODEL, "adapter_modules": adapter_count,
              "model_parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in parameters),
              "prompt_tokens": len(prompt), "answer_tokens": len(answer) + 1,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
              "protocol_sha256": hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest(),
              "steps": diagnostics}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
