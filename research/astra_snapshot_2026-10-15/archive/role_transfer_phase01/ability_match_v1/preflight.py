"""Small validity check before the ability-only weight sweep."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

import match_experiment as exp

HERE = Path(__file__).resolve().parent


def main() -> None:
    exp.audit()
    exp.prior.audit()
    exp.prior.configure(0, "sham")
    rows = next(exp.prior.batches(0, "sham", exp.SEEDS[0]))
    outputs = []
    for weight in exp.WEIGHTS:
        exp.prior.base.set_seed(exp.SEEDS[0])
        model = exp.prior.auxiliary.RoleListener()
        exp.prior.auxiliary.WEIGHT = weight
        model.zero_grad(set_to_none=True)
        loss, detail = exp.prior.objective(model, rows, "ordered_roles")
        assert torch.isfinite(loss)
        loss.backward()
        grad = float(model.sym.grad.norm())
        role_grad = float(model.first_head.weight.grad.norm())
        assert grad > 0 and role_grad > 0 and math.isfinite(grad) and math.isfinite(role_grad)
        outputs.append({"weight": weight, "total_loss": float(loss.detach()),
                        "native_loss": detail["native"], "auxiliary_loss": detail["auxiliary"],
                        "query_gradient": grad, "role_head_gradient": role_grad})
    result = {"status": "valid", "stage": "one_batch_gradient_preflight",
              "focal_scores_computed": False, "outputs": outputs,
              "protocol_sha256": exp.digest(HERE / "PROTOCOL.json")}
    (HERE / "PREFLIGHT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
