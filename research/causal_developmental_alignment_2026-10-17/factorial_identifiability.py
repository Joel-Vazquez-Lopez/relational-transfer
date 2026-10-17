"""Exact toy controls for interpreting a two-factor representation contrast.

Run: python factorial_identifiability.py
No model inference or external dependencies beyond NumPy.
"""
import json

import numpy as np


def interaction(cells):
    return cells[1, 1] - cells[1, 0] - cells[0, 1] + cells[0, 0]


def result(cells):
    effect = interaction(cells)
    role = cells[1, 0] - cells[0, 0]
    voice = cells[0, 1] - cells[0, 0]
    return {
        "interaction_norm": float(np.linalg.norm(effect)),
        "role_norm": float(np.linalg.norm(role)),
        "voice_norm": float(np.linalg.norm(voice)),
    }


def main():
    # Independent binary factors, explicitly stored in two coordinates.
    independent = np.array([[[r, v] for v in (0, 1)] for r in (0, 1)], dtype=float)
    # The same factors under an injective nonlinear encoding on this domain.
    nonlinear_encoding = np.array(
        [[[r, v, r * v] for v in (0, 1)] for r in (0, 1)], dtype=float
    )
    assert np.allclose(interaction(independent), 0)
    assert np.allclose(interaction(nonlinear_encoding), [0, 0, 1])
    # An additive code can still feed a nonlinear downstream function.
    nonlinear_output = np.array([[r * v for v in (0, 1)] for r in (0, 1)], dtype=float)
    assert nonlinear_output[1, 1] - nonlinear_output[1, 0] - nonlinear_output[0, 1] + nonlinear_output[0, 0] == 1
    # A common orthogonal change of basis preserves all factorial norms and Grams.
    rng = np.random.default_rng(741)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = nonlinear_encoding @ q
    assert np.isclose(np.linalg.norm(interaction(rotated)), np.linalg.norm(interaction(nonlinear_encoding)))
    print(json.dumps({
        "additive_state": result(independent),
        "same_factors_nonlinear_code": result(nonlinear_encoding),
        "additive_state_nonlinear_output_interaction": float(interaction(nonlinear_output)),
        "common_orthogonal_rotation_preserves_interaction_norm": True,
        "conclusion": "State interaction alone does not identify computational dependence."
    }, indent=2))


if __name__ == "__main__":
    main()
