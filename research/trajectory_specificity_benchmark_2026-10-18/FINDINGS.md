# What this benchmark actually found

**Decision: NARROW.** A controlled geometric regularity survives execution-shape correction, but neither a shared semantic algorithm nor a novel mechanism has been identified.

## 1. The artifact does not explain the primary sentence-period result

All 16 family/model content-null checks passed exactly. All 176 declared forward passes completed, with no unavailable 64/65 null pairs. Qwen changed by at most 0.005417 in the active/passive role-response cosine; Llama changed by exactly zero. No family exceeded the frozen 0.05 materiality threshold, and none of the eight primary native/fixed gap signs reversed. This rejects the benchmark-specific prediction that correcting this numerical artifact materially alters the primary period-level result.

## 2. There is a modest, stable within-model pattern

Using only Gram entries for layer steps at least eight apart:

| Model | Role/role | Role/unrelated | Difference | Leave-one-family-out difference | Largest angle correction |
|---|---:|---:|---:|---:|---:|
| qwen7b | 0.4477 | 0.3515 | +0.0961 | 0.0653–0.1301 | 0.005417 |
| llama3b | 0.2695 | 0.1171 | +0.1524 | 0.1154–0.1809 | 0.000000 |

Each leave-one-family-out gap is positive. These are sensitivity checks on eight author-constructed families, not confidence intervals or independent replications. The comparison is descriptive: the unrelated paths change names and usually verbs, whereas role reversal preserves the lexical inventory and swaps its order. A lexical-permutation explanation remains untested. The result should be called controlled role-reversal **geometry**, not evidence that the model represents an abstract role variable.

## 3. An apparent family difference at the first person disappears

At the first person token, active_a and passive_b have identical causal token prefixes. Native captures produced a nonzero whole-layer contrast in 8/8 Qwen families and 0/8 Llama families. Fixed-shape capture produced zero in all 16 family/model cases. Qwen native prompts have 63 versus 65 tokens, while the Llama prompts are already above that boundary. This demonstrates a specific way tokenization/prompt length and a shared runtime can masquerade as a model-family difference at a causally ineligible site. It does not explain the valid sentence-period contrast.

## 4. Long-range null-path coherence is weak

Numerical-null within-model Gram correlations fall from 0.319 to 0.0036 for Qwen and 0.745 to 0.0538 for Llama when the minimum lag changes from 1 to 8. Nearby-step structure supplies much of the apparent numerical-path coherence. This is evidence against a strong long-range shared numerical trajectory in this fixture. Absolute null endpoint norms are at most 2.38% and 2.18% of the active role-contrast endpoint norm for Qwen/Llama respectively.

## 5. The cross-model matrix does not establish operation matching

Fixed-shape long-range role/role similarity across models is about 0.167 (active) and 0.191 (passive). However, a Qwen unrelated contrast compared with a Llama active-role contrast is about 0.148. Much of the matrix favors Llama role trajectories irrespective of the Qwen operation. The complete six-by-six means are saved separately. Matched-family advantages are small. High individual entries cannot be treated as successful matching of labeled computations.

## Research decision

The next discriminating linguistic control would preserve roles while matching lexical reordering, rather than use an unrelated sentence as the only rival. This is a proposed control, not an experiment run here. No learned centering or new similarity measure was introduced after outcomes. The numerical artifact line is now bounded; the residual within-model geometry is a lead, not yet a paper-level semantic discovery. A technical-note draft makes the actual claim concrete for review.
