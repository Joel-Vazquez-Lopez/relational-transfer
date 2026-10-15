# Comput Report 008 — Counterbalanced synthetic roles still admit a perfect shortcut
Date: 15 October 2026. Scope: **executed deterministic synthetic falsification**, not LLM inference or unseen-natural-language evaluation.

## Design
Six distinct name pairs, four transitive verbs, both actor/patient assignments, four surface templates (active, passive, subject cleft, object cleft), and both actor/patient questions: 6×4×2×4×2 = **384** question-level items.

Baseline input features: participant positions extracted from literal sentence surface, question-role indicator and, only for the final rule, construction tag. None of the predictors receives `gold`, actor/patient reference fields, or name-to-semantic-role mappings.

The three ordinary order rules are fixed independent of construction. `construction_position` uses the predeclared purely positional mapping: active/subject-cleft -> first is actor; passive/object-cleft -> last is actor. This is deliberately **not** a language parser or learned circuit.

## Findings: number correct / 96 per form
| Rule | Active | Passive | Subject cleft | Object cleft |
|---|---:|---:|---:|---:|
| always first | 48 | 48 | 48 | 48 |
| always last | 48 | 48 | 48 | 48 |
| first for actor else last | 96 | 0 | 96 | 0 |
| last for actor else first | 0 | 96 | 0 | 96 |
| construction-position shortcut | **96** | **96** | **96** | **96** |

### Strong negative conclusion
This panel is **fully counterbalanced by names and semantic-role assignment**, but remains **trivially solvable by construction-conditioned answer position**. Such balancing does not break shortcuts, even though first-only and last-only controls are at chance. Perfect model accuracy on this kind of panel would not distinguish a template router from an abstract role-binding computation.

### Integrity
An initial exploratory test incorrectly asserted that the question-conditioned positional rules should also be at chance in each form; it FAILED. We corrected this assumption and retained the failure in the report. Seven tests pass in the repository's self-contained version. A separate extended local prototype with syntactic-subject abstention passed eight tests. No pretrained LLM was run, no sealed data used, no inference spending.

## Next discriminating control
Hold lexical items and semantic roles fixed while varying surface templates **within a construction label**, so positions cease to be a deterministic function of construction. Pre-register exact label-free feature extraction and train/test splits by pair and verb. To rule out more sophisticated structural template routers, add naturally occurring and novel syntactic variants with human-verified roles and out-of-distribution evaluation. However, no finite behavioral probe can uniquely identify an internal circuit without further mechanistic controls.

Run: `python3 research/comput/comput008.py --output /tmp/comput008.json` and `python3 -m unittest discover -s research/comput -p 'test_comput008.py'`.
