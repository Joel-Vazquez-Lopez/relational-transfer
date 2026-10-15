# COMPUT-009: Surface-shortcut equivalence and smallest discriminator

**Scope:** executed pure-Python synthetic control, not pretrained-model inference. Code is in `research/comput/comput009/`. Locally verified: 13 tests in full package; GitHub's consolidated test file has 10 test methods. No sealed data or paid inference.

## Experimental design
Six name pairs, four verbs, both role assignments, four constructions, two requested roles and six surface variants produce **2,304 question instances from 192 base event/construction instances**. All stimuli are generated with fixed templates. Introductory clauses mention the actor or patient first; another variant inserts an internal comma-delimited parenthetical.

Three predictors consume the rendered sentence and question only: global first/last distinct-name order conditioned on construction, strip everything before first comma then apply that positional router, and explicit removal of supported prefixes/parentheticals before applying the same router. No gold roles used as predictor features. Closed-form question interpretation, construction recognition and name extraction are hard-coded to generator grammar.

| Variant | Global position /384 | Strip-first-comma /384 | Fixed surface grammar /384 |
|---|---:|---:|---:|
| Canonical | 384 | 384 | 384 |
| As for actor | 192 | 384 | 384 |
| As for patient | 192 | 384 | 384 |
| Regarding actor | 192 | 384 | 384 |
| Regarding patient | 192 | 384 | 384 |
| Internal parenthetical | 384 | **0 (384 abstain)** | 384 |

Both prefatory marker families yield 384/384 positional answer flips when actor vs patient is topicalized, while the fixed grammar's answers remain unchanged.

### Controlled selection by synthetic correctness
- Training canonical, testing As-for prefixes: all three rules training-tied, **384–768/768** possible held-out correctness.
- Training canonical + As-for, testing Regarding: comma strip and fixed grammar tied, both **768/768**.
- Training canonical + all prefaces, testing internal parenthetical: same two policies remain tied on 1,920 earlier examples; they achieve **0/384** versus **384/384** on the final challenge.

This is *design identifiability*: two transparent non-event-computing programs have identical observed answers on 1,920 questions but are separated by a new punctuation condition. The held-out variants were known to the author of the hard-coded policies; **not a valid demonstration of learned transfer**. Merely counterbalancing semantic roles and mention order does not eliminate construction/punctuation shortcuts.

## Limitations
Stylized topicalizations, name repetition, simplistic titlecase participant detector, fixed verb list, explicit actor/patient questions, template-sensitive syntactic tags. These programs need not describe how real LLMs compute. Question count is not count of independent linguistic source events. Format validity is reported distinctly: comma heuristic abstains on all 384 internal-parenthetical questions; it does not issue 384 wrong names.

## Reproduction
`cd research/comput/comput009 && python3 -m unittest discover -p 'test_*.py' -v`
`python3 comput009.py --results /tmp/results.json --stimuli /tmp/stimuli.jsonl`
`python3 selection.py`

Full original executed outputs and 13-test log are supplied in separate downloadable package `COMPUT_009_SURFACE_SHORTCUT_AUDIT.zip`; GitHub contains source and summary report. Historical archived model outputs remain unavailable event-by-event.

## Next falsification gate
Human-review a small, natural, paired counterfactual stimulus panel using original unsealed events. Predeclare rule features by text-only extraction and compare first/last, comma strip, fixed grammar and a genuine parser on held-out participant pairs, predicates and discourse structures. No internal mechanism claim without matched causal controls.
