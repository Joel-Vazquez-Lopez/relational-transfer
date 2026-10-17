# Counterfactual transport across syntax: development gate failed

**17 October 2026. One frozen, bounded local study.** No training, model download, paid API, evaluation-family causal patch or post-outcome design change. The [lock](TRANSPORT_LOCK.json) and [gate script](transport_gate.py) were committed on the isolated research branch before fresh model outputs. The [patch script](transport_patch.py) and [analysis script](transport_analyze.py) were committed before intervention outcomes. The first upload attempt of the patch script timed out in automatic approval review; the single permitted retry succeeded before patch execution. All source files and raw outputs are preserved.

## Scientific setup

The four donor sentences orthogonalize B's semantic role and first-mentioned position through active/passive voice:

| Donor | B's role | B's surface position | Example |
|---|---|---|---|
| B Agent, active | Agent | first | William rescued Sophie. |
| B Agent, passive | Agent | second | Sophie was rescued by William. |
| B Patient, active | Patient | second | Sophie rescued William. |
| B Patient, passive | Patient | first | William was rescued by Sophie. |

The recipient was an Agent-first passive cleft, “It was by Sophie that William was rescued.” Its correct answer is Sophie; the desired counterfactual answer is William. All prompts asked “Who rescued someone?” (or the corresponding verb) and used the same person-only instruction. The primary patch replaced the residual state of the **sentence-final period token, before the question**, at block outputs 7, 15 and 23. Because of the causal mask, that token has seen the complete event but has not seen the question. The wrong-site patch used the `Sentence:` prefix token; self-patch and a same-answer unrelated event donor were included. We scored complete-name `log P(B) − log P(A)` after each patch.

At each layer, the predeclared *role contrast* `D` was mean(B-Agent donor effects) minus mean(B-Patient donor effects). The *position contrast* `P` was mean(B-first) minus mean(B-second). These are answer-effect contrasts, not internal vectors or proof of operation order. The raw log-probabilities, prompts, token positions, patch margins and controls are in [Qwen](TRANSPORT_PATCH_qwen7b_development.json) and [Llama](TRANSPORT_PATCH_llama3b_development.json) records. The captured full-dimensional development states are also saved locally as `TRANSPORT_STATES_qwen7b_development.npz` and `TRANSPORT_STATES_llama3b_development.npz`; their SHA-256 values appear in the patch JSON.

## Competence screen: passed narrowly

Four fresh families were locked as two development and two evaluation families. Every family had seven distinct Agent questions: four donor cells, both cleft answers and the unrelated donor. All failures remained in the denominator.

| Family | Split | Qwen 7B | Llama 3B |
|---|---|---:|---:|
| Laura/Michael, *greeted* | development | 5/7 | 6/7 |
| Sophie/William, *rescued* | development | 7/7 | 7/7 |
| Anna/Robert, *greeted* | evaluation | 6/7 | 7/7 |
| Megan/Patrick, *rescued* | evaluation | 7/7 | 7/7 |

Both models therefore had one common competent development family and one common competent evaluation family, satisfying the frozen gate. Qwen repeatedly misread the passive *greeted* case with B as second-mentioned Agent; Llama missed a different *greeted* role cell in development. See the complete [Qwen](TRANSPORT_GATE_qwen7b.json) and [Llama](TRANSPORT_GATE_llama3b.json) gate outputs, including generated names, expected names and full-name scores. The study remains tiny and verb-sensitive.

## Development intervention results

Only the shared competent development family, Sophie/William *rescued*, was patched. The unpatched recipient favored Sophie, with B-minus-A margins **−3.875 Qwen / −1.047 Llama**. Both models favored William in the reversed cleft, **+10.750 / +2.688**; therefore the target counterfactual answer was behaviorally accessible without a patch.

| Model | Mean role `D` over layers | Mean position `P` | Wrong-site `D` | Frozen development gate |
|---|---:|---:|---:|---|
| Qwen 7B | **+0.195** | +0.133 | 0.000 | Fail: `D` < 0.5 nat |
| Llama 3B | **+0.036** | +0.026 | 0.000 | Fail: `D` < 0.5 nat |

The complete three-layer profiles are in the [machine decision](TRANSPORT_DEV_DECISION.json). Qwen `D` was +0.289 at block 7, +0.313 at block 15 and −0.016 at block 23. At block 7, its **position contrast (+0.461) exceeded the role contrast**. Llama `D` was +0.078, +0.016 and +0.016. The Qwen same-answer *unrelated-event* donor shifted the B margin by **+1.094** and **+0.922** at blocks 7 and 15, larger than its role contrast; this is consistent with a broad answer-source effect and does not identify a role-specific relation. Its effect was nearly zero at block 23. The wrong-site prefix contrasts were zero, as expected when a common prefix precedes all sentence content. Self patches matched unpatched margins within the predeclared 0.05-nat technical tolerance.

The development rule required `D > 0.5` nats, `D > |P|`, and a wrong-site `|D| < 0.5 D` in **each** model, averaging the three frozen layers and competent development families. Both failed the first requirement. As required by the lock, **no evaluation-family patches were run**. We did not move the patch to a more promising site, add layers or change the sentence frame after seeing these results.

## Interpretation and decision

**NO-GO for this particular transport test.** At the sentence-final event token and three tested layers, the intervention did not selectively transport B's Agent status strongly enough across syntax in either model. Qwen showed a modest middle-layer effect, but its early position signal and larger unrelated-donor effect defeat a semantic-role reading. Llama was nearly insensitive to this patch. The result does **not** falsify the broader possibility of task-conditioned relations among computational paths: a single token patch may miss distributed information, and the competence set contains only one development family per model. Those are limitations, not a license to adaptively try another site in this study.

The first path-similarity failure, the factorial competence failure and this native causal failure jointly support a stricter research standard: geometric relations should earn interpretation only by prospectively predicting selective intervention effects beyond output and position controls. We have not yet found such an effect or a publishable new mechanism. The complete negative record is preserved so a later, separately frozen study can build on it without relabeling this run as a success.
