# Complete relational assignments in saved output scores

**Exploratory result:** a fixed external one-to-one decoder recovers the correct three-role assignment for all36 development sources in both local checkpoints.

|Checkpoint|Correct individual native answers|Complete native three-question cases|Complete external assignment cases|Native-incomplete cases recovered|
|---|---:|---:|---:|---:|
|Qwen2.5-7B-Instruct4bit|86/108|15/36|36/36|21|
|Llama3.2-3B-Instruct4bit|101/108|29/36|36/36|7|

There are no tied best assignments and no losses of native-complete cases. The smallest gold-versus-next-assignment score margins are5.2031 for Qwen and1.4928 for Llama. These are differences of sums of leading-space name-prefix log probabilities,not calibrated posterior odds for full answers. Native greedy generation and candidate-path ranking agree on all108 individual answers in each checkpoint,so the improvement comes from joint assignment selection rather than replacing a first-token metric with a different same-question winner.

## What the decoder supplies

For each source,three separate question runs provide a3x3 table of scores. Among all six one-to-one role/name mappings,choose the assignment with greatest total score. It uses no fitting or labeled-data threshold,but it does supply the task's bijection constraint and combines three responses. It is not the model's native behavior. It is not evidence that one hidden state simultaneously encodes the assignment or that an internal Hungarian/matching algorithm exists.

The assignment margins exactly cancel arbitrary question-only and person-name-only additive score offsets within each source. Five CPU fixtures and per-source arbitrary-offset checks verify that algebra to below1e-12 on these values. A single sent-person score row alone leaves two best assignments tied; a shared preferred-person effect alone cannot identify the complete mapping. This supports the narrower observation that the joint output scores distinguish the whole assignment on this panel,including cases whose individual winners repeat a wrong participant.

## Scientific use and limitations

This connects the research objective to a functionally defined measurement:follow changes in complete-assignment preferences under controlled interventions,rather than assuming a geometric trajectory is semantic. It separates a graded output readout from native expression and from causal use. The already frozen causal experiment retains native eligibility and all native/unit success gates; this decoder must not retrospectively turn a failed gate into a success.

This is an exposed-data exploratory calculation. The idea was specified after Qwen outcomes and53 Llama answers(49 correct) were observed. Both checkpoints use the same two authored name triples,one verb andthree templates. There are36 correlated cases,not36 independent linguistic replications. No new model call,p-value,held-out confirmation or novelty claim follows. Parser/copying/answer-table explanations remain viable. A supplied one-to-one constraint may be inappropriate in natural events with coreference,missing arguments or repeated participants.

## Prior work boundary

Maximum-weight assignment and constrained joint decoding are established. [BERT Is Not The Count:Learning to Match](https://homepages.inf.ed.ac.uk/scohen/eacl23match.pdf) explicitly compares local decoding with global bipartite matching in a language-model scoring task. [Asai and Hajishirzi(2020)](https://arxiv.org/abs/2004.10157) already improve QA consistency using logical/linguistic constraints. [Batch Calibration(ICLR2024)](https://research.google/pubs/batch-calibration-rethinking-calibration-for-in-context-learning-and-prompt-engineering/) already studies inference-only correction of contextual answer biases. Our simple assignment computation is not a new decoding or calibration method. These precedents do not establish the present dataset's result,but they prevent claiming the method itself as a new contribution.

## Reproduce

Run analyze.py qwen7b and analyze.py llama3b using the existing NumPy environment. Every source matrix,all six assignment scores,generated native strings,gold margin,ambiguity outcome and source-file hash is retained in the two RESULTS.json files and ledgers. Model-native raw-output replay audits in their original directories separately verify the probability inputs. Neither source's native results are overwritten.
