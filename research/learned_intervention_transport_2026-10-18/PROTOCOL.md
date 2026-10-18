# Learned intervention transport: frozen bounded protocol

2026-10-18. Follow-up to ba6c5bcd1cdac74908f4ebe026c016aaef5979ad. All rules below and both scripts are frozen before training or held-out outcomes. One fixed run; no adaptive training, changing seeds, doses, sites, scoring, data or thresholds after results. Local CPU, existing PyTorch/NumPy environment; no paid APIs, downloads or pretrained inference.

## Question and scope

Can an intervention-fitting procedure distinguish a real route difference from a mismatch introduced by coordinates? A second, harder question: do matched developmental response profiles predict individual counterfactuals across independently learned representations?

This is a small learned-network methods study. Networks consume **supplied grammar features**, not raw sentence tokens; they are tanh MLPs, not Transformers. We train a role function. Questions and answer-option identity are routed algebraically after the role score. Generalization to new names/templates therefore does NOT demonstrate learned parsing, question understanding or lexical generalization. The task is linguistically grounded as an interpretable factor system; only neural feature processing and its causal use are learned.

## Data and annotations

Independent binary factors: role R (canonical A is Agent), voice V (active vs passive), mention order O (A first), question Q (Agent/Patient), answer option order L. Construction F=RVO indicates the normal or fronted/relative realization. Feature input [O,V,F,N1,N2,T] excludes R, Q and L; the learned role is R=OVF. N1,N2 are deterministic bounded hashes of two names, and T encodes template context, a nuisance. The readout answer coordinate is role_logit × Q × L, mapped to two supplied name options.

Full 32-factor-cell grid × eight contexts (256 cases) for training; × four contexts (128) each for development and held sets. Distinct name pools per split. Train/development texts use praised, today/yesterday contexts; held uses thanked and at-noon forms. Names do not repeat across splits. Held template scalar is ±.5 rather than ±1. The actual network sees feature encodings, so these textual changes are annotation/manifold checks, not substantive linguistic OOD tests. All generated texts, features, roles, questions and answers are saved.

Normal active: Agent praised Patient. Fronted active relative: Patient was the person whom Agent praised. Normal passive: Patient was praised by Agent. Agent-fronted passive: By Agent, Patient was praised. The relative-clause annotation targets the lexical action, not the matrix copula; agent-fronted passives can sound marked. All annotations are deterministic AI-authored generation rules, no human annotation and no external model calls. This is synthetic data; no source corpus claim.

## Six trained route instances, three paired seeds

Seeds 11,23,37; float64, two CPU threads. Each seed constructs a learned encoder 6→16→16→16 with tanh and a learned head 16→16→1 with tanh. Train encoder+head on binary role loss for exactly 1,800 full-batch Adam steps, lr=.01, weight decay=1e-5. No early stopping or choosing the best checkpoint.

Then freeze that trained encoder/head and train a separate 3→16→16→1 tanh branch on the raw [O,V,F] grammar features, same fixed optimization. This branch's output bypasses the encoder entirely. The cached route uses the trained intermediate state; the bypass route computes the same role task independently. Both routes expose the exact same encoder states within a seed; those states are epiphenomenal for bypass. There are six distinct trained route instances in three shared-encoder containers, not six fully independent end-to-end training runs. Three encoders/heads and three bypass heads are optimized. This deliberately constructed unused branch is a control, not a claim that natural Transformers have such branches.

Across seeds, shared access routes do NOT guarantee identical learned algorithms. Same-route is an architectural label whose usefulness for transfer is tested, not ground truth for unrestricted algorithm identity.

## Mandatory competence gate

After fixed training and before any intervention, require all normal cases and their O-reversed counterfactuals correct in all splits, in both routes and all seeds. No candidate may be substituted if it fails. Also require signed logit contrast denominator >1e-8. Any failure stops the intervention assay and yields NO-GO for this run. Preserve every failure, weights and training trace.

## Stage sites, coordinates and interventions

Patch post-tanh states after each of three encoder layers. At a patched cached site, resume the remaining encoder layers and head; at a bypass site, the separate branch ignores the cache. These are known native stage identities; there is no claim to have solved correspondence between architectures.

Coordinate variants applied independently at the selected site:

- Identity.
- Orthogonal Q from QR of NumPy Gaussian seed=100×model_seed+stage.
- Nonlinear T(h)=[sinh(2h)/2]Q, exact inverse asinh(2uQᵀ)/2.

The whole wrapped graph executes T followed by T^-1 before the original downstream map. Thus normal functions are exactly the same mathematically. Arbitrary nonlinear hidden-coordinate changes do not preserve standard neural architecture form. This is a measurement invariance stress, not an ordinary independently trained checkpoint or a biological reparameterization claim.

Fit native contrast d=E[h|R=+1]−E[h|R=-1] on balanced normal development cases. Patch I(h)=h−dose×R×d. This standard difference of means uses experimental role labels; it does not infer semantics without labels. The role contrast is a fitted direction, not a private native register.

For each transformed copy compare:

1. Native original I(h).
2. Oracle transported intervention T(I(T^-1(u))). It uses the known T and must reproduce native outputs.
3. Blind fitted intervention u−dose×R×d_u, where d_u is fit on transformed normal states without access to T or native directions. Execution necessarily inverts T as part of the wrapped model, but the fitting function receives only transformed states and role labels.

This tests a fitting procedure rather than introducing a new similarity metric. Exact transport is an oracle upper-bound/control; its accuracy is guaranteed algebraically and is not a discovery.

Development doses 0,.25,.5,1. Held doses 0,.125,.375,.625,.875,1; four interior held doses are withheld from alignment. Dose zero is sham. Include natural full-state role-reversed donors (O flips) and same-role changed-surface donors (O and V flip together). Store every scalar native/oracle/blind logit, not just normalized summaries, all natural hidden states, directions, weights, per-case manifests and full donor outcomes.

Primary response E=R(clean_role_logit−patched_role_logit) / [R(clean_role_logit−O-flipped_role_logit)]. This per-case denominator anchors to a correct natural alternative within each model rather than subtracting raw costs across differently confident models. No clipping; record denominator and raw logits. It remains a chosen logit response scale, not a privileged semantic metric. Probabilities/answer identities are reconstructable from logits and Q,L.

## Frozen predictions and gates

P1: normal wrapped outputs, native vs oracle outputs, and all shams differ by <1e-8. Failure invalidates transport conclusions.
P2: blind orthogonal and identity effects match native within 1e-8. Difference-of-means is equivariant to linear changes here.
P3: blind nonlinear effects need not match native, despite identical functions. Report each seed/stage error and fraction of individual errors >.05 and >.1. Practical replication target: cached held RMSE >.05 in each of three seeds, pooled across stages and doses. Absence of that effect is a failed prediction, not grounds to increase warp strength.
P4: bypass cache interventions have exactly zero effect despite identical matched encoder geometry. Cached natural role donors reverse all answers; same-role surface donors preserve their identity. This verifies permitted route use, not a complete algorithm.
P5: same-route cross-seed prediction is a genuine open test, not presumed success. It passes only if each of the two primary positive transfers has held individual normalized-response RMSE ≤.1.

## Developmental transfer, held individual predictions

Reference sources seeds 11 and 23; primary evaluation target seed37, cached and bypass. Target normal development states are allowed to fit its contrast. For each pair, align three stages by (a) identity depth, (b) linear CKA DTW, or (c) development native intervention profiles (Agent Q=+1, doses .25,.5,1). CKA is numerically clamped to [0,1] before distance use, fixing the previous study's numerical guard prospectively. DTW has diagonal/horizontal/vertical moves, extra .01 per nondiagonal move, covers both endpoints, fixed tie order.

All three alignment methods are assessed with the same developmental causal-disagreement score (RMS response-profile discrepancy along the path). Thus this is NOT a CKA-vs-causality route-classifier contest: CKA/depth choose paths, while every final route decision uses intervention information. Frozen calibration: source11→target23 cached positive and bypass negative. If positive+1e-7<negative, midpoint threshold is applied to four source11/23→target37 route pairs; otherwise no route classification is licensed. Tiny correlated pair counts are descriptive only.

Using the fixed path, source responses to each individual held Patient case at four new interior doses predict target responses for that case; average only source stages mapped to a given target stage. We do not average across cases for the primary prediction score. Source held intervention responses are measured inputs to this prediction; the test does not predict both models' responses from normal states alone. Target development intervention responses are available for causal alignment. All ordered pairs and self checks are reported; only two nonself same-route and two bypass transfers to target37 are primary. Same-seed cached self transfer is a control, not independent evidence. Agent/Patient score symmetry is explicit algebra, so held question transfer is a control, not learned generalization.

## Analysis, prior work, stopping

Independent NumPy reconstruction of all stored scalar intervention logits from saved weights must pass tolerance1e-8; inspect source freeze hashes. No retraining or outcome-dependent repair. Archive code, checkpoint arrays, raw outputs, all training/competence failures, controls and report on the existing research branch; coordinate Issue35. Analysis and plots may read saved data but may not generate new experimental conditions.

This builds on causal abstraction (Geiger et al., https://arxiv.org/abs/2106.02997), IIT (https://arxiv.org/abs/2112.00826), InterpBench (https://papers.nips.cc/paper_files/paper/2024/file/a8f7d43ae092d9a5295775eb17f3f4f7-Paper-Datasets_and_Benchmarks_Track.pdf), and Sun–Toneva Congruity (https://arxiv.org/abs/2603.30002). Known-algorithm calibration and interpretation comparisons are established. Congruity's implementation sampler is not reproduced; no score or superiority claim is permitted. The interpretation-illusion debate, including https://arxiv.org/abs/2401.12631, cautions that rejection under one perturbation class is not rejection of every interpretation. Dai et al.2024 binding edits (https://aclanthology.org/2024.emnlp-main.967/) remain relevant; this feature-coded study establishes no novel binding mechanism.

A clean assay with blind-coordinate errors is NARROW: measurement procedure failure under declared changes, not universal impossibility or paper novelty. Successful cross-seed prediction licenses this task/intervention distribution only. Failed prediction must be reported even if route classification passes. No subsequent trained-network or LLM study is authorized by a positive outcome of this run. Stop and report after this single experiment.
