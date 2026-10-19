# Fixed factorial experiment: host restart, partial data preserved

## What happened

The authorized local Qwen run completed **248 outputs** and journaled attempt **249**, then the Mac restarted. The boot time was 22:49:46 on 19 October 2026 (Europe/Stockholm). A new macOS panic report recorded at 22:50:11 reports **Kernel data abort**. The inference process is gone and its execution session is no longer available. No Python traceback or runner-generated ERROR.json was saved.

The panic report names `com.apple.appkit.xpc.openAndSave` as the panicked task. Its compressor summary marks its page/segment limits and swap space OK. Earlier resource observations showed high swap use, but **neither memory pressure nor Qwen is established as the cause of the kernel panic**. Disk space was about 16 GiB free before the restart and about 30 GiB afterward; this was not an observed disk-reserve failure. The full system diagnostic was not copied into the research repository; only this minimal summary is retained.

## Exact scope of the partial run

|Item|Count|
|---|---:|
|This execution's attempted forwards|249|
|Completed fresh payloads|248|
|Completed scientific baseline calls|246|
|Completed validation calls|2|
|Held-out outputs|0|
|Causal intervention outputs|0|
|All investigation attempts, including earlier failures|252|
|All completed outputs, including two earlier exposed-fixture preflights|250|
|Authorized combined attempt cap|1120|

No development candidate was frozen and no held-out family was evaluated. Full-panel wrapper/repeat/invariance/sham/disconnection gates were not reached. This is **not a completed experiment or a test of the competing scientific explanations**. No hypothesis is accepted, rejected or selected from the partial answers.

## Preservation and audit

PARTIAL_ARCHIVAL_AUDIT.json verifies all committed partial payload hashes, full-vector target scores, argmax classes, trace hashes/shapes and their journal joins. This validates the saved bytes and scores, not the unexecuted scientific/technical gates. The recorded bytes for call 249 are retained separately as an uncommitted output fragment and excluded from results. PARTIAL_PAYLOAD_BUNDLES.json maps every original NPZ byte sequence into verified tar bundles. Expanded originals remain locally intact.

The scientific sources, input passages, AI-only source annotations, intervention masks, thresholds, scoring and hypotheses remain unchanged. The pre-output restart freeze is [ed334bd](https://github.com/Joel-Vazquez-Lopez/relational-transfer/commit/ed334bd999ce580b39f2a079225a3035dd185378). RESTART_AMENDMENT.json explicitly recorded the one unfinished validation reattempt and cap adjustment before any fresh output. Both prior failed runs remain preserved.

No RESULTS.json, completed-outcome figure or prospective scientific decision was generated. `write_report.py` and the frozen analysis/plotting scripts are available but were not invoked as if collection had completed.

## Scientific interpretation and prior work

The planned question remains whether construction-dependent score gaps track source voice, question voice or literal output coding, with a separate fixed routing diagnostic. It is unanswered by this interrupted collection. There is no new event-selection or binding mechanism to compare favorably with [Dai et al. (2024)](https://aclanthology.org/2024.emnlp-main.967/), whose study identifies and intervenes on a binding-ID subspace. Prompt/answer bias remains relevant background ([Zhao et al., 2021](https://proceedings.mlr.press/v139/zhao21c.html)); interpretation of interventions remains sensitive to methodological choices ([Zhang and Nanda, 2024](https://arxiv.org/abs/2309.16042)). This run establishes no novelty or population significance.

**Conclusion: technical stop after a host restart; no scientific decision. No further inference or automatic retry.** Diagnose system stability before considering another execution. Earlier completed studies and their failed mechanistic gates remain unchanged.
