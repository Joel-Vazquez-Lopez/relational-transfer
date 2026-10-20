# B-D32 frozen discovery protocol: which listener branch must learn?

Status: **discovery only**. Selected after B-D31 found source/target additive event-query structure already at random initialization, while training rotated query directions and produced opposite candidate-side cue/relation sensitivity.

Use exactly B-D21 frozen NO_CUE/CUE data, oracle messages, seeds3801/3802, initialization, batch-index sequence, AdamW rate .0008, batch128, 3000 updates, and 4-choice loss. Speaker/emit remain frozen. Cross two worlds with two branch interventions: **query fixed**, freezing sym/mpos/mcls/msg_blocks/msg_norm at their seeded initialization while candidate encoder trains; and **candidate fixed**, freezing candidate encoder while query parameters train. There are eight new discovery-only models. Save final checkpoints. Full B-D21 trained models serve as reference, not refits.

Evaluate independent four-choice accuracy and the same 600-scene relation×cue score grid at steps1000 and3000 (save those checkpoints). Report query and candidate branch parameter hashes before/after to verify the freeze. If a branch-fixed model fails ordinary task accuracy, do not infer a unique mechanism: branch capacity, optimizer, and untrained feature quality differ. If both branch-fixed models succeed, test whether the same experience-directed cue/relation extension survives; that would strongly weaken claims that learning one particular encoder operation is necessary.

No model or canonical artifact overwritten. No protected/confirmation data.
