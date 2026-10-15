# User-requested early peek — 14 October 2026

This is a **partial development read** on the first previously examined graph family only, made after the protocol and code were frozen. It does not change hypotheses, losses, graph families, seeds, checkpoint selection, analysis, or stopping rules. All 160 models remain scheduled.

At 6,000 updates and scene suite 260107, family 0's participant-only high-minus-sham focal binary-role penalty was −25.98 percentage points across two paired seeds. The ordered-role penalty was −8.59 points, yielding +17.38 points attenuation for this one family. Familiar accuracy on the four sampled observed pairs was at least 99.80% in these four cells. Ordered-role models also had higher nonfocal unseen accuracy on average, so this partial difference cannot establish a structural interaction beyond general performance improvement. Seven independent graph families and the performance-matching analysis are still pending.

Raw values were read from `full/family0/{sham,high}/seed_{321001,321002}/{participant_only,ordered_roles}/checkpoint_6000.json`. No role-swap or compute-control conclusion was inspected for this peek.

## Second user-requested peek, before leaving the computer

At this time 58/160 models were complete; **only families 0 and 1** had every arm, condition, and paired start complete. The fixed 6,000-update, scene-260107 contrasts below are descriptive development values:

| Family | Participant-only high−sham penalty | Ordered-role high−sham penalty | Ordered−participant attenuation | Role-swap−compute-control attenuation |
| --- | ---: | ---: | ---: | ---: |
| 0 | −25.98 pp | −8.59 pp | +17.38 pp | +5.57 pp |
| 1 | −24.22 pp | −0.20 pp | +24.02 pp | −10.55 pp |

Ordered-role training has higher nonfocal unseen accuracy in both completed families (family 0: 94.95% versus 89.06%; family 1: 95.68% versus 86.04%). Thus the early attenuation **cannot yet be separated from general performance improvement**. The role-swap arm does not show a consistent benefit across these two families. These partial outcomes do not alter the frozen protocol, model queue, analysis, or stopping rules; six families and the performance-matched analysis remain pending.

## Third user-requested peek, 74/160 models complete

Three of eight families (0–2) now have all arms, conditions, and paired starts complete. Ordered-role versus participant-only attenuation is +17.38, +24.02, and +40.14 percentage points in those families. The corresponding participant-only high-minus-sham penalties are −25.98, −24.22, and −41.21 points; ordered penalties are −8.59, −0.20, and −1.07 points. The role-swap versus permutation-control attenuation is mixed: +5.57, −10.55, and 0.00 points. Ordered-role nonfocal unseen accuracy exceeds participant-only by 5.89, 9.64, and 15.13 points, respectively. This is a major remaining rival explanation, and ordered focal scores approach ceiling in some cells. Five families, performance matching, and complete familiar/all-pair diagnostics remain pending. No method or stopping rule changed.

## Fourth user-requested peek, 88/160 models complete

Four of eight families (0–3) now have all arms, graph conditions, and paired starts complete. The ordered-role versus participant-only attenuation in these families is +17.38, +24.02, +40.14, and +23.73 percentage points (mean +26.32). The new family 3 has a participant-only high-minus-sham penalty of −31.05 points versus an ordered-role penalty of −7.32 points. Its ordered-role nonfocal unseen accuracy is 94.46% versus 80.10% for participant-only. Ordered-role nonfocal accuracy is higher in all four completed families, by 5.89, 9.64, 15.13, and 14.36 points, so general performance and ceiling remain serious rival explanations. Role-swap versus permutation-control attenuation across the four families is +5.57, −10.55, 0.00, and −1.37 points: no consistent direction. The ledger reports 88 successful completions and no failed models. These are descriptive development outcomes, not a final analysis or independent confirmation. Four families, performance matching, and full-pair diagnostics remain pending. The frozen protocol and model queue are unchanged.

## Fifth user-requested peek, after family 4 completed

Five of eight families (0–4) have all 20 models complete. Ordered-role versus participant-only attenuation in these families is +17.38, +24.02, +40.14, +23.73, and +27.25 percentage points. In the new family 4, participant-only high-minus-sham focal binary-role penalty is −27.73 points, while ordered-role penalty is −0.49 points. Ordered-role nonfocal unseen binary-role accuracy is 97.29% versus 82.59% for participant-only. The role-swap versus compute-control attenuation in family 4 is −0.78 points, extending the absence of a consistent directional advantage. Ordered-role nonfocal accuracy is higher in every complete family and focal scores approach ceiling in some cells. Therefore these partial development results do not isolate a structural-sensitivity effect from general model quality or ceiling. Three families and the frozen performance-matched and all-pair analyses remain pending. No protocol or method changed.
