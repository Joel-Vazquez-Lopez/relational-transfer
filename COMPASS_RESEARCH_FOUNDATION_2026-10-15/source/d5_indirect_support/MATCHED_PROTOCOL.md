# Fresh exact-pairing replication lock — 5 October 2026

The first six-pair feasibility run is complete. It found a large near-minus-far difference in the opposite direction from the original positive hypothesis. This follow-up is a **prospective replication of that observed reverse-direction pattern**, using new seeds and exactly paired rendered training schedules. It is not an independent discovery or a public preregistration.

- Source: `matched.py` SHA-256 `6e48ad19b7d54e2428db631b0560b320d43d539e4084b74a83eb4eb1036aee23`, importing frozen `run.py` SHA-256 `b12e4bf3e65770bdd39d1fb2640627eb76f6b96e6903f7efcb72f4c44c1ee03d`.
- New seeds: 8201–8206, both conditions per seed; 6,000 updates, batch 128, learning rate 0.0008, fixed final checkpoint.
- Training schedule: generate the same canonical episode in each condition, then exactly relabel identity one-hots and oracle endpoint symbols. Answer positions, candidate trajectories, distractor roles, and stochastic schedules are pairwise identical under relabeling. The audit verifies all 768 semantic pool entries and independently rendered evaluation examples.
- Held-out evaluation: the same literal `(0,1)` scenes in both conditions, 1,024 examples per model. Familiar evaluation: 1,024 canonically paired examples.
- Primary: six paired near-minus-far mean correct-minus-reverse held-out score margins. Report each seed, the mean/median, an exact two-sided sign-flip p-value, and paired bootstrap 95% interval. Call the reverse-direction pattern replicated only if all twelve familiar accuracies are at least 0.95, the mean contrast is at most −0.20 score units, and exact two-sided p ≤ 0.05. No exclusions for poor held-out performance.
- Held-out four-way accuracy and query separation are secondary. A positive, null, or mixed result will be reported without changing this rule.

Even a successful replication would concern this one graph's combined indirect-support configuration; common-neighbour count, shortest path, and other local differences are not separately identified.
