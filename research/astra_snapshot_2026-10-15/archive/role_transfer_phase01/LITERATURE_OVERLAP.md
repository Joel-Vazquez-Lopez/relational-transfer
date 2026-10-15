# Related-work boundary for the next experiment

Checked 6 October 2026. The claim descriptions below are supported by the linked primary abstracts/pages. Method-level novelty remains open until the full papers and follow-on work are reviewed.

| Work | Setting and contribution relevant here | Boundary for this project |
| --- | --- | --- |
| [Bogin, Gupta & Berant 2022](https://aclanthology.org/2022.emnlp-main.175/) | Sequence-to-sequence semantic parsing; unobserved local structures predict instance difficulty across five datasets, and example selection can improve generalization with limited training. | Instance-level prediction and selecting valuable training examples already exist. A future graph predictor or data-selection study must compare directly. |
| [Diwan et al. 2022](https://aclanthology.org/2022.emnlp-main.143/) | Winoground image–caption matching; fine-grained error analysis, probes, and evidence that cross-modal fusion is a challenge. | A generic claim that image/text alignment is the bottleneck has clear precedent. Our scenes are generated trajectories, not natural images. |
| [Yin et al. 2023](https://aclanthology.org/2023.acl-long.72/) | Consistency regularization improves Transformer compositional generalization in semantic parsing and machine translation; validation consistency can be informative. | A generic consistency loss or consistency predictor is not new. The exact transformation and comparisons matter. |
| [Yuksekgonul et al. 2023](https://arxiv.org/abs/2210.01936) | Tests relationships, attributes and order in vision–language models; composition-aware hard negatives improve performance. | Our observed-pair training already contains role reversals. The interesting question is why those negatives fail to guarantee transfer. |
| [An & Du 2026 preprint](https://arxiv.org/abs/2601.18858) | Representation homomorphism as a diagnostic and training target for Transformer compositional generalization. | “An internal diagnostic that predicts and improves generalization” is too broad a novelty claim. |
| [Zhou et al. 2023](https://aclanthology.org/2023.emnlp-main.898/) | Transformer generalization varies with training-data factors such as scale, pattern complexity and difficulty. | We must specify the relational/data factor beyond dataset size and benchmark it against simpler factors. |

The most defensible contribution to pursue is a **task-specific, controlled diagnosis** that distinguishes role encoding, retention and readout; an intervention chosen from that diagnosis; and independent validation across all unseen pairs. This is a working position, not a verified publication claim. The full methods, citations, and newer work must be checked before writing a novelty statement.
