# Independent-family development result: Llama3.2 3B

All1,728 prospectively frozen local prompts completed. The additional192-prompt, prospectively fixed no-prefill control also completed. Original data, source locks, local completion hashes and closed archive checks passed. No natural-language review set or sealed outcomes were used.

Both actor and patient answers correct, out of12 events per cell:

| Construction | Lexical questions | Explicit role descriptions | No-prefill lexical | No-prefill explicit |
|---|---:|---:|---:|---:|
| Active |11|12|12|11|
| Passive |8|10|12|9|
| Subject cleft |12|12|12|12|
| Object cleft |2|0|4|0|

Object-cleft exact-name compliance is24/24 responses per wording/interface. Thus the joint failures in these cells are not malformed-output errors. In the explicit wording, actor accuracy is0/12 and patient accuracy12/12, with and without prefill: all24 answers select the patient, which is also the first-mentioned participant. This simple conditional heuristic explains this complete cell. It does not establish a hidden role-binding mechanism. Lexical cells are less uniform.

The object-cleft question-score contrast reverses on1/12 lexical and9/12 explicit events with prefill, and4/12 and11/12 without. Those score signs and generated decisions are different measurements; reversal is not equivalent to both answers being wrong. Compare signs, not raw log-probability magnitudes between tokenizers.

Explicit wording loses2 previously jointly correct events and gains none with prefill; the paired difference is−16.7pp, six-name-pair cluster bootstrap95% interval[−33.3,0]. Without prefill, it loses4 and gains none:−33.3pp[−50,−16.7]. Only six reused participant pairs support these intervals. All-zero bootstrap cells yield degenerate intervals, which do not establish zero population accuracy or remove sampling uncertainty.

Changing candidate order flips21/48 stronger object-cleft responses, versus8/48 active,16/48 passive,8/48 subject-cleft responses. These repeated prompts are not independent events.

Qwen7B improves from3/12 to8/12 object-cleft joint correctness under explicit wording; Phi3 declines9/12 to1/12, and Llama declines2/12 to0/12. These exploratory patterns undermine the claim that one wording is universally stronger. Models differ in family, size, native templates and quantization details; this is not an isolated causal scaling effect.

The Gemma API panel remains separate until complete. Its public alias cannot be treated as a pinned weight revision, and it lacks our arbitrary candidate-score access. The local no-prefill bridge checks one interface difference, not all serving differences.

**Next scientific gate:** complete the fixed Gemma panel, then select fresh independently checked natural-language examples that separate patient identity, first mention and grammatical roles. A new intervention or mechanistic claim is not justified by these repeated synthetic examples alone.
