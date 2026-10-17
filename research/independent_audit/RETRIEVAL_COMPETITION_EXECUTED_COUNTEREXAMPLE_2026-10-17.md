# Independent executed retrieval-competition counterexample (2026-10-17)

## Question
Can wrong-event participant borrowing increase after adding irrelevant names even if role bindings remain intact?

## Executed local toy simulation
Self-contained Python, fixed RNG seeds, 20,000 trials per condition. Two records have distinct, permanently correct Agent/Patient fields. Query targets event 2. A noisy score chooses event; the selected event's requested role is returned without role corruption. In the bystander condition only, a positive random salience increment is added to the competing event. This increment is an **assumed mechanism**, not inferred from LLM outputs.

- Plain: 19,971/20,000 correct (99.855%); 29 wrong-event selections; zero bystander answers.
- With bystanders: 18,126/20,000 correct (90.63%); 1,874 wrong-event selections; zero bystander answers.
- Separate deterministic salience sweep (20,000 each): competitor boosts 0, 0.5, 1.0, 1.5 yield wrong-event rates 0.12%, 1.515%, 9.195%, 31.095%.

## Interpretation
**Existence counterexample only.** A role-preserving retrieval system can exhibit bystander-induced wrong-event borrowing and never answer with a bystander. Thus the qualitative signature alone cannot identify a binding failure. The simulation does NOT reproduce the Qwen model, establish that names increase competitor salience, estimate an empirical effect size, or explain the observed seven errors. The chosen parameters are illustrative, not fitted. A model with a different salience mechanism could behave differently.

## Next empirical discriminator
Acquire frozen 64-item raw prompts/outputs with authorized provenance and independently compare a competent joint lexical+syntax+position+salience retrieval predictor with role-corruption alternatives on heldout cases. Preserve the preregistered failure (7 observed versus 8 required), source rights and preflight #31. No new pretrained inference was run.

## Reproduction
```python
import random
N=20000
def trial(load,rng):
    target=1.8+rng.gauss(0,.42)
    competitor=rng.gauss(0,.42)
    if load: competitor+=rng.expovariate(1/.65)
    records=[('Ada','Ben'),('Cara','Dan')]
    return records[int(target>=competitor)][rng.randrange(2)] in records[1]
for load in (False,True):
    rng=random.Random(71)
    k=sum(trial(load,rng) for _ in range(N))
    print(load,k,N)
```
Expected: False 19971 20000; True 18126 20000.

**Note:** This uses a deliberately simplified generative mechanism and is a logical non-identifiability demonstration, not empirical evidence of how pretrained models retrieve events.