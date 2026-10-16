# COMPUT-017 — V5 undirected co-occurrence falsification

Independent replay, 16 October 2026. Original Mira V5 ZIP SHA256: `01c1b682e6f3810feb00c67023121f7714cb583796e6527c8f8e174cee0580e9`. Source CRC, Mira five tests and independent validator passed. New baseline and five regression tests executed against the exact 26,496 open synthetic items.

**Finding:** An undirected **name–verb co-occurrence graph** answers **26,496/26,496** without ever parsing subject/object, grammatical voice, actor/patient direction, gold, generator bits or IDs. It uses visible predicate names and joins neighbor sets (so it is not a bag-of-words baseline). All 48 name families, all 3 sentence styles, both unsealed splits pass: chain2 1152/1152, chain3 4608/4608, chain4 18432/18432, conjunctive 2304/2304. Zero abstentions.

**Scope:** Dataset structure uses disjoint name layers and matching predicates, so the inverse name–verb adjacency is sufficient. This challenges V5 as a **direction-sensitive semantic-role discriminator**, NOT its genuine predicate-conditioned traversal difficulty or the possibility of pretrained role computation. No LLM was run, no unseen natural language or confirmatory benchmark was tested.

**Missing discriminator:** `Alice warned Bob. Bob warned Carol. Dana thanked Alice.` Query `Who thanked the person who warned Bob?` has directed answer `Dana`, but undirected adjacency at `Bob` produces two neighbors and abstains. Include reverse-direction same-verb collisions, independently adjudicated roles, and competence-matched surface parsers in a successor challenge. The data have 48 clusters; 26,496 dependent prompts are not independent observations.

**Reproducibility:** the user received `COMPUT_017_MIRA_V5_UNDIRECTED_FALSIFICATION.zip`, source-locked Python scripts, five additional passing tests, results JSON and full report. The package excludes private raw prompts and gold.