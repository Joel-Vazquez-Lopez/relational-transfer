# Infrastructure repairs before freeze

The initial reference preflight encountered two ordinary setup issues, both before N4 evaluation:

1. Chex imported toolz, absent from the initial isolated dependency list. Added pinned toolz1.0.0 to the isolated runtime.
2. The published InterpBench conversion assumes its Tracr fork exposes `output_encoder.basis`; official pinned Tracr exposes `encoding_map`. The adapter now locates the explicitly named Boolean sink coordinates in the residual labels. No intervention site, semantic prediction, scoring or task was selected using N4 outcomes.

The final N3 preflight matches upstream intermediate states, and its two declared semantic predictions pass. The N3 sampler preflight passes54 native/augmented output comparisons. Two unrelated-array calls validate the released classifier API. These setup runs are not held-out research results. The final preflight logs are retained; this note preserves the earlier errors whose working logs were replaced during repair.

The first shell request for public metadata hit the network sandbox; the approved read-only network request succeeded. No private SCP connection was attempted. `hf` was not on PATH; public checkpoint metadata was read through the unauthenticated Hub API. No checkpoint weights were downloaded.
