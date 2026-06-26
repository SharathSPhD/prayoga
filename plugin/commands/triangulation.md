---
description: Run the X-1 cross-axis triangulation (Axis A/B/C on the same prompts) — aggregate-safe.
argument-hint: "<model-name> <hf-id> <layer>"
---
Run the prayoga X-1 cross-axis triangulation for `$ARGUMENTS` after confirming `/prayoga:status`:

```bash
docker exec -e PYTHONPATH=/workspace/prayoga/src -w /workspace/prayoga prayoga-gpu \
  python -m prayoga.axis_x.run_triangulation --model <short> --hf-id <hf-id> --layer <layer>
```

Measures, per prompt under one injection, Axis A (refusal order parameter), Axis B
(precision margin), Axis C (dormant distance, descriptive), and the behavioral flip,
pooling several injection families. Report only aggregate scalars: the A↔B coupling
(correlation + partial correlation), the group-centered predictive AUC vs the
random-direction control and label-shuffle null, and per-family flip rates. Do NOT print
harmful prompt text, injection payloads, direction vectors, probes, or raw completions.

Tier discipline: the A↔B coupling is partly a measurement consequence (both are linear
refusal readouts of the same residual), NOT proof of unification. The behavioral keystone
is **inconclusive** under the substring refusal metric (it mis-scores deflections as
compliance) — flag this and recommend a content-faithful judge. No machine-state claim.
