---
description: Active-inference discovery of the refusal circuit over SAE features (vs greedy/random).
argument-hint: "<model-name> <hf-id> <layer>"
---
Run the prayoga active-inference circuit-discovery experiment for `$ARGUMENTS`:

```bash
docker exec -e PYTHONPATH=/workspace/prayoga/src -w /workspace/prayoga prayoga-gpu \
  python -m prayoga.axis_a.run_active --model <short> --hf-id <hf-id> --layer <layer>
```

Trains a BatchTopK SAE, then builds an ablation "circuit" under an intervention
budget using an Expected-Free-Energy acquisition (pragmatic harmful-gap prior ×
epistemic diversity), compared against greedy and random. Report the REAL
interventions-to-target for each strategy and the final ASR. Interpret: guided
search is dramatically more sample-efficient than random; its advantage over greedy
is gated by circuit dimensionality (negligible for a low-dimensional refusal core).
