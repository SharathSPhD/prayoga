---
description: Measure the refusal symmetry order parameter (paraphrase-orbit invariance + injection symmetry-breaking).
argument-hint: "<model-name> <hf-id> <layer>"
---
Run the prayoga symmetry order-parameter experiment for `$ARGUMENTS`:

```bash
docker exec -e PYTHONPATH=/workspace/prayoga/src -w /workspace/prayoga prayoga-gpu \
  python -m prayoga.axis_b.run_symmetry --model <short> --hf-id <hf-id> --layer <layer>
```

Report the REAL F-ratio (refusal dir vs random), the harmful/harmless order-parameter
means, and the order-parameter collapse under injection. Interpret: a high F-ratio
means refusal is a paraphrase-orbit INVARIANT; the collapse under injection is
measured symmetry-breaking ("jailbreak = symmetry-breaking").
