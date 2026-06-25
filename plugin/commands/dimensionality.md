---
description: Measure the effective dimension of the refusal subspace (Arditi single-direction vs Marshall affine).
argument-hint: "<model-name> <hf-id> <layer>"
---
Run the prayoga dimensionality experiment for `$ARGUMENTS`:

```bash
docker exec -e PYTHONPATH=/workspace/prayoga/src -w /workspace/prayoga prayoga-gpu \
  python -m prayoga.axis_a.run_dimensionality --model <short> --hf-id <hf-id> --layer <layer>
```

Report the REAL effective dimension and the accuracy-after-removing-k curve.
Interpret: dim ≈ 1 ⇒ Arditi-like (single direction, addition steers); dim > 1 ⇒
Marshall-like (affine subspace, single-direction addition may fail).
