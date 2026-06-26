---
description: Measure the abliteration dose-response curve and EC50 for a model's refusal direction.
argument-hint: "<model-name> <hf-id> <layer>"
---
Run the prayoga dose-response experiment for `$ARGUMENTS`:

```bash
docker exec -e PYTHONPATH=/workspace/prayoga/src -w /workspace/prayoga prayoga-gpu \
  python -m prayoga.axis_a.run_dose --model <short> --hf-id <hf-id> --layer <layer>
```

Report the REAL fit: EC50, slope, R², the ASR-vs-α curve, and confirm the
random-direction control stays flat. Interpret EC50 as the ablation strength at
half-maximal jailbreak ("vaśīkaraṇa-as-dose").
