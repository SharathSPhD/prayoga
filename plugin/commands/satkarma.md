---
description: Run the ṣaṭkarma intervention taxonomy (six tantric acts as activation interventions) on a model.
argument-hint: "<model-name> <hf-id> <layer>"
---
Run the prayoga ṣaṭkarma taxonomy for `$ARGUMENTS`:

```bash
docker exec -e PYTHONPATH=/workspace/prayoga/src -w /workspace/prayoga prayoga-gpu \
  python -m prayoga.satkarma.run_satkarma --model <short> --hf-id <hf-id> --layer <layer>
```

Report the REAL six-act table (vaśīkaraṇa, śānti, stambhana, vidveṣaṇa, uccāṭana,
māraṇa), each with its measured effect vs control and whether it separates. State
how many of six control-separate and that this is the honest test of the taxonomy,
not an assumption.
