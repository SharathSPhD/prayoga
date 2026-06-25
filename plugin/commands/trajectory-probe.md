---
description: Run the aggregate-safe multi-turn refusal trajectory diagnostic.
argument-hint: "<model-name> <hf-id> <layer> <direction-json>"
---
Run the prayoga trajectory probe for `$ARGUMENTS` after confirming `/prayoga:status`:

```bash
docker exec -e PYTHONPATH=/workspace/prayoga/src -w /workspace/prayoga prayoga-gpu \
  python -m prayoga.axis_b.run_trajectory --model <short> --hf-id <hf-id> \
  --layer <layer> --direction <direction-json>
```

Report only redacted turn labels, refusal order parameter per turn, precision proxy
if available, refusal flag, hallucination proxy, and controls. Do not print harmful
turn text, injection payloads, direction vectors, or raw completions. Interpret
trajectory collapse as an analogy-tier monitor-collapse diagnostic unless tied to
the measured residual order parameter.
