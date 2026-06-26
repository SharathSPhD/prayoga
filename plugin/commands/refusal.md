---
description: Extract the refusal direction for a HuggingFace chat model and run the ablation/addition causal gates (Arditi method).
argument-hint: "<model-name> <hf-id> [layer]"
---
Run the prayoga refusal-direction experiment on the model given in `$ARGUMENTS`
(format: `<short-name> <hf-id> [layer]`, e.g. `gemma-2-2b-it google/gemma-2-2b-it`).

Execute it inside the GPU container and report the REAL result (do not fabricate):

```bash
docker exec -e PYTHONPATH=/workspace/prayoga/src -w /workspace/prayoga prayoga-gpu \
  python -m prayoga.axis_a.run_direction --model <short> --hf-id <hf-id>
```

Then summarize: baseline refusal rate, ablation ASR Δ (with CI), the 10-direction
random control, the addition over-refusal Δ, and whether all gates PASS. Note the
selected layer. Remind the user that the raw direction vector is dual-use and stays
git-ignored.
