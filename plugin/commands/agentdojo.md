---
description: Run the real AgentDojo agentic injection benchmark against Claude via claude -p.
argument-hint: "<suite=banking> <n-user> <n-inj>"
---
Run the prayoga AgentDojo behavioral evaluation for `$ARGUMENTS` (Tier-1, no GPU):

```bash
.venv-ad/bin/python scripts/agentdojo_run.py --suite <suite> --n-user <n> --n-inj <n> \
    --attack important_instructions --out results/tier1/agentdojo.json
```

Drives the real AgentDojo benchmark through a custom `claude -p` prompted-tool-calling
adapter (reuses AgentDojo's `<function=…>` prompt + parser). Report the REAL task
utility and attack-success rate over the user×injection rollouts. Interpret as the
behavioral "sophisticated reference end": a high-utility agent that nonetheless resists
injection — the contrast to white-box small-model fragility. See `docs/agentdojo_demo.md`.
