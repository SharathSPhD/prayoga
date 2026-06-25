---
description: Preflight the local prayoga GPU/container environment before running dual-use experiments.
argument-hint: ""
---
Run a local prayoga preflight before any experiment:

```bash
docker ps --format '{{.Names}}' | grep -q '^prayoga-gpu$' && \
docker exec -e PYTHONPATH=/workspace/prayoga/src -w /workspace/prayoga prayoga-gpu \
  python - <<'PY'
from pathlib import Path
import json

checks = {
    "repo_mounted": Path("/workspace/prayoga/src/prayoga").exists(),
    "registry_present": Path("/workspace/prayoga/data/findings_registry.json").exists(),
    "public_findings_present": Path("/workspace/prayoga/site/public/data/findings.json").exists(),
    "results_private_path_exists": Path("/workspace/prayoga/results").exists(),
}
print(json.dumps(checks, indent=2))
PY
```

Report whether the container is running, the repo is mounted, the registry exists,
and public aggregate data is present. Do not print raw vectors, harmful prompts, or
private result payloads.
