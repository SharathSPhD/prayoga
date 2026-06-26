# prayoga — Colab notebooks

One-click, GitHub-connected notebooks. They rely only on what Colab ships
(torch / transformers / matplotlib) plus the repo itself — no custom environment.

| Notebook | Runtime | What it does |
|---|---|---|
| [**00 · Explore the results**](https://colab.research.google.com/github/SharathSPhD/prayoga/blob/main/notebooks/00_explore_results.ipynb) | CPU, seconds | Pulls the committed aggregate JSON (`findings.json`, `metrics.json`) straight from GitHub and reproduces every headline figure — dose–response/EC50 (F2), the layer-sweep dimensionality (F18), the orbit-invariance measurement + injection collapse (F11), the ṣaṭkarma taxonomy, and the public findings ledger. No model, no GPU, no installs. |
| [**01 · Reproduce F1 (refusal direction)**](https://colab.research.google.com/github/SharathSPhD/prayoga/blob/main/notebooks/01_reproduce_refusal_direction.ipynb) | **GPU (T4)** + HF token | Clones the repo, installs it, loads Gemma-2-2b, and reproduces the linchpin finding with the actual prayoga code: extract the difference-in-means refusal direction, **ablate** it (jailbreak), **add** it (over-refusal). |
| [**02 · Agentic symmetry breaking**](https://colab.research.google.com/github/SharathSPhD/prayoga/blob/main/notebooks/02_agentic_symmetry_breaking.ipynb) | CPU, seconds | Aggregate-safe walkthrough of the multi-turn trajectory diagnostic. Uses redacted turn labels and public registry data only; no harmful prompts, vectors, or model completions. |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/prayoga/blob/main/notebooks/00_explore_results.ipynb)

**Dual-use:** notebooks 00 and 02 use only public aggregate statistics. Notebook 01
produces an abliteration vector locally in your session — keep it for
interpretability/safety research; do not redistribute the vector or the harmful
generations. Direction vectors and generations are never committed to this repo.

Regenerate with `python scripts/make_notebooks.py`.
