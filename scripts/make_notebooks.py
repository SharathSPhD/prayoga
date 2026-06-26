"""Generate Colab-ready reproduction notebooks into notebooks/.

Two notebooks:
  00_explore_results.ipynb  — zero-dependency: pulls the committed aggregate JSON
      (findings.json, metrics.json) straight from GitHub and plots every headline
      result. Runs in seconds on a CPU Colab, no model, no GPU, no extra installs.
  01_reproduce_refusal_direction.ipynb — real reproduction on a Colab GPU: clones
      the repo, installs it, loads Gemma-2-2b (HF token), and reproduces F1
      (refusal-direction ablation/addition) with the actual prayoga code.

Run:  python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

RAW = "https://raw.githubusercontent.com/SharathSPhD/prayoga/main"
ND = Path("notebooks"); ND.mkdir(exist_ok=True)


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": list(lines)}


def nb(cells: list, gpu: bool = False) -> dict:
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], **({"gpuType": "T4"} if gpu else {})},
            "accelerator": "GPU" if gpu else "CPU"}
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}


# ---------------------------------------------------------------- 00 explore --
explore = [
    md("# prayoga — Explore the results (zero dependencies)\n",
       "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]",
       "(https://colab.research.google.com/github/SharathSPhD/prayoga/blob/main/notebooks/00_explore_results.ipynb)\n",
       "\n",
       "This notebook pulls **prayoga's committed aggregate results** straight from GitHub and ",
       "reproduces every headline figure. No model, no GPU, no installs — just `Runtime > Run all`.\n",
       "\n",
       "**Thesis:** LLM jailbreak, hypnotic suggestion, and tantric *vaśīkaraṇa* are three instances of ",
       "one mechanism — suppress the monitoring/refusal faculty, co-opt automatic generation. ",
       "Honest hardened headline: *asymmetry with a shared necessary core*."),
    code("import json, urllib.request\n",
         "import matplotlib.pyplot as plt, numpy as np\n",
         f"RAW = '{RAW}'\n",
         "def load(p):\n",
         "    return json.loads(urllib.request.urlopen(f'{RAW}/{p}').read())\n",
         "findings = load('site/public/data/findings.json')['findings']\n",
         "M = load('site/public/data/metrics.json')\n",
         "print(f'{len(findings)} findings; metrics: {list(M.keys())}')"),
    md("## F2 — Refusal abliteration dose–response (EC50)"),
    code("d = M['dose']; a = np.array(d['alphas'])\n",
         "plt.figure(figsize=(6,4))\n",
         "plt.scatter(a, d['asr_real'], color='#c0392b', zorder=3, label='refusal direction')\n",
         "plt.scatter(a, d['asr_random'], color='#7f8c8d', marker='s', label='random direction')\n",
         "plt.axvline(d['ec50'], ls='--', color='#2c3e50'); plt.text(d['ec50']+.02,.1,f\"EC50={d['ec50']:.2f}\")\n",
         "plt.xlabel('ablation strength α'); plt.ylabel('attack success rate'); plt.legend()\n",
         "plt.title(f\"Dose–response (R²={d['r2']:.3f})\"); plt.show()"),
    md("## F18 — Effective dimension is **layer-dependent** (qualifies F8)"),
    code("plt.figure(figsize=(6,4))\n",
         "for mdl, c in [('gemma-2-2b-it','#c0392b'),('qwen2.5-3b-it','#2980b9'),('gemma-2-9b-it','#27ae60')]:\n",
         "    if mdl in M.get('dimsweep',{}):\n",
         "        s = M['dimsweep'][mdl]; plt.plot(s['layers'], s['dim'], 'o-', ms=3, color=c, label=mdl)\n",
         "plt.xlabel('layer'); plt.ylabel('refusal-subspace effective dim')\n",
         "plt.title('Dimension varies across layers — the clean per-model contrast is layer-specific')\n",
         "plt.legend(); plt.show()"),
    md("## F11 — Refusal is a paraphrase-orbit invariant; injection breaks it"),
    code("sym = M['symmetry']; models = list(sym.keys()); x = np.arange(len(models)); w=.35\n",
         "fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,3.6))\n",
         "ax1.bar(x-w/2,[sym[m]['F_refusal'] for m in models],w,color='#c0392b',label='refusal dir')\n",
         "ax1.bar(x+w/2,[sym[m]['F_random'] for m in models],w,color='#7f8c8d',label='random')\n",
         "ax1.set_yscale('log'); ax1.set_xticks(x); ax1.set_xticklabels(models,fontsize=8); ax1.set_ylabel('orbit F-ratio (log)'); ax1.legend(); ax1.set_title('orbit invariance')\n",
         "ax2.bar(x-w/2,[sym[m]['m_plain'] for m in models],w,color='#2980b9',label='plain')\n",
         "ax2.bar(x+w/2,[sym[m]['m_injected'] for m in models],w,color='#e67e22',label='injected')\n",
         "ax2.set_xticks(x); ax2.set_xticklabels(models,fontsize=8); ax2.set_ylabel('order parameter m'); ax2.legend(); ax2.set_title('injection breaks it'); plt.show()"),
    md("## ṣaṭkarma — the six-act intervention taxonomy"),
    code("sk = M['satkarma']\n",
         "names=[r['sanskrit'] for r in sk]; eff=[r['effect'] for r in sk]; ctl=[r['control'] for r in sk]\n",
         "y=np.arange(len(names))[::-1]; h=.36\n",
         "plt.figure(figsize=(7,4))\n",
         "plt.barh(y+h/2,eff,h,color=['#c0392b' if r['separated'] else '#bbb' for r in sk],label='effect')\n",
         "plt.barh(y-h/2,ctl,h,color='#7f8c8d',label='control')\n",
         "plt.yticks(y,names); plt.xlabel('measured effect'); plt.legend(); plt.title('ṣaṭkarma as activation interventions'); plt.show()"),
    md("## The honest ledger — 18 findings across the three tiers"),
    code("from collections import Counter\n",
         "print('tier counts:', Counter(f['tier'] for f in findings))\n",
         "for f in findings:\n",
         "    print(f\"  {f['id']:4s} [{f['tier']:9s}] {f['verdict']:10s} {f['title'][:70]}\")"),
    md("**Read more:** [repo](https://github.com/SharathSPhD/prayoga) · ",
       "[site](https://sharathsphd.github.io/prayoga/) · the paper PDF is in `paper/paper.pdf`.\n",
       "\n*Dual-use:* only aggregate statistics are public; direction vectors and generations are withheld."),
]
(ND / "00_explore_results.ipynb").write_text(json.dumps(nb(explore), indent=1))

# ----------------------------------------------------- 01 reproduce F1 (GPU) --
repro = [
    md("# prayoga — Reproduce F1: refusal is a single direction (GPU)\n",
       "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]",
       "(https://colab.research.google.com/github/SharathSPhD/prayoga/blob/main/notebooks/01_reproduce_refusal_direction.ipynb)\n",
       "\n",
       "Reproduces the program's linchpin (**F1**) with the *actual* prayoga code on a Colab GPU: ",
       "extract the difference-in-means refusal direction in Gemma-2-2b, then **ablate** it ",
       "(suppress refusal → harmful compliance) and **add** it (induce over-refusal on benign prompts).\n",
       "\n",
       "Needs a **GPU runtime** (`Runtime > Change runtime type > T4 GPU`) and a Hugging Face token ",
       "with Gemma access (`from google.colab import userdata` → set `HF_TOKEN` in the secrets panel).\n",
       "\n",
       "⚠️ *Dual-use:* this produces an abliteration vector. Keep it for interpretability/safety ",
       "research; do not redistribute the vector or the harmful generations."),
    code("!git clone -q https://github.com/SharathSPhD/prayoga.git\n",
         "%cd prayoga\n",
         "!pip install -q -e . 2>/dev/null\n",
         "import numpy as np, os\n",
         "from google.colab import userdata\n",
         "os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')  # gated Gemma access"),
    code("from prayoga.lm.hf_model import HFModel\n",
         "from prayoga.axis_a.direction_extraction import directions_all_layers\n",
         "from prayoga.axis_a.intervention_engine import InterventionEngine\n",
         "from prayoga.axis_a.refusal import asr, refusal_rate\n",
         "load = lambda p: [l.strip() for l in open(p) if l.strip()]\n",
         "harmful, harmless = load('data/prompts/harmful.txt'), load('data/prompts/harmless.txt')\n",
         "model = HFModel('google/gemma-2-2b-it'); eng = InterventionEngine(model)\n",
         "print('loaded', model.n_layers, 'layers, d =', model.d_model)"),
    md("### Extract the refusal direction (layer 7) and run the causal gates"),
    code("L = 7\n",
         "d = directions_all_layers(model, harmful[:24], harmless[:24])[L]\n",
         "base = asr(eng.baseline_generate(harmful[24:], 32))\n",
         "abl  = asr(eng.ablate_generate(harmful[24:], d, 32))\n",
         "h = model.capture_all_layers_last_token(harmful[:24])[L]; s = model.capture_all_layers_last_token(harmless[:24])[L]\n",
         "sep = float(np.linalg.norm(h.mean(0)-s.mean(0)))\n",
         "over = refusal_rate(eng.add_generate(harmless[24:], d, 8*sep, L, 32))\n",
         "print(f'baseline harmful ASR     : {base:.2f}')\n",
         "print(f'ABLATION  → harmful ASR  : {abl:.2f}   (jailbreak via direction removal)')\n",
         "print(f'ADDITION  → over-refusal : {over:.2f}   (benign prompts now refused)')"),
    md("You should see baseline ASR ≈ 0, **ablation ASR ≈ 0.9** (refusal removed), and ",
       "**over-refusal ≈ 1.0** (benign prompts refused) — F1 reproduced. The program's other findings ",
       "(dose-response F2, dimensionality F8/F18, SAE F13, symmetry F11) live in `prayoga.axis_a` / ",
       "`prayoga.axis_b`; see the repo READMEs and `docs/FINDINGS.md`."),
]
(ND / "01_reproduce_refusal_direction.ipynb").write_text(json.dumps(nb(repro, gpu=True), indent=1))

print("wrote", *[p.name for p in ND.glob("*.ipynb")])
