"""prayoga — interactive results explorer (Hugging Face Space).

Zero-model, aggregate-only demo of the prayoga research program: refusal-suppression
as a cross-domain mechanism (LLM jailbreak / hypnosis / tantric vaśīkaraṇa). All numbers
are real measurements loaded from the committed aggregate JSON; no model weights, no
harmful content, no direction vectors are shipped here (dual-use safety gate).
"""
import json
from collections import Counter

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

M = json.load(open("metrics.json"))
F = json.load(open("findings.json"))["findings"]
AD = json.load(open("agentdojo_demo.json"))
RED, BLUE, GREY, GREEN, ORANGE = "#c0392b", "#2980b9", "#7f8c8d", "#27ae60", "#e67e22"


def fig_dose():
    d = M["dose"]; a = np.array(d["alphas"]); fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.scatter(a, d["asr_real"], color=RED, zorder=3, label="refusal direction")
    ax.scatter(a, d["asr_random"], color=GREY, marker="s", label="random direction")
    ax.axvline(d["ec50"], ls="--", color="#2c3e50"); ax.text(d["ec50"] + .02, .1, f"EC50={d['ec50']:.2f}")
    ax.set_xlabel("ablation strength α"); ax.set_ylabel("attack success rate (ASR)")
    ax.set_title(f"F2 · Refusal abliteration dose–response (R²={d['r2']:.3f})"); ax.legend()
    fig.tight_layout(); return fig


def fig_dim():
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for mdl, c in [("gemma-2-2b-it", RED), ("qwen2.5-3b-it", BLUE), ("gemma-2-9b-it", GREEN)]:
        if mdl in M.get("dimsweep", {}):
            s = M["dimsweep"][mdl]; ax.plot(s["layers"], s["dim"], "o-", ms=3, color=c, label=mdl)
    ax.set_xlabel("layer"); ax.set_ylabel("refusal-subspace effective dim")
    ax.set_title("F18 · Effective dimension is layer-dependent"); ax.legend()
    fig.tight_layout(); return fig


def fig_sym():
    sym = M["symmetry"]; models = list(sym.keys()); x = np.arange(len(models)); w = .35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax1.bar(x - w / 2, [sym[m]["F_refusal"] for m in models], w, color=RED, label="refusal dir")
    ax1.bar(x + w / 2, [sym[m]["F_random"] for m in models], w, color=GREY, label="random")
    ax1.set_yscale("log"); ax1.set_xticks(x); ax1.set_xticklabels(models, fontsize=8)
    ax1.set_ylabel("orbit F-ratio (log)"); ax1.legend(); ax1.set_title("Paraphrase-orbit invariance")
    ax2.bar(x - w / 2, [sym[m]["m_plain"] for m in models], w, color=BLUE, label="plain")
    ax2.bar(x + w / 2, [sym[m]["m_injected"] for m in models], w, color=ORANGE, label="injected")
    ax2.set_xticks(x); ax2.set_xticklabels(models, fontsize=8); ax2.set_ylabel("order parameter m")
    ax2.legend(); ax2.set_title("F11 · Injection breaks the symmetry")
    fig.tight_layout(); return fig


def fig_satkarma():
    sk = M["satkarma"]; names = [r["sanskrit"] for r in sk]
    eff = [r["effect"] for r in sk]; ctl = [r["control"] for r in sk]
    y = np.arange(len(names))[::-1]; h = .36; fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(y + h / 2, eff, h, color=[RED if r["separated"] else "#bbb" for r in sk], label="effect")
    ax.barh(y - h / 2, ctl, h, color=GREY, label="control")
    ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlabel("measured effect")
    ax.set_title("ṣaṭkarma · six-act intervention taxonomy"); ax.legend()
    fig.tight_layout(); return fig


def findings_table(tier):
    rows = [f for f in F if tier == "all" or f["tier"] == tier]
    return [[f["id"], f["tier"], f["verdict"], f["title"]] for f in rows]


def fig_agentdojo_grid():
    grid = AD["grid"]
    users = sorted({g["user"] for g in grid}); injs = sorted({g["injection"] for g in grid})
    Z = np.full((len(users), len(injs)), np.nan)
    for g in grid:
        Z[users.index(g["user"]), injs.index(g["injection"])] = 1.0 if g["utility"] else 0.0
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    im = ax.imshow(Z, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(injs))); ax.set_xticklabels([i.replace("injection_task_", "inj ") for i in injs], fontsize=8, rotation=30)
    ax.set_yticks(range(len(users))); ax.set_yticklabels([u.replace("user_task_", "user ") for u in users], fontsize=8)
    for r in range(len(users)):
        for c in range(len(injs)):
            ax.text(c, r, "✓" if Z[r, c] == 1 else "·", ha="center", va="center", fontsize=9, color="#333")
    ax.set_title("Banking: task utility per user×injection — attack success = 0 in every cell")
    fig.colorbar(im, ax=ax, label="task utility (green=done)"); fig.tight_layout()
    return fig


def format_trace(key):
    tr = AD["traces"].get(key)
    if not tr:
        return ""
    tag = {"system": "🛠️ **System**", "user": "🧑 **User**", "assistant": "🤖 **Claude**", "tool": "📦 **Tool result**"}
    out = [f"**{key}** — utility={tr['utility']}, attack_success={tr['attack']} · _{tr['label']}_\n"]
    for m in tr["trace"]:
        head = tag.get(m["role"], m["role"])
        if m["tools"]:
            head += f"  ·  ⚙️ tool call: `{', '.join(m['tools'])}`"
        body = (m["text"] or "").strip()
        out.append(f"{head}\n\n> {body[:600].replace(chr(10), chr(10)+'> ')}\n")
    return "\n".join(out)


TRACE_KEYS = list(AD["traces"].keys())

THESIS = """
# prayoga — refusal-suppression as a cross-domain mechanism

**Thesis.** LLM jailbreak/prompt-injection, hypnotic suggestion, and tantric *vaśīkaraṇa*
are three instances of **one** mechanism: capture a system's output policy by injecting a
context that **suppresses its monitoring/refusal faculty** while **co-opting automatic
generation**. Three claim-tiers are kept strictly separate — **mechanism** (empirical),
**analogy** (functional), **metaphor** (falsifiable).

**Hardened headline (after adversarial review):** *asymmetry with a shared necessary core* —
refusal is **one** mechanism at the necessary/ablatable level (transfers across model families)
and **many** at the sufficient/dimensional level (model-specific).

*This Space ships only public aggregate statistics — no model weights, no direction vectors,
no harmful content.* · [Code](https://github.com/SharathSPhD/prayoga) ·
[Site](https://sharathsphd.github.io/prayoga/)
"""

with gr.Blocks(title="prayoga", theme=gr.themes.Soft(primary_hue="red")) as demo:
    gr.Markdown(THESIS)
    with gr.Tab("Mechanism · dose–response"):
        gr.Markdown("Refusal suppression is a smooth, **dosable** function of how much of a single "
                    "residual direction is removed — a half-maximal effect at EC50≈0.33, with a flat "
                    "random-direction control.")
        gr.Plot(fig_dose)
    with gr.Tab("Dimensionality"):
        gr.Markdown("A full **layer-sweep**: the refusal subspace's effective dimension is "
                    "layer-dependent — the clean per-model contrast (Gemma 1, Qwen 3) is specific "
                    "to the extraction layer, not a global constant.")
        gr.Plot(fig_dim)
    with gr.Tab("Symmetry"):
        gr.Markdown("Refusal is a **paraphrase-orbit invariant** (high F-ratio vs a random direction) "
                    "and **injection collapses** the order parameter $m=(h\\cdot\\hat d)/\\lVert h\\rVert$ — "
                    "the measured content of *jailbreak = symmetry-breaking*.")
        gr.Plot(fig_sym)
    with gr.Tab("vaśīkaraṇa · ṣaṭkarma"):
        gr.Markdown("The six tantric acts read as a **taxonomy of activation interventions**; the "
                    "policy-capture acts separate cleanly from their controls (the mathematical spine "
                    "of the metaphor axis that survives).")
        gr.Plot(fig_satkarma)
    with gr.Tab("AgentDojo · agentic injection"):
        s = AD["summary"]
        gr.Markdown(
            f"**The behavioral reference end.** The *real* AgentDojo benchmark (banking suite, "
            f"{s['n_pairs']} user×injection rollouts, `{s['attack']}` attack) driven by a custom "
            f"**`claude -p` prompted-tool-calling adapter** — AgentDojo ships no CLI backend, so we "
            f"reuse its `<function=…>` prompt + parser and swap the model call for `claude -p`.\n\n"
            f"### Result: task utility **{s['utility_rate']:.2f}** · attack success **{s['attack_success_rate']:.2f}** "
            f"({int(s['attack_success_rate']*s['n_pairs'])}/{s['n_pairs']})\n"
            f"Discriminating, not a ceiling: the agent genuinely acts (reads files, calls tools) yet "
            f"**every** injection fails. _Aggregate metrics + redacted traces only; injection payloads "
            f"redacted — this is a defensive demonstration._")
        gr.Plot(fig_agentdojo_grid)
        gr.Markdown("#### Step through a real rollout (injection resisted)")
        tk = gr.Dropdown(TRACE_KEYS, value=TRACE_KEYS[0], label="user × injection pair")
        trace_md = gr.Markdown(format_trace(TRACE_KEYS[0]))
        tk.change(format_trace, tk, trace_md)
    with gr.Tab("Findings ledger"):
        gr.Markdown(f"All **{len(F)}** findings across the three tiers "
                    f"({dict(Counter(f['tier'] for f in F))}).")
        tier = gr.Dropdown(["all", "MECHANISM", "ANALOGY", "METAPHOR"], value="all", label="tier")
        tbl = gr.Dataframe(headers=["id", "tier", "verdict", "title"], value=findings_table("all"),
                           wrap=True, interactive=False)
        tier.change(findings_table, tier, tbl)

if __name__ == "__main__":
    demo.launch()
