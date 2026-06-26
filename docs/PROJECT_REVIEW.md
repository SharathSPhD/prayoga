# prayoga — Holistic Project Review

*A cohesiveness and gap audit across the paper, website, notebooks, Hugging Face
demo, plugin, data layer, and supporting docs. Written 2026-06-26.*

This document is a critical review, not a status report. It assumes the work is
strong (the mechanism tier is real, the falsifications are honest, the triangulation
keystone is a genuine result) and focuses on where the **project as a system** loses
cohesiveness, where claims have drifted between surfaces, and what is missing before
the work reads as one finished artifact. Items already fixed in the 2026-06-26 pass
are marked **[fixed]**; everything else is a live recommendation.

---

## 1. Executive summary

The science is in good shape and the findings ledger (`docs/FINDINGS.md`, F1–F25) is
the single most trustworthy surface in the repository. The weakness is **not** the
research — it is that the research moved faster than its presentation layers, so the
same claim is told three different ways depending on which file you open.

The one structural fault that explains most of the incoherence: **the F23/F25 reframe
has not propagated outward.** Findings F23 (the cross-family "addition asymmetry" is a
dose-calibration artifact, not model-specific insufficiency) and F25 (both automatic
behavioural metrics are biased) overturned the earlier headline that "sufficiency is
model-specific." The findings ledger records this; the paper now reflects it
**[fixed]**; but the README, several finding summaries, the HF/site narrative copy,
and the plugin descriptions still lead with the older "necessary-but-not-sufficient
asymmetry / dimensionality predicts steerability" story. A reader who triangulates
across artifacts will conclude the project contradicts itself.

The second structural issue is **headline drift**: there are at least three competing
one-line theses in circulation — "Refusal as a Captured Symmetry" (paper title),
"Asymmetry with a Shared Necessary Core" (former site hero), and "one mechanism at
the necessary level, many at the sufficient level" (README / F17). These are not
quite synonyms, and after F23 the third is partly outdated. The project needs one
canonical sentence, used everywhere.

Everything else is downstream of these two.

---

## 2. Cross-artifact consistency map

| Surface | Findings coverage | Reflects F23/F25 reframe | Build/health |
|---|---|---|---|
| `docs/FINDINGS.md` | F1–F25 (canonical) | Yes | n/a — source of truth |
| Paper (`paper/`) | F1–F25 | Yes **[fixed]** | Compiles, 31 pp **[fixed]** |
| Website (`site/`) data | 25 findings | Partial (F6 summary stale) | Was broken (deps); rebuilt **[fixed]** |
| Website narrative copy | Was F1–F18 only | Now yes **[fixed]** | — |
| HF Space (`hf_space/`) | mirrors metrics | Needs check | App present |
| `README.md` | F1–F25 referenced | **No — stale** | — |
| Notebooks | F1, symmetry only | No | 3 notebooks |
| Plugin commands | per-finding | Partly stale | Manifest present |

The takeaway: the **data** is largely current everywhere, but the **prose that
interprets the data** lags by one or two findings on every surface except the ledger
and (now) the paper.

---

## 3. The central fault line: claim drift after F23/F25

The pre-F23 story (still visible in README lines 5 and 51–63, the `findings.json` F6
title, and earlier site captions) is:

> Ablation transfers across families, but **addition is sufficient only in Gemma, not
> Qwen**, because Qwen's refusal subspace is 3-dimensional. **Dimensionality predicts
> steerability.**

The post-F23 story (ledger, paper) is:

> Ablation transfers; addition, **once its coefficient is calibrated, is sufficient in
> both families** (Qwen peaks then collapses off-distribution at the fixed 64× point
> the old test used). What stays model-specific is **quantitative** — dose window,
> effective dimension, EC50 — not sufficiency itself.

These are materially different claims. The first is falsified-by-its-own-program; the
second is the current truth. Recommended actions:

- **[P0]** Update `README.md` §"Status & results" and the one-line thesis to the
  post-F23 statement. The phrase "effective dimension … predicts an addition
  asymmetry" should be retired.
- **[P0]** Reconcile `findings.json` F6: keep the entry but add a `superseded_by:
  "F23"` field and soften the title from "necessary-not-sufficient asymmetry" to
  "cross-family transfer (asymmetry later shown to be a dose artifact)." The ledger
  already says "SUPERSEDED by F23"; the machine-readable copy should too.
- **[P1]** Audit HF `metrics.json` / `app.py` and plugin command blurbs for the words
  "asymmetry", "not sufficient", and "dimensionality predicts" and align them.

---

## 4. Headline / framing coherence

The project keeps three slogans alive. Recommendation **[P0]**: adopt a single
canonical thesis sentence and propagate it to the paper title block, site hero,
README first line, HF README, and plugin description. A candidate that survives F23:

> *Refusal is a measurable, ablatable, dosable residual-stream direction with a shared
> necessary core across model families; its quantitative structure (dose, dimension,
> potency) is model-specific, and prompt-level injection collapses its internal
> signature without producing behavioural capture.*

"Captured symmetry" is a good evocative title; "shared necessary core" is the accurate
subtitle; the two should be stated together rather than competing.

A related framing nit: the word **"symmetry"** is doing double duty as (a) the measured
orbit-invariance/order-parameter result and (b) a loose metaphor for the whole
program. The paper now disciplines this (mechanism vs analogy); the site and README
should make the same split explicit so a physicist reader is not over-promised a group
action.

---

## 5. Per-artifact review

### 5.1 Paper — **substantially repaired this pass**

Problems found (now **[fixed]** unless noted):
- The **appendix was the single worst surface in the project**: it contained
  fabricated placeholder numbers that contradict the ledger — EC50 "0.8–2.3" (real:
  0.13–0.33), "MMLU 70.2%", "many-shot ASR 52%", "perplexity 7.8 vs 7.9", "norm–logit
  r=0.71", "cosine 0.94±0.03 across 5 seeds". None of these appear in any results
  JSON. Rewritten from real findings.
- Internal finding-IDs (F3/F5 in the TikZ figure, F1/F6/F11/F18 in body prose) leaked
  into the manuscript — an internal-ledger artifact inappropriate for publication.
- Two genuine LaTeX bugs (`2--3\times` outside math mode in `background.tex` and
  `positioning.tex`) would error on a clean build.
- `positioning.tex` was the most stale section: it called the order parameter
  "post-hoc and not validated" (contradicting the validated F11), used a
  "novelty-strength" bullet scaffold, and was written in mixed voice.
- First-person voice throughout; converted to third person.
- Figure under-coverage: the strongest recent results (triangulation, truth direction,
  māraṇa-v2) had no figure, and the existing `f23_addition_dose.png` was generated but
  never referenced. Eleven figures now, all referenced.

Remaining **[P1]**: small-n is the honest ceiling — most non-primary findings still
rest on n≈20, single seed, single layer. Before submission, at minimum the
order-parameter (F11) and EC50 (F19) results deserve multi-seed CIs, and a pre-
registered primary-vs-exploratory split would pre-empt the reviewer objection.

### 5.2 Website — **rebuilt and re-narrated this pass**

Problems found:
- **The build was broken**: `node_modules` was incomplete (missing transitive deps
  such as `yargs-parser`), so `astro build` failed outright. Dependencies reinstalled.
- The homepage covered only F1–F18 and omitted the newest, strongest results
  (triangulation keystone, dose-artifact reframe, truth direction, AgentDojo, EC50
  pharmacology). Rewritten as an ACD-style scrollytelling essay (Acts I–VI) covering
  the full arc for both audiences, with expandable technical detail boxes.
- Content accuracy bugs: asserted the SAE unit as "feature #566" as fact (F17 shows
  the index is seed-specific and should not be stated as a stable label); glossed
  vaśīkaraṇa as "feminine power" (inaccurate); shouted "ONE/MANY" in caps. All
  corrected.
- The secondary pages (`mechanism.astro`, `three-tiers.astro`, `vasikarana.astro`,
  `debate.astro`, `reproducibility.astro`, `findings.astro`, `about.astro`) were
  **not** revised this pass and may still carry pre-F23 copy. **[P1]** audit them for
  the same drift, or fold them into the new single-essay structure.

### 5.3 Figures — **regenerated this pass**

The old figures were functional but used matplotlib defaults with inconsistent styling
and no shared palette. They are now regenerated under one house style (serif to match
the Palatino body, a fixed colourblind-safe palette, despined axes), and four new
figures were added (triangulation, truth direction, calibrated addition dose, māraṇa
v2). **[P2]** The figure pipeline (`scripts/make_figures.py`) reads from `results/`,
which is git-ignored; document in the script header that a contributor without the raw
results cannot regenerate figures, and consider committing the aggregate JSON the
figures actually need (they contain no dual-use vectors).

### 5.4 Hugging Face Space — **not modified this pass**

`hf_space/` duplicates `findings.json` and `metrics.json` and ships an `app.py`.
**[P1]** Verify the demo text reflects F23/F25 and that its data copies are
regenerated from the same source as the site (see §6 on duplication). The
`agentdojo_demo.json` and metrics should be confirmed current.

### 5.5 Notebooks — **gap**

Only three notebooks exist (`00_explore_results`, `01_reproduce_refusal_direction`,
`02_agentic_symmetry_breaking`). They cover F1 and the symmetry result but **none of
the post-review program** — no EC50 pharmacology, no triangulation, no dose-artifact
sweep, no truth direction. **[P1]** Add a notebook that reproduces the triangulation
keystone (the paper's headline bound) and the addition dose sweep from aggregate JSON;
these are the results a skeptical reader will most want to run.

### 5.6 Plugin — **not modified this pass**

Commands exist for `refusal`, `dose`, `dimensionality`, `symmetry`, `satkarma`,
`active`, `ec50-scaling`, `agentdojo`, `triangulation`, `trajectory-probe`. **[P2]**
There is no command for the truth-direction or addition-dose source-of-truth results;
and the `dimensionality` / `triangulation` blurbs should be checked for pre-F23
language.

### 5.7 Data layer — **duplication risk**

`findings.json` and `metrics.json` each exist in **three** places (`site/src/data`,
`site/public/data`, `hf_space/`), plus `data/findings_registry.json` as a fourth
representation. They are currently close to consistent, but three hand-synced copies
is a drift generator — the F6 staleness in `findings.json` is exactly this failure
mode in miniature. **[P1]** Make `docs/FINDINGS.md` + one JSON the single source, and
have `scripts/export_aggregates.py` generate every downstream copy in CI so the
surfaces cannot diverge.

---

## 6. Structural recommendations (prioritized)

**P0 — do before any external sharing**
1. Propagate the F23/F25 reframe to README, `findings.json` F6, and the HF/plugin
   blurbs. (Paper and site narrative already done.)
2. Choose one canonical thesis sentence and use it verbatim everywhere.
3. Reconcile the "symmetry" word into its measured-vs-metaphor split on every surface,
   matching the paper.

**P1 — before the paper is submitted / the site is announced**
4. Single-source the data layer; generate all JSON copies via `export_aggregates.py`
   in CI; delete hand-maintained duplicates.
5. Audit the secondary site pages and HF demo for residual pre-F23 copy.
6. Add notebooks reproducing the triangulation keystone and the addition dose sweep.
7. Add multi-seed CIs (or an explicit primary-vs-exploratory pre-registration) for at
   least F11 and F19.

**P2 — polish**
8. Add plugin commands for the source-of-truth results (truth direction, addition
   dose).
9. Commit the figure-relevant aggregate JSON (no dual-use content) so figures are
   reproducible without the gated `results/`.
10. Add a short CONTRIBUTING note describing the FINDINGS-as-source-of-truth discipline
    so future results propagate outward in one place.

---

## 7. What is genuinely strong (keep)

The tier discipline is the project's best feature and should be defended: the
mechanism/analogy/metaphor split, the transfer/null/surface gates, and the willingness
to demote (F3), falsify (F7), and overturn its own prior claim (F6→F23) are exactly
what makes the work credible. The triangulation keystone (internal collapse is
necessary but not sufficient for behavioural capture) and the dependent-variable bias
result (F25) are the two findings most likely to be cited by others, because they are
methodological cautions the wider steering literature needs. The figure set and the
paper now foreground both. The honest negatives are an asset, not a liability — the
recommendation is only to make sure every surface tells the *same* honest story.
