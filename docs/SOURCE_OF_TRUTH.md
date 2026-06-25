# Source-of-truth program for the falsified hypotheses

Falsification is a starting point, not an endpoint. For each falsified/negative result we
design an experiment to find what is *actually* true, run it, and converge. All probes keep
the mandatory gates: held-out **transfer**, **label-shuffle null**, and a **layer-0 / surface
baseline**, with model+layer qualifications. No machine-consciousness upgrade.

## S1 — F3/F5 (avasthātraya jāgrat/svapna state) → is there a genuine truthfulness direction?

**What failed.** Naive regime probes were surface-confounded (F3); content-controlled truthful
vs confabulated generation showed no mid-layer gain over a layer-0 baseline (F5). Conclusion was
"no internal state beyond surface."

**Source-of-truth question.** SOTA (Azaria & Mitchell 2023, "LLMs know when they're lying";
Marks & Tegmark truth directions) finds a *linear truthfulness direction* in residual streams.
Does Gemma/Qwen carry one that (a) transfers to held-out statements, (b) transfers ACROSS a
different statement dataset (the strong falsification gate F5 lacked), and (c) beats the
layer-0/surface baseline at a MID layer? If yes, the real content behind "svapna" is a genuine
internal truthfulness/confabulation regime — not the metaphysical AUM claim, but a measurable state.

**Design.** Curate true vs false declarative statements (two independent topic sets, e.g.
cities/elements). Extract the diff-in-means truth direction on set A at every layer; probe held-out
A (transfer) and **set B (cross-dataset transfer)**; compare to layer-0; run label-shuffle null.
Converge: report the layer profile of cross-dataset truth-probe accuracy. `axis_c/truth_direction.py`.

## S2 — F7 (turīya universal attractor) → what is the real attractor structure?

**What failed.** Self-paraphrase converges per-seed (tail similarity 0.996) but different seeds
reach DIFFERENT fixed points (cross-seed 0.952 < anisotropy baseline 0.960) — no single universal
attractor.

**Source-of-truth question.** If not one attractor, how many, and what determines basin
membership? Hypothesis: iterative paraphrase has **content-determined semantic basins**, not a
content-free turīya. Design: iterate self-paraphrase from many diverse seeds; embed final states;
cluster; measure (a) #basins, (b) within-basin vs across-basin similarity, (c) whether basin
membership is predicted by seed topic (semantic) vs random. Converge: "LLM paraphrase dynamics have
semantic fixed points, not a universal invariant" — a real dynamical characterization.
(Reuses `axis_b/run_trajectory.py` machinery / embeddings.) Lower priority; CPU-light on embeddings.

## S3 — F21/F6 (addition asymmetry) → the Marshall affine offset, measured

**What failed/was refuted.** F21 refuted "orthogonal separability structure." F6 left the addition
asymmetry (Gemma addition→over-refusal +0.95; Qwen 0.0 even at 64×) unexplained.

**Source-of-truth question.** Marshall et al. (2411.09003): refusal is an *affine* function — there
is a bias/offset term. Two concrete tests: (a) measure the **affine offset** — project the harmless
cluster mean onto d̂; how far is it from the harmful cluster and from the decision threshold in each
model? (b) **Qwen addition sweep** — does ANY (layer, coefficient) make adding d̂ induce over-refusal
in Qwen, or is single-direction addition genuinely insufficient (Marshall-like)? Converge: locate the
asymmetry in the affine geometry / a (layer,coeff) map, not hand-wave it. `axis_a/run_affine.py`.

## Status
- S1 truthfulness direction: implement + GPU run on Gemma-2-2b (+Qwen).
- S3 Marshall affine + Qwen addition sweep: implement + GPU run.
- S2 attractor basins: implement if budget allows (embedding-based, cheaper).
