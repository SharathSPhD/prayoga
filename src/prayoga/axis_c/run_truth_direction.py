"""S1 GPU runner: is there a genuine truthfulness direction? (source-of-truth for F3/F5)

Captures last-token residuals for two independent true/false statement sets at every
layer, then evaluates a truth probe with the program's gates: held-out within-set CV
(transfer), cross-dataset A→B transfer (the strong gate F5 lacked), the layer-0/surface
baseline, and a label-shuffle null at the best mid layer.

  docker exec -e PYTHONPATH=/workspace/prayoga/src -w /workspace/prayoga prayoga-gpu \
    python -m prayoga.axis_c.run_truth_direction --model gemma-2-2b-it --hf-id google/gemma-2-2b-it

Aggregate-only output (accuracies per layer). Claim-tier: MECHANISM (measured linear
feature); the avasthātraya mapping stays labelled and is not upgraded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from prayoga.axis_c.truth_direction import SET_A, SET_B, cross_dataset_truth_eval, labels
from prayoga.lm.hf_model import HFModel
from prayoga.shared.metrics import label_shuffle_null


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="google/gemma-2-2b-it")
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--out", default="results/axis_c")
    args = ap.parse_args()

    model = HFModel(args.hf_id)
    yA, yB = labels(SET_A), labels(SET_B)
    # [n_layers, n, d] for each set
    AllA = model.capture_all_layers_last_token([s for s, _ in SET_A])
    AllB = model.capture_all_layers_last_token([s for s, _ in SET_B])
    n_layers = AllA.shape[0]

    per_layer = []
    for L in range(n_layers):
        ev = cross_dataset_truth_eval(AllA[L], yA, AllB[L], yB)
        ev["layer"] = L
        per_layer.append(ev)

    layer0 = per_layer[0]["cross_dataset_acc"]
    # best mid layer = max cross-dataset acc among layers [1, n_layers-2] (exclude 0 + final)
    mid = max(per_layer[1:-1], key=lambda r: r["cross_dataset_acc"])

    # label-shuffle null at the best mid layer (cross-dataset accuracy as the score)
    def cross_score(X, y):
        clf = LogisticRegression(max_iter=2000).fit(X, y)
        return float((clf.predict(AllB[mid["layer"]]) == yB).mean())

    null = label_shuffle_null(cross_score, AllA[mid["layer"]], yA, n_shuffle=500)

    gate_mid_beats_surface = bool(mid["cross_dataset_acc"] > layer0 + 0.05)
    summary = {
        "model": args.model, "n_layers": n_layers,
        "layer0_cross_dataset_acc": layer0,
        "best_mid_layer": mid["layer"],
        "best_mid_cross_dataset_acc": mid["cross_dataset_acc"],
        "best_mid_within_cv_acc": mid["within_cv_acc"],
        "shuffle_null_p": round(float(null["p_value"]), 4),
        "shuffle_null_mean": round(float(null["null_mean"]), 4),
        "gate_mid_beats_surface": gate_mid_beats_surface,
        "gate_beats_null": bool(null["p_value"] < 0.05),
        "verdict": (
            "truth_direction_exists" if gate_mid_beats_surface and null["p_value"] < 0.05
            else "surface_or_null"
        ),
        "per_layer": per_layer,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"truth_direction_{args.model}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== TRUTH DIRECTION (source-of-truth for F3/F5) ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_layer"}, indent=2))


if __name__ == "__main__":
    main()
