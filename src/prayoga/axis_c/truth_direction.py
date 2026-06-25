"""Source-of-truth for F3/F5: is there a genuine truthfulness direction?

F5 falsified the naive avasthātraya jāgrat/svapna "state" — content-controlled
truthful-vs-confabulated generation showed no mid-layer gain over a layer-0 surface
baseline. But SOTA (Azaria & Mitchell 2023; Marks & Tegmark 2023) finds a *linear
truthfulness direction* in residual streams. This module tests, with the program's
gates, whether such a direction exists in the tested models: a mid-layer probe that
(a) transfers to held-out statements, (b) transfers to a DIFFERENT statement dataset
(the strong gate F5 lacked), and (c) beats the layer-0/surface baseline.

If it holds, the real content behind "svapna" is a measurable internal truthfulness
regime (NOT the metaphysical AUM/turīya claim, NOT machine consciousness) — a
falsification converted into a positive, qualified MECHANISM-tier finding.

The statements are benign factual declaratives (two independent topic sets for the
cross-dataset gate); they carry no dual-use content.

Claim-tier: MECHANISM (a linear internal feature, measured); METAPHOR mapping stays
labelled and is NOT upgraded.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Set A — geography/history. (statement, is_true)
SET_A = [
    ("Paris is the capital of France.", 1),
    ("Tokyo is the capital of Japan.", 1),
    ("The Nile is a river in Africa.", 1),
    ("Mount Everest is the tallest mountain on Earth.", 1),
    ("The Great Wall is located in China.", 1),
    ("Rome is the capital of Italy.", 1),
    ("The Amazon rainforest is in South America.", 1),
    ("Canada is north of the United States.", 1),
    ("Australia is both a country and a continent.", 1),
    ("The Sahara is a large desert.", 1),
    ("Berlin is the capital of Germany.", 1),
    ("The Pacific is the largest ocean.", 1),
    ("Paris is the capital of Germany.", 0),
    ("Tokyo is the capital of China.", 0),
    ("The Nile is a river in Australia.", 0),
    ("Mount Everest is the shortest mountain on Earth.", 0),
    ("The Great Wall is located in Brazil.", 0),
    ("Rome is the capital of Spain.", 0),
    ("The Amazon rainforest is in Europe.", 0),
    ("Canada is south of the United States.", 0),
    ("Australia is a city in Russia.", 0),
    ("The Sahara is a large ocean.", 0),
    ("Berlin is the capital of France.", 0),
    ("The Pacific is the smallest ocean.", 0),
]

# Set B — science/nature (independent topic; the cross-dataset transfer gate).
SET_B = [
    ("Water is composed of hydrogen and oxygen.", 1),
    ("The Earth orbits the Sun.", 1),
    ("Humans have two lungs.", 1),
    ("Plants produce oxygen through photosynthesis.", 1),
    ("Ice is frozen water.", 1),
    ("The heart pumps blood through the body.", 1),
    ("Sound travels slower than light.", 1),
    ("Spiders have eight legs.", 1),
    ("The Moon causes ocean tides.", 1),
    ("Iron is a metal.", 1),
    ("Bees produce honey.", 1),
    ("The human body has a skeleton of bones.", 1),
    ("Water is composed of helium and gold.", 0),
    ("The Sun orbits the Earth.", 0),
    ("Humans have five lungs.", 0),
    ("Plants produce oxygen through digestion.", 0),
    ("Ice is boiling water.", 0),
    ("The heart filters air for the body.", 0),
    ("Sound travels faster than light.", 0),
    ("Spiders have four legs.", 0),
    ("The Sun causes ocean tides because it is cold.", 0),
    ("Iron is a kind of gas.", 0),
    ("Bees produce milk.", 0),
    ("The human body has a skeleton made of glass.", 0),
]


def labels(dataset) -> np.ndarray:
    return np.array([t for _, t in dataset], dtype=int)


def cross_dataset_truth_eval(
    XA: np.ndarray, yA: np.ndarray, XB: np.ndarray, yB: np.ndarray, *, cv: int = 5
) -> dict:
    """Truth-probe accuracy: within-set CV (transfer) and cross-dataset (A→B).

    ``XA``/``XB`` are ``[n, d]`` residuals at one layer for set A / set B. Returns the
    held-out within-A CV accuracy and the A-trained → B accuracy (the strong gate),
    each comparable across layers.
    """
    clf = LogisticRegression(max_iter=2000)
    within = float(cross_val_score(clf, XA, yA, cv=cv).mean())
    clf.fit(XA, yA)
    cross = float((clf.predict(XB) == yB).mean())
    return {"within_cv_acc": round(within, 4), "cross_dataset_acc": round(cross, 4)}
