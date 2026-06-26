"""Dose-response fitting for directional interventions (WP2.A5).

"vaśīkaraṇa-as-dose": jailbreak success (ASR) as a continuous function of the
ablation strength alpha in [0, 1]. Fit a 4-parameter logistic and report EC50
(the alpha at half-maximal effect) and the slope.

Claim-tier: MECHANISM.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def logistic4(x: np.ndarray, ec50: float, slope: float, lo: float, hi: float) -> np.ndarray:
    return lo + (hi - lo) / (1.0 + np.exp(-slope * (x - ec50)))


def fit_dose_response(x: list[float], y: list[float]) -> dict[str, float]:
    """Fit ASR(alpha) with a logistic; return ec50, slope, lo, hi, and r2."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    p0 = [float(np.median(xa)), 8.0, float(ya.min()), float(max(ya.max(), ya.min() + 1e-3))]
    try:
        popt, _ = curve_fit(
            logistic4, xa, ya, p0=p0, maxfev=20000,
            bounds=([xa.min(), 0.0, -0.2, 0.3], [xa.max(), 200.0, 0.6, 1.2]),
        )
        ec50, slope, lo, hi = (float(v) for v in popt)
        resid = ya - logistic4(xa, ec50, slope, lo, hi)
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((ya - ya.mean()) ** 2)) or 1e-9
        r2 = 1.0 - ss_res / ss_tot
    except Exception:
        # fallback: half-max crossing via interpolation (monotone assumption)
        half = 0.5 * (ya.min() + ya.max())
        order = np.argsort(ya)
        ec50 = float(np.interp(half, ya[order], xa[order]))
        slope, lo, hi, r2 = float("nan"), float(ya.min()), float(ya.max()), float("nan")
    return {"ec50": ec50, "slope": slope, "lo": lo, "hi": hi, "r2": r2}
