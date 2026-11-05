"""Shared statistical helpers.

This module carries only the helper functions shared by the paper drivers.
The full paper-A driver is released separately at
github.com/miladkhademinori/mahalanobis-margin; this repository releases
the paper-B code only.
"""

from typing import List, Sequence

import numpy as np


def _agg(rows: List[dict], keys: Sequence[str]) -> dict:
    return {k: _ms([r[k] for r in rows]) for k in keys}


def _ms(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return dict(mean=float(v.mean()) if v.size else None, sd=float(v.std(ddof=1)) if v.size > 1 else 0.0, n=int(v.size),
                values=[float(x) for x in v])


def _ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(intercept=float(beta[0]), slope=float(beta[1]), r2=1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), ss_res=ss_res, n=int(len(x)))
