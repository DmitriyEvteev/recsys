"""iALS with a custom target signal built from listening quality and explicit feedback.

The baseline iALS (models/ials.py) uses p_ui = 1 for every listened track, so a track
skipped after 5 seconds counts as much as a track played to the end ten times.
Here the preference p_ui is a graded value in [0, 1]:

    g_ui = number of "good" listens (played_ratio >= 50 %),  n_ui = number of listens
    q_ui = (g_ui + kappa * mu) / (n_ui + kappa)        # smoothed completion rate
    p_ui = 1                 if the user liked the track,
           0                 if the user disliked it (and did not like it afterwards),
           q_ui ** gamma     otherwise.

* q_ui is a Bayesian-smoothed share of good listens: with prior mean mu and strength
  kappa a single skip gives a low but non-zero target, many skips give ~0.
* gamma > 1 makes the signal sharper (only consistently completed tracks get p close to 1).
* Likes/dislikes are explicit statements and override the implicit evidence.

The confidence is EXACTLY the baseline one, c_ui = 1 + alpha * log(1 + r_ui / eps),
with r_ui = number of listens (an event without listens counts as 1),
so the comparison with the baseline isolates the effect of the target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .ials import ImplicitALS, confidence_from_counts


def build_custom_target(
    pairs: pd.DataFrame,
    shape: tuple[int, int],
    kappa: float = 2.0,
    mu: float = 0.5,
    gamma: float = 1.0,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Return (r_ui listen counts, p_ui target) as CSR matrices with equal sparsity.

    `pairs` has one row per (user, item) with columns:
    uidx, iidx, n_listens, n_good, liked, disliked.
    """
    n = np.maximum(pairs["n_listens"].to_numpy(np.float32), 1.0)
    g = pairs["n_good"].to_numpy(np.float32)
    q = (g + kappa * mu) / (n + kappa)
    p = q ** gamma
    p = np.where(pairs["disliked"].to_numpy(bool), 0.0, p)
    p = np.where(pairs["liked"].to_numpy(bool), 1.0, p).astype(np.float32)
    # Build R and P from one sorted COO so both share the sparsity pattern.
    # Zero targets stay stored: they carry confidence-weighted negative evidence.
    rows, cols = pairs["uidx"].to_numpy(), pairs["iidx"].to_numpy()
    order = np.lexsort((cols, rows))
    rows, cols, n, p = rows[order], cols[order], n[order], p[order]
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=shape[0]))])
    R = sp.csr_matrix((n, cols, indptr), shape=shape)
    P = sp.csr_matrix((p, cols, indptr), shape=shape)
    return R, P


class CustomTargetALS(ImplicitALS):
    def __init__(self, kappa: float = 2.0, mu: float = 0.5, gamma: float = 1.0, **ials_kwargs):
        super().__init__(**ials_kwargs)
        self.kappa = kappa
        self.mu = mu
        self.gamma = gamma

    def fit_pairs(self, pairs: pd.DataFrame, shape: tuple[int, int]) -> "CustomTargetALS":
        R, P = build_custom_target(pairs, shape, self.kappa, self.mu, self.gamma)
        C = confidence_from_counts(R, self.alpha, self.eps)
        return self.fit_target(C, P)
