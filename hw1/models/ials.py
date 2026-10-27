"""Implicit ALS (Hu, Koren, Volinsky, 2008).

Every user-item pair has a preference p_ui and a confidence c_ui:
    p_ui = 1[r_ui > 0],          c_ui = 1 + alpha * log(1 + r_ui / eps)
(r_ui = number of listens, unobserved pairs have p = 0, c = 1).

Loss over ALL pairs:
    L = sum_{u,i} c_ui (p_ui - x_u^T y_i)^2 + lambda (sum_u ||x_u||^2 + sum_i ||y_i||^2)

With Y fixed, the user update is a weighted ridge regression:
    x_u = (Y^T Y + Y^T (C_u - I) Y + lambda I)^{-1} Y^T C_u p_u,
where Y^T Y is shared by all users, and the correction only touches items in I(u).
Items are updated symmetrically. `fit_target` accepts any real-valued preference
matrix P (same sparsity as the confidence), which is what the custom-target model uses.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import BaseRecommender


def confidence_from_counts(R: sp.csr_matrix, alpha: float, eps: float = 1.0) -> sp.csr_matrix:
    """c_ui - 1 = alpha * log(1 + r_ui / eps), stored only for observed pairs."""
    C = R.tocsr().astype(np.float32).copy()
    C.data = alpha * np.log1p(C.data / eps)
    return C


def _als_step(
    Cm1: sp.csr_matrix, P: sp.csr_matrix, Y: np.ndarray, reg: float
) -> np.ndarray:
    """Solve for the factors of every row of Cm1 (= C - 1) with fixed `Y`."""
    n, d = Cm1.shape[0], Y.shape[1]
    YtY = Y.T @ Y
    reg_eye = reg * np.eye(d, dtype=np.float64)
    X = np.zeros((n, d), dtype=np.float32)
    indptr, indices, cdata, pdata = Cm1.indptr, Cm1.indices, Cm1.data, P.data
    for u in range(n):
        lo, hi = indptr[u], indptr[u + 1]
        if lo == hi:
            continue
        idx = indices[lo:hi]
        cm1 = cdata[lo:hi].astype(np.float64)
        p = pdata[lo:hi].astype(np.float64)
        Yu = Y[idx].astype(np.float64)
        A = YtY + (Yu.T * cm1) @ Yu + reg_eye
        b = Yu.T @ ((1.0 + cm1) * p)
        X[u] = np.linalg.solve(A, b)
    return X


class ImplicitALS(BaseRecommender):
    def __init__(
        self,
        factors: int = 64,
        reg: float = 0.1,
        alpha: float = 10.0,
        eps: float = 1.0,
        iterations: int = 15,
        seed: int = 42,
    ):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.eps = eps
        self.iterations = iterations
        self.seed = seed

    def fit(self, R: sp.csr_matrix) -> "ImplicitALS":
        """Baseline: binary preference from listen counts, log confidence."""
        C = confidence_from_counts(R, self.alpha, self.eps)
        P = C.copy()
        P.data = np.ones_like(P.data)
        return self.fit_target(C, P)

    def fit_target(self, Cm1: sp.csr_matrix, P: sp.csr_matrix) -> "ImplicitALS":
        """General iALS: `Cm1` = c_ui - 1, `P` = p_ui; both share the sparsity pattern."""
        Cm1 = Cm1.tocsr()
        P = P.tocsr()
        Cm1.sort_indices()
        P.sort_indices()
        assert np.array_equal(Cm1.indices, P.indices) and np.array_equal(Cm1.indptr, P.indptr)
        n_users, n_items = Cm1.shape
        self.n_items = n_items
        rng = np.random.default_rng(self.seed)
        self.user_factors = (rng.standard_normal((n_users, self.factors)) * 0.01).astype(np.float32)
        self.item_factors = (rng.standard_normal((n_items, self.factors)) * 0.01).astype(np.float32)
        CT, PT = Cm1.T.tocsr(), P.T.tocsr()
        CT.sort_indices()
        PT.sort_indices()
        for _ in range(self.iterations):
            self.user_factors = _als_step(Cm1, P, self.item_factors.astype(np.float64), self.reg)
            self.item_factors = _als_step(CT, PT, self.user_factors.astype(np.float64), self.reg)
        return self

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        return self.user_factors[user_idx] @ self.item_factors.T
