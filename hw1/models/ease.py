"""EASE (Steck, 2019): Embarrassingly Shallow Autoencoder.

Find an item-item weight matrix B with zero diagonal:
    min_B ||X - X B||_F^2 + lambda ||B||_F^2   s.t.  diag(B) = 0.
Closed form (Lagrange multipliers for the diagonal constraint):
    G = X^T X + lambda I,   P = G^{-1},   B_ij = -P_ij / P_jj  (i != j),  B_jj = 0.
Score of user u: x_u^T B.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .base import BaseRecommender


class EASE(BaseRecommender):
    def __init__(self, reg: float = 500.0, weighting: str = "binary"):
        self.reg = reg
        self.weighting = weighting  # "binary" | "log"

    def _weigh(self, X: sp.csr_matrix) -> sp.csr_matrix:
        X = X.tocsr().astype(np.float32)
        X.data = np.ones_like(X.data) if self.weighting == "binary" else np.log1p(X.data)
        return X

    def fit(self, X: sp.csr_matrix) -> "EASE":
        self.n_items = X.shape[1]
        self.X = self._weigh(X)
        Xd = torch.from_numpy(self.X.toarray()).double()
        G = Xd.T @ Xd  # dense Gram matrix: X^T X is nearly dense for a 10k catalogue
        G.diagonal().add_(self.reg)
        # G is symmetric positive definite -> invert through the Cholesky factor
        P = torch.cholesky_inverse(torch.linalg.cholesky(G))
        B = -P / torch.diagonal(P)[None, :]
        B.fill_diagonal_(0.0)
        self.B = B.float().numpy()
        return self

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        return self.X[user_idx] @ self.B
