"""Item-based collaborative filtering (ItemKNN).

Similarity: cosine with shrinkage
    s_ij = <x_i, x_j> / (||x_i|| * ||x_j|| + h),
where x_i is the item column of the (weighted) user-item matrix.
Only the `n_neighbors` most similar items are kept for every item.

Score of item j for user u:
    score_uj = sum_{i in I(u)} r_ui * s_ij,
i.e. the user's history vector multiplied by the pruned similarity matrix.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import BaseRecommender


class ItemKNN(BaseRecommender):
    def __init__(self, n_neighbors: int = 200, shrink: float = 10.0, weighting: str = "log"):
        self.n_neighbors = n_neighbors
        self.shrink = shrink
        self.weighting = weighting  # "binary" | "log"

    def _weigh(self, X: sp.csr_matrix) -> sp.csr_matrix:
        X = X.tocsr().astype(np.float32)
        if self.weighting == "binary":
            X.data = np.ones_like(X.data)
        elif self.weighting == "log":
            X.data = np.log1p(X.data)
        else:
            raise ValueError(self.weighting)
        return X

    def fit(self, X: sp.csr_matrix) -> "ItemKNN":
        self.n_items = X.shape[1]
        self.X = self._weigh(X)
        Xd = self.X.toarray()
        norms = np.sqrt((Xd * Xd).sum(axis=0))
        # Dense co-occurrence: the catalogue has 10k items, so a dense BLAS product is fastest.
        co = Xd.T @ Xd
        sim = co / (np.outer(norms, norms) + self.shrink + 1e-12)
        np.fill_diagonal(sim, 0.0)
        k = min(self.n_neighbors, self.n_items - 1)
        # Keep top-k neighbours of every item (row i = neighbours of item i).
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        vals = np.take_along_axis(sim, idx, axis=1)
        rows = np.repeat(np.arange(self.n_items), k)
        S = sp.csr_matrix((vals.ravel(), (rows, idx.ravel())), shape=sim.shape, dtype=np.float32)
        S.eliminate_zeros()
        self.S = S
        return self

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        return (self.X[user_idx] @ self.S).toarray()
