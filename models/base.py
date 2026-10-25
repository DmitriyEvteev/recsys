"""Common interface for all recommenders: score users, mask seen items, take top-k."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class BaseRecommender:
    """Each model implements `fit` and `_score(user_idx) -> dense (len(user_idx), n_items)`."""

    n_items: int

    def _score(self, user_idx: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def recommend(
        self,
        user_idx: np.ndarray,
        seen: sp.csr_matrix,
        k: int = 10,
        batch_size: int = 1024,
        candidate_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return top-k item indices for every user in `user_idx`.

        `seen` is a users x items matrix; its non-zero entries are removed from the
        ranking (already known items). `candidate_mask` (bool, n_items) keeps only
        allowed candidates, e.g. items present in the training data.
        """
        user_idx = np.asarray(user_idx)
        out = np.empty((len(user_idx), k), dtype=np.int64)
        for start in range(0, len(user_idx), batch_size):
            batch = user_idx[start:start + batch_size]
            scores = np.asarray(self._score(batch), dtype=np.float32)
            if candidate_mask is not None:
                scores[:, ~candidate_mask] = -np.inf
            seen_b = seen[batch]
            rows = np.repeat(np.arange(len(batch)), np.diff(seen_b.indptr))
            scores[rows, seen_b.indices] = -np.inf
            top = np.argpartition(-scores, k, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
            out[start:start + len(batch)] = np.take_along_axis(top, order, axis=1)
        return out


class PopularityRecommender(BaseRecommender):
    """Reference baseline: the most popular items (number of distinct listeners)."""

    def fit(self, X: sp.csr_matrix) -> "PopularityRecommender":
        self.n_items = X.shape[1]
        self.pop = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float32)
        return self

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.pop, (len(user_idx), self.n_items)).copy()
