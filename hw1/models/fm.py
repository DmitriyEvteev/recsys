"""Factorization Machine (Rendle, 2010) with metadata, trained with the BPR loss.

A (user, item) pair is described by a sparse feature vector x (weights x_f):
  user side:  user ID (1.0) + bag of artists the user listened to in train
              (weights = normalised log listen counts);
  item side:  item ID, artist ID, album ID, track-length bucket, popularity bucket
              (popularity is counted on the training data only).

FM score:
    y(x) = w_0 + sum_f w_f x_f + sum_{f<g} <v_f, v_g> x_f x_g
         = w_0 + sum_f w_f x_f + 1/2 (||sum_f v_f x_f||^2 - sum_f ||v_f||^2 x_f^2).

Split the features into user part U and item part I. Then
    y(u, i) = const(u) + <V_U(u), V_I(i)> + b(i),
    V_U(u) = sum_{f in U} v_f x_f,   V_I(i) = sum_{f in I} v_f,
    b(i)   = sum_{f in I} w_f + 1/2 (||V_I(i)||^2 - sum_{f in I} ||v_f||^2).
const(u) does not change the ranking of a user's items, so it is dropped.
This makes scoring of all items a single matrix product.

BPR loss for a positive item i and a sampled negative j:
    L = -log sigma(y(u,i) - y(u,j))
        + lambda (||V_U(u)||^2 + sum_{f in I(i)} ||v_f||^2 + sum_{f in I(j)} ||v_f||^2).
Negatives j are sampled uniformly from the catalogue.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .base import BaseRecommender


class FactorizationMachine(BaseRecommender):
    def __init__(
        self,
        factors: int = 64,
        lr: float = 0.005,
        reg: float = 1e-5,
        epochs: int = 20,
        batch_size: int = 16384,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose

    # ---------------------------------------------------------------- features
    def _build_features(
        self, R: sp.csr_matrix, item_fields: dict[str, np.ndarray], user_bag: sp.csr_matrix
    ):
        n_users, n_items = R.shape
        offset = 0
        self.user_offset = offset
        offset += n_users
        item_cols = [np.arange(n_items) + offset]
        offset += n_items
        bag_offset = None
        for name, values in item_fields.items():
            values = np.asarray(values, dtype=np.int64)
            item_cols.append(values + offset)
            offset += int(values.max()) + 1
        if user_bag is not None:
            bag_offset = offset
            offset += user_bag.shape[1]
        self.n_features = offset
        self.item_feat = torch.as_tensor(np.stack(item_cols, axis=1), dtype=torch.long)

        # user side as an EmbeddingBag input: [user id] + [artist bag]
        if user_bag is not None:
            bag = user_bag.tocsr()
            counts = np.diff(bag.indptr) + 1
            idx = np.empty(counts.sum(), dtype=np.int64)
            w = np.empty(counts.sum(), dtype=np.float32)
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            idx[starts] = np.arange(n_users) + self.user_offset
            w[starts] = 1.0
            mask = np.ones(len(idx), dtype=bool)
            mask[starts] = False
            idx[mask] = bag.indices + bag_offset
            w[mask] = bag.data
        else:
            starts = np.arange(n_users)
            idx = np.arange(n_users) + self.user_offset
            w = np.ones(n_users, dtype=np.float32)
        self.user_idx_t = torch.as_tensor(idx)
        self.user_w_t = torch.as_tensor(w)
        self.user_offsets_t = torch.as_tensor(starts, dtype=torch.long)

    # ------------------------------------------------------------------ model
    def _user_vectors(self) -> torch.Tensor:
        return F.embedding_bag(
            self.user_idx_t, self.V, self.user_offsets_t,
            mode="sum", per_sample_weights=self.user_w_t,
        )

    def _item_parts(self, items: torch.Tensor):
        feats = self.item_feat[items]  # (b, n_fields)
        v = self.V[feats]  # (b, n_fields, d)
        s = v.sum(dim=1)
        sq = v.pow(2).sum(dim=(1, 2))  # sum_f ||v_f||^2, also used for L2 regularisation
        bias = self.w[feats].sum(dim=1) + 0.5 * (s.pow(2).sum(-1) - sq)
        return s, bias, sq

    def fit(
        self,
        R: sp.csr_matrix,
        item_fields: dict[str, np.ndarray] | None = None,
        user_bag: sp.csr_matrix | None = None,
        callback=None,
    ) -> "FactorizationMachine":
        """`item_fields` (name -> category per item) and `user_bag` (users x artists) are the
        metadata; with both empty the model reduces to a pure ID-based FM (= BPR-MF + biases).
        `callback(model, epoch)` is called after every epoch (e.g. validation metrics);
        returning True stops training early."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        R = R.tocsr()
        self.n_items = R.shape[1]
        self._build_features(R, item_fields or {}, user_bag)

        self.V = torch.nn.Parameter(torch.randn(self.n_features, self.factors) * 0.01)
        self.w = torch.nn.Parameter(torch.zeros(self.n_features))
        opt = torch.optim.Adam([self.V, self.w], lr=self.lr)

        coo = R.tocoo()
        users, pos = coo.row.astype(np.int64), coo.col.astype(np.int64)
        n = len(users)
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            neg_all = rng.integers(0, self.n_items, size=n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                b = perm[start:start + self.batch_size]
                u = torch.as_tensor(users[b])
                i = torch.as_tensor(pos[b])
                j = torch.as_tensor(neg_all[b])
                U = self._user_vectors()[u]
                si, bi, sqi = self._item_parts(i)
                sj, bj, sqj = self._item_parts(j)
                diff = (U * (si - sj)).sum(-1) + bi - bj
                loss = -F.logsigmoid(diff).mean() + self.reg * (
                    U.pow(2).sum(-1) + sqi + sqj
                ).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            if self.verbose:
                print(f"epoch {epoch + 1}: loss {total / n:.4f}")
            self._finalize()
            if callback is not None and callback(self, epoch + 1):
                break
        return self

    def _finalize(self) -> None:
        """Pre-compute V_U(u), V_I(i) and b(i) for fast scoring."""
        with torch.no_grad():
            self.user_emb = self._user_vectors().numpy().copy()
            s, bias, _ = self._item_parts(torch.arange(self.n_items))
            self.item_emb = s.numpy().copy()
            self.item_bias = bias.numpy().copy()

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        return self.user_emb[user_idx] @ self.item_emb.T + self.item_bias[None, :]
