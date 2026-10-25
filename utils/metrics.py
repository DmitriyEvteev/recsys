"""Top-K ranking metrics. All models are evaluated with the same function."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ground_truth_lists(rel: pd.DataFrame, n_users: int) -> list[np.ndarray]:
    """rel: columns uidx, iidx -> list of relevant item indices per user index."""
    gt = [np.empty(0, dtype=np.int64) for _ in range(n_users)]
    for u, items in rel.groupby("uidx")["iidx"]:
        gt[u] = np.unique(items.to_numpy())
    return gt


def evaluate(recs: np.ndarray, users: np.ndarray, gt: list[np.ndarray], n_items: int,
             k: int = 10) -> dict:
    """NDCG@k, Recall@k, HitRate@k, MRR@k averaged over `users`; Coverage@k over the catalogue.

    NDCG@k  = DCG@k / IDCG@k with binary gains, DCG = sum_r rel_r / log2(r + 1)
    Recall@k = |top-k ∩ rel| / min(|rel|, k)   (so 1.0 is reachable)
    Coverage@k = share of catalogue items shown to at least one user
    """
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ndcg, recall, hit, mrr = [], [], [], []
    for row, u in zip(recs[:, :k], users):
        g = gt[u]
        hits = np.isin(row, g)
        dcg = (hits * discounts).sum()
        idcg = discounts[: min(len(g), k)].sum()
        ndcg.append(dcg / idcg)
        recall.append(hits.sum() / min(len(g), k))
        hit.append(float(hits.any()))
        first = np.flatnonzero(hits)
        mrr.append(1.0 / (first[0] + 1) if len(first) else 0.0)
    res = {
        f"NDCG@{k}": float(np.mean(ndcg)),
        f"Recall@{k}": float(np.mean(recall)),
        f"HitRate@{k}": float(np.mean(hit)),
        f"MRR@{k}": float(np.mean(mrr)),
        f"Coverage@{k}": float(len(np.unique(recs[:, :k])) / n_items),
    }
    return res
