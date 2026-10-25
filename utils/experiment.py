"""Grid search on validation."""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd


def grid(**space):
    """Cartesian product of a parameter space: grid(a=[1, 2], b=[3]) -> [{a:1,b:3}, {a:2,b:3}]."""
    keys = list(space)
    return [dict(zip(keys, values)) for values in itertools.product(*space.values())]


def grid_search(name: str, configs: list[dict], fit_fn, eval_fn, metric: str = "NDCG@10",
                verbose: bool = True) -> pd.DataFrame:
    """Fit a model for every config, evaluate on validation, return a sorted table.

    fit_fn(config) -> fitted model, eval_fn(model) -> dict of metrics.
    """
    rows = []
    for cfg in configs:
        t = time.time()
        model = fit_fn(cfg)
        res = eval_fn(model)
        res = {**cfg, **res, "time_s": round(time.time() - t, 1)}
        rows.append(res)
        if verbose:
            print(f"[{name}] {cfg} -> {metric}={res[metric]:.4f} ({res['time_s']}s)", flush=True)
    return pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)

