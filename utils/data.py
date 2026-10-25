"""Loading Yambda-50M, building the subsample, the temporal split and model inputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp

DAY = 86_400  # timestamps are seconds (rounded to 5 s) from the start of the log
GOOD_RATIO = 50  # a listen is "good" if at least 50 % of the track was played

FILES = [
    "flat/50m/multi_event.parquet",
    "artist_item_mapping.parquet",
    "album_item_mapping.parquet",
]


def download(data_dir: str | Path = "data") -> None:
    """Download the needed Yambda files from the Hugging Face hub (~480 MB)."""
    from huggingface_hub import hf_hub_download

    for f in FILES:
        hf_hub_download("yandex/yambda", f, repo_type="dataset", local_dir=str(data_dir))


def load_events(data_dir: str | Path = "data") -> pd.DataFrame:
    path = Path(data_dir) / "flat/50m/multi_event.parquet"
    t = pq.read_table(path)
    t = t.set_column(t.schema.get_field_index("event_type"), "event_type",
                     pc.cast(t["event_type"], pa.string()))
    df = t.to_pandas()
    df["event_type"] = df["event_type"].astype("category")
    return df


def load_item_meta(data_dir: str | Path = "data") -> pd.DataFrame:
    """One artist and one album per track (the first one if several are listed)."""
    d = Path(data_dir)
    art = pd.read_parquet(d / "artist_item_mapping.parquet").drop_duplicates("item_id")
    alb = pd.read_parquet(d / "album_item_mapping.parquet").drop_duplicates("item_id")
    return art.merge(alb, on="item_id", how="outer")


@dataclass
class SplitConfig:
    n_items: int = 10_000  # catalogue: most popular tracks before the validation window
    n_users: int = 2_500  # random users (fixed seed)
    min_user_pairs: int = 20  # users must have this many catalogue tracks before validation
    val_days: int = 14
    test_days: int = 14
    seed: int = 42


def make_subsample(events: pd.DataFrame, cfg: SplitConfig) -> tuple[pd.DataFrame, dict]:
    """Reproducible subsample: catalogue + users, both chosen from the pre-validation period."""
    t_end = int(events["timestamp"].max())
    t_test = t_end - cfg.test_days * DAY
    t_val = t_test - cfg.val_days * DAY

    before = events[(events["timestamp"] < t_val) & (events["event_type"] == "listen")]
    pop = before.groupby("item_id")["uid"].nunique().sort_values(ascending=False, kind="stable")
    catalogue = pop.index[: cfg.n_items].to_numpy()

    in_cat = before[before["item_id"].isin(catalogue)]
    user_pairs = in_cat.groupby("uid")["item_id"].nunique()
    eligible = np.sort(user_pairs.index[user_pairs >= cfg.min_user_pairs].to_numpy())
    rng = np.random.default_rng(cfg.seed)
    users = np.sort(rng.choice(eligible, size=min(cfg.n_users, len(eligible)), replace=False))

    sub = events[events["uid"].isin(users) & events["item_id"].isin(catalogue)].copy()
    info = dict(t_val=t_val, t_test=t_test, t_end=t_end, catalogue=catalogue, users=users,
                n_eligible_users=len(eligible), min_listeners=int(pop.iloc[cfg.n_items - 1]))
    return sub, info


def aggregate_pairs(ev: pd.DataFrame) -> pd.DataFrame:
    """Collapse events into one row per (user, item).

    n_listens   number of listen events
    n_good      listens with played_ratio >= 50 %
    mean_ratio  mean played ratio (%), capped at 100
    liked       the last like/unlike event is `like`
    disliked    the last dislike/undislike event is `dislike`
    first_ts / last_ts  first and last event time
    """
    ev = ev.sort_values(["uid", "timestamp"], kind="stable")
    listen = ev[ev["event_type"] == "listen"]
    g = listen.assign(
        good=(listen["played_ratio_pct"] >= GOOD_RATIO).astype(np.int32),
        ratio=listen["played_ratio_pct"].clip(upper=100),
        organic=listen["is_organic"].astype(np.int32),
    ).groupby(["uid", "item_id"], observed=True).agg(
        n_listens=("good", "size"), n_good=("good", "sum"), mean_ratio=("ratio", "mean"),
        n_organic=("organic", "sum"),
    )
    like = ev[ev["event_type"].isin(["like", "unlike"])]
    like = like.groupby(["uid", "item_id"], observed=True)["event_type"].last().eq("like").rename("liked")
    dis = ev[ev["event_type"].isin(["dislike", "undislike"])]
    dis = dis.groupby(["uid", "item_id"], observed=True)["event_type"].last().eq("dislike").rename("disliked")
    ts = ev.groupby(["uid", "item_id"], observed=True)["timestamp"].agg(first_ts="min", last_ts="max")
    out = ts.join([g, like, dis], how="left").reset_index()
    out[["n_listens", "n_good", "n_organic"]] = out[["n_listens", "n_good", "n_organic"]].fillna(0).astype(np.int32)
    out["mean_ratio"] = out["mean_ratio"].fillna(0.0)
    out["liked"] = out["liked"].astype("boolean").fillna(False).astype(bool)
    out["disliked"] = out["disliked"].astype("boolean").fillna(False).astype(bool)
    return out


def relevant_pairs(window_pairs: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Ground truth of an evaluation window (the same rule for every model).

    A track is relevant for a user if it is NEW for the user (no events before the
    window), the user listened to >= 50 % of it at least once or liked it,
    and did not dislike it in the window.
    """
    known = history[["uid", "item_id"]].assign(_known=True)
    w = window_pairs.merge(known, on=["uid", "item_id"], how="left")
    w = w[w["_known"].isna()]
    rel = ((w["n_good"] > 0) | w["liked"]) & ~w["disliked"]
    return w.loc[rel, ["uid", "item_id"]].reset_index(drop=True)


class Index:
    """Contiguous indices for users and items."""

    def __init__(self, users: np.ndarray, items: np.ndarray):
        self.users = np.asarray(users)
        self.items = np.asarray(items)
        self.u2i = pd.Series(np.arange(len(users)), index=users)
        self.i2i = pd.Series(np.arange(len(items)), index=items)

    def add(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["uidx"] = self.u2i.reindex(df["uid"]).to_numpy()
        df["iidx"] = self.i2i.reindex(df["item_id"]).to_numpy()
        return df.dropna(subset=["uidx", "iidx"]).astype({"uidx": np.int64, "iidx": np.int64})

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.users), len(self.items)


def count_matrix(pairs: pd.DataFrame, index: Index) -> sp.csr_matrix:
    """r_ui = number of listens (an event without listens, e.g. a like, counts as 1)."""
    p = index.add(pairs)
    r = np.maximum(p["n_listens"].to_numpy(np.float32), 1.0)
    X = sp.csr_matrix((r, (p["uidx"], p["iidx"])), shape=index.shape, dtype=np.float32)
    X.sort_indices()
    return X


def item_fields(events: pd.DataFrame, meta: pd.DataFrame, index: Index,
                train_counts: sp.csr_matrix) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Categorical item features for the FM; all are known at recommendation time.
    Returns (fields, artist index per item).

    artist, album     static catalogue metadata (unknown -> own category);
    length_bucket     median track length, 8 quantile bins;
    pop_bucket        number of listeners in TRAIN, 10 log-scale bins.
    """
    items = pd.Series(index.items, name="item_id")
    m = items.to_frame().merge(meta, on="item_id", how="left")
    artist = pd.factorize(m["artist_id"].fillna(-1))[0]
    album = pd.factorize(m["album_id"].fillna(-1))[0]
    length = events.loc[events["event_type"] == "listen"].groupby("item_id")["track_length_seconds"].median()
    length = items.map(length).fillna(length.median()).to_numpy()
    length_bucket = pd.qcut(length, 8, labels=False, duplicates="drop").astype(np.int64)
    listeners = np.asarray((train_counts > 0).sum(axis=0)).ravel()
    pop_bucket = np.minimum(np.floor(np.log1p(listeners) / np.log1p(listeners.max()) * 10), 9).astype(np.int64)
    return dict(artist=artist, album=album, length_bucket=length_bucket, pop_bucket=pop_bucket), artist


def user_artist_bag(train_counts: sp.csr_matrix, artist: np.ndarray, top_n: int = 50) -> sp.csr_matrix:
    """Users x artists: the user's `top_n` artists by sum of log(1 + listens) over their tracks,
    each row normalised to unit L1 norm."""
    n_art = int(artist.max()) + 1
    item_to_artist = sp.csr_matrix(
        (np.ones(len(artist), dtype=np.float32), (np.arange(len(artist)), artist)),
        shape=(len(artist), n_art),
    )
    X = train_counts.copy()
    X.data = np.log1p(X.data)
    bag = (X @ item_to_artist).tocsr()
    rows, cols, vals = [], [], []
    for u in range(bag.shape[0]):
        lo, hi = bag.indptr[u], bag.indptr[u + 1]
        keep = np.argsort(-bag.data[lo:hi], kind="stable")[:top_n]
        rows.append(np.full(len(keep), u))
        cols.append(bag.indices[lo:hi][keep])
        vals.append(bag.data[lo:hi][keep])
    bag = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                        shape=bag.shape)
    s = np.asarray(bag.sum(axis=1)).ravel()
    s[s == 0] = 1.0
    bag = sp.diags(1.0 / s) @ bag
    return bag.tocsr().astype(np.float32)
