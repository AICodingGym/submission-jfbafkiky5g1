"""Build (breaths, 80, features) arrays for the sequence model and cache them."""
import os
import numpy as np, pandas as pd
from features_v4 import add_features
from features_v5 import FEATS

CACHE = "data/seq_cache.npz"
STEPS = 80


def build():
    tr = add_features(pd.read_parquet("data/train.parquet")).sort_values(["breath_id", "time_step"])
    te = add_features(pd.read_parquet("data/test.parquet")).sort_values(["breath_id", "time_step"])
    for df in (tr, te):
        for r in (5, 20, 50):
            df[f"R_{r}"] = (df["R"] == r).astype(np.float32)
        for c in (10, 20, 50):
            df[f"C_{c}"] = (df["C"] == c).astype(np.float32)
    feats = FEATS + [f"R_{r}" for r in (5, 20, 50)] + [f"C_{c}" for c in (10, 20, 50)]

    mu = tr[feats].mean()
    sd = tr[feats].std().replace(0, 1)  # scaler fit on train only

    def to_seq(df):
        x = ((df[feats] - mu) / sd).to_numpy(np.float32)
        return x.reshape(-1, STEPS, len(feats))

    X, Xt = to_seq(tr), to_seq(te)
    assert np.isfinite(X).all() and np.isfinite(Xt).all()
    np.savez(CACHE,
             X=X, y=tr["pressure"].to_numpy(np.float32).reshape(-1, STEPS),
             mask=(tr["u_out"] == 0).to_numpy(np.float32).reshape(-1, STEPS),
             R=tr["R"].to_numpy().reshape(-1, STEPS)[:, 0], C=tr["C"].to_numpy().reshape(-1, STEPS)[:, 0],
             Xt=Xt, test_mask=(te["u_out"] == 0).to_numpy(np.float32).reshape(-1, STEPS),
             test_ids=te["id"].to_numpy().reshape(-1, STEPS), feats=np.array(feats))


def load():
    if not os.path.exists(CACHE):
        build()
    return dict(np.load(CACHE))


if __name__ == "__main__":
    build()
    d = load()
    print({k: v.shape for k, v in d.items()})
