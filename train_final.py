import pandas as pd, numpy as np, lightgbm as lgb, time
from sklearn.model_selection import GroupShuffleSplit
from features_v4 import add_features
from features_v5 import FEATS

PARAMS = dict(objective="mae", learning_rate=0.05, num_leaves=127, verbose=-1,
              min_child_samples=200, colsample_bytree=0.7, subsample=0.8, subsample_freq=1, reg_lambda=1.0)

tr = add_features(pd.read_parquet("data/train.parquet"))
te = add_features(pd.read_parquet("data/test.parquet"))
insp = tr[tr.u_out == 0]
X, y = insp[FEATS], insp.pressure

# Step 1: holdout estimate + pick tree count
t = time.time()
tri, vai = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(insp, groups=insp.breath_id))
m = lgb.LGBMRegressor(n_estimators=15000, **PARAMS)
m.fit(X.iloc[tri], y.iloc[tri], eval_set=[(X.iloc[vai], y.iloc[vai])], callbacks=[lgb.early_stopping(200, verbose=False)])
pv = m.predict(X.iloc[vai]); err = np.abs(pv - y.iloc[vai].values)
print(f"STEP 1 holdout: val MAE = {err.mean():.4f}  best_iter = {m.best_iteration_}  ({time.time()-t:.0f}s)", flush=True)
rc = insp.iloc[vai][["R", "C"]].assign(err=err).groupby(["R", "C"]).err.mean().round(3)
print("val MAE by R,C:\n" + rc.to_string(), flush=True)

# Step 2: refit on all training breaths, predict test
t = time.time(); n_full = int(m.best_iteration_ * 1.1)
mf = lgb.LGBMRegressor(n_estimators=n_full, **PARAMS).fit(X, y)
sub = pd.DataFrame({"id": te["id"], "pressure": mf.predict(te[FEATS])})
sub.to_csv("submission.csv", index=False)
print(f"STEP 2 full refit: trees = {n_full}  ({time.time()-t:.0f}s)", flush=True)

# Step 3: format checks
ss = pd.read_csv("data/raw/sample_submission.csv")
print("CHECK columns match:", list(sub.columns) == list(ss.columns))
print("CHECK ids match in order:", sub["id"].equals(ss["id"]))
print("CHECK rows:", len(sub), "NaNs:", sub.pressure.isna().sum())
ins = te.u_out == 0
print("test pred (scored rows):", sub.pressure[ins.values].describe().round(2).to_dict())
print("train pressure (insp):   ", y.describe().round(2).to_dict())
