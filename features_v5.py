import pandas as pd, numpy as np, lightgbm as lgb, time
from sklearn.model_selection import GroupShuffleSplit
from features_v4 import add_features

FEATS = ["u_in", "u_in_diff", "u_in_diff2", "u_in_rolling_mean", "u_out", "time_step", "R", "C",
         "u_in_cumsum", "u_in_dt_cumsum", "u_in_lag1", "u_in_lag2", "u_in_lag3",
         "RC", "time_over_RC", "R_x_cumsum", "C_x_cumsum", "vol_over_C", "R_x_u_in",
         "u_in_lead1", "u_in_lead2", "u_in_lead3", "vol_lag1", "vol_lag2", "vol_lag3", "vol_last5", "vol_last10",
         "total_insp_vol", "vol_frac", "total_vol_over_C", "u_in_max", "u_in_mean"]

if __name__ == "__main__":
    tr = add_features(pd.read_parquet("data/train.parquet"))
    insp = tr[tr.u_out == 0]
    tri, vai = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(insp, groups=insp.breath_id))
    tr_sample = np.random.default_rng(0).choice(tri, size=len(vai), replace=False)
    X, y = insp[FEATS], insp.pressure

    common = dict(objective="mae", n_estimators=10000, learning_rate=0.05, num_leaves=127, verbose=-1)
    reg = dict(min_child_samples=200,   # default 20: leaves must cover many rows, harder to fit single breaths
               colsample_bytree=0.7,    # each tree sees 70% of features, so it can't always lean on breath-level ones
               subsample=0.8, subsample_freq=1,  # each tree sees 80% of rows
               reg_lambda=1.0)          # L2 penalty on leaf values
    for name, params in [("lr 0.05, no regularization", common), ("lr 0.05 + regularization", {**common, **reg})]:
        t = time.time()
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tri], y.iloc[tri], eval_set=[(X.iloc[vai], y.iloc[vai])], callbacks=[lgb.early_stopping(200, verbose=False)])
        va = np.abs(m.predict(X.iloc[vai]) - y.iloc[vai]).mean()
        trm = np.abs(m.predict(X.iloc[tr_sample]) - y.iloc[tr_sample]).mean()
        imp = pd.Series(m.booster_.feature_importance("gain"), FEATS); imp = (imp/imp.sum()).round(3).sort_values(ascending=False)
        print(f"\n=== {name}: val MAE = {va:.3f}  train MAE = {trm:.3f}  gap = {va-trm:.3f}  iters={m.best_iteration_}  ({time.time()-t:.0f}s)\n{imp.head(10).to_string()}", flush=True)
