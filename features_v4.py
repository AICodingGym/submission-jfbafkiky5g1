import pandas as pd, numpy as np, lightgbm as lgb, time
from sklearn.model_selection import GroupShuffleSplit
from features_v3 import add_features as add_v3  # guarded below

def add_features(df):
    df = add_v3(df)
    # v4: breath-level volume (inspiratory phase only, from control inputs)
    is_insp = df["u_out"] == 0
    flow = (df["u_in"] * df["dt"]).where(is_insp, 0)
    gb = df["breath_id"]
    df["total_insp_vol"] = flow.groupby(gb).transform("sum")
    df["vol_frac"] = (df["u_in_dt_cumsum"] / df["total_insp_vol"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["total_vol_over_C"] = df["total_insp_vol"] / df["C"]
    df["u_in_max"] = df["u_in"].where(is_insp).groupby(gb).transform("max")
    df["u_in_mean"] = df["u_in"].where(is_insp).groupby(gb).transform("mean")
    return df

if __name__ == "__main__":
    tr = add_features(pd.read_parquet("data/train.parquet"))
    insp = tr[tr.u_out == 0]
    tri, vai = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(insp, groups=insp.breath_id))
    rng = np.random.default_rng(0); tr_sample = rng.choice(tri, size=len(vai), replace=False)

    v2 = ["u_in", "u_in_diff", "u_in_diff2", "u_in_rolling_mean", "u_out", "time_step", "R", "C",
          "u_in_cumsum", "u_in_dt_cumsum", "u_in_lag1", "u_in_lag2", "u_in_lag3",
          "RC", "time_over_RC", "R_x_cumsum", "C_x_cumsum", "vol_over_C", "R_x_u_in"]
    v3 = v2 + ["u_in_lead1", "u_in_lead2", "u_in_lead3", "vol_lag1", "vol_lag2", "vol_lag3", "vol_last5", "vol_last10"]
    voltot = ["total_insp_vol", "vol_frac", "total_vol_over_C"]
    stats = ["u_in_max", "u_in_mean"]
    runs = [("v3 (baseline)", v3), ("v3 + breath volume totals", v3+voltot), ("v3 + totals + u_in max/mean", v3+voltot+stats)]
    for name, feats in runs:
        t = time.time(); X, y = insp[feats], insp.pressure
        m = lgb.LGBMRegressor(objective="mae", n_estimators=4000, learning_rate=0.1, num_leaves=127, verbose=-1)
        m.fit(X.iloc[tri], y.iloc[tri], eval_set=[(X.iloc[vai], y.iloc[vai])], callbacks=[lgb.early_stopping(100, verbose=False)])
        va = np.abs(m.predict(X.iloc[vai]) - y.iloc[vai]).mean()
        trm = np.abs(m.predict(X.iloc[tr_sample]) - y.iloc[tr_sample]).mean()
        imp = pd.Series(m.booster_.feature_importance("gain"), feats); imp = (imp/imp.sum()).round(3).sort_values(ascending=False)
        print(f"\n=== {name}: val MAE = {va:.3f}  train MAE = {trm:.3f}  gap = {va-trm:.3f}  iters={m.best_iteration_}  ({time.time()-t:.0f}s)\n{imp.head(10).to_string()}", flush=True)
