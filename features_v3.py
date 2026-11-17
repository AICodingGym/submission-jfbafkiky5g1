import pandas as pd, numpy as np, lightgbm as lgb, time
from sklearn.model_selection import GroupShuffleSplit

def add_features(df):
    g = df.groupby("breath_id")
    # v1
    df["u_in_diff"] = g["u_in"].diff().fillna(0)
    df["u_in_diff2"] = df.groupby("breath_id")["u_in_diff"].diff().fillna(0)
    df["u_in_rolling_mean"] = g["u_in"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    # v2
    df["dt"] = g["time_step"].diff().fillna(0)
    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["u_in_dt_cumsum"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()
    for k in (1, 2, 3):
        df[f"u_in_lag{k}"] = g["u_in"].shift(k).fillna(0)
    df["RC"] = df["R"] * df["C"]
    df["time_over_RC"] = df["time_step"] / df["RC"]
    df["R_x_cumsum"] = df["R"] * df["u_in_cumsum"]
    df["C_x_cumsum"] = df["C"] * df["u_in_cumsum"]
    df["vol_over_C"] = df["u_in_dt_cumsum"] / df["C"]
    df["R_x_u_in"] = df["R"] * df["u_in"]
    # v3: future valve values (full control sequence is known at test time)
    for k in (1, 2, 3):
        df[f"u_in_lead{k}"] = g["u_in"].shift(-k).fillna(0)
    # v3: volume history
    gv = df.groupby("breath_id")["u_in_dt_cumsum"]
    for k in (1, 2, 3):
        df[f"vol_lag{k}"] = gv.shift(k).fillna(0)
    for w in (5, 10):  # volume delivered over the last w steps
        df[f"vol_last{w}"] = df["u_in_dt_cumsum"] - gv.shift(w).fillna(0)
    return df

if __name__ == "__main__":
    tr = add_features(pd.read_parquet("data/train.parquet"))
    insp = tr[tr.u_out == 0]
    tri, vai = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(insp, groups=insp.breath_id))

    v2 = ["u_in", "u_in_diff", "u_in_diff2", "u_in_rolling_mean", "u_out", "time_step", "R", "C",
          "u_in_cumsum", "u_in_dt_cumsum", "u_in_lag1", "u_in_lag2", "u_in_lag3",
          "RC", "time_over_RC", "R_x_cumsum", "C_x_cumsum", "vol_over_C", "R_x_u_in"]
    leads = ["u_in_lead1", "u_in_lead2", "u_in_lead3"]
    volhist = ["vol_lag1", "vol_lag2", "vol_lag3", "vol_last5", "vol_last10"]
    runs = [("v2 + future u_in", v2+leads), ("v2 + volume history", v2+volhist), ("v2 + both", v2+leads+volhist)]
    for name, feats in runs:
        t = time.time(); X, y = insp[feats], insp.pressure
        m = lgb.LGBMRegressor(objective="mae", n_estimators=4000, learning_rate=0.1, num_leaves=127, verbose=-1)
        m.fit(X.iloc[tri], y.iloc[tri], eval_set=[(X.iloc[vai], y.iloc[vai])], callbacks=[lgb.early_stopping(100, verbose=False)])
        mae = np.abs(m.predict(X.iloc[vai]) - y.iloc[vai]).mean()
        imp = pd.Series(m.booster_.feature_importance("gain"), feats); imp = (imp/imp.sum()).round(3).sort_values(ascending=False)
        print(f"\n=== {name}: val MAE = {mae:.3f}  iters={m.best_iteration_}  ({time.time()-t:.0f}s)\n{imp.head(10).to_string()}", flush=True)
