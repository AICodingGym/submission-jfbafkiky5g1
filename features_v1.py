import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

def add_features(df):
    g = df.groupby("breath_id")["u_in"]
    df["u_in_diff"] = g.diff().fillna(0)
    df["u_in_diff2"] = df.groupby("breath_id")["u_in_diff"].diff().fillna(0)
    df["u_in_rolling_mean"] = g.transform(lambda s: s.rolling(5, min_periods=1).mean())
    return df

tr = add_features(pd.read_parquet("data/train.parquet"))
base = ["u_in", "u_in_diff", "u_in_diff2", "u_in_rolling_mean", "u_out", "time_step"]
insp = tr[tr.u_out == 0]  # metric only scores inspiratory phase
tri, vai = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(insp, groups=insp.breath_id))
for name, feats in [("user set", base), ("user set + R,C", base + ["R", "C"])]:
    X, y = insp[feats], insp.pressure
    m = lgb.LGBMRegressor(objective="mae", n_estimators=2000, learning_rate=0.1, num_leaves=127, verbose=-1)
    m.fit(X.iloc[tri], y.iloc[tri], eval_set=[(X.iloc[vai], y.iloc[vai])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    mae = np.abs(m.predict(X.iloc[vai]) - y.iloc[vai]).mean()
    imp = pd.Series(m.booster_.feature_importance("gain"), feats); imp = (imp/imp.sum()).round(3).sort_values(ascending=False)
    print(f"\n{name}: val MAE = {mae:.3f}  (iters={m.best_iteration_})\n{imp.to_string()}")
