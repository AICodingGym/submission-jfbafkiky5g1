import pandas as pd, numpy as np
tr = pd.read_csv("data/raw/train.csv"); te = pd.read_csv("data/raw/test.csv")
tr.to_parquet("data/train.parquet"); te.to_parquet("data/test.parquet")
for n, d in [("train", tr), ("test", te)]:
    g = d.groupby("breath_id")
    print(f"{n}: rows={len(d):,} breaths={d.breath_id.nunique():,} steps/breath={g.size().unique()}")
print("breath overlap:", len(set(tr.breath_id) & set(te.breath_id)))
print("\nR,C combos (train share vs test share):")
rc = pd.concat([tr.groupby(["R","C"]).breath_id.nunique().rename("train")/tr.breath_id.nunique(),
                te.groupby(["R","C"]).breath_id.nunique().rename("test")/te.breath_id.nunique()],axis=1).round(3)
print(rc)
print("\nu_out==0 (scored) share:", round((tr.u_out==0).mean(),3))
print("inspiratory steps/breath:", tr[tr.u_out==0].groupby("breath_id").size().describe().round(1).to_dict())
print("\npressure:", tr.pressure.describe().round(2).to_dict())
print("unique pressure values:", tr.pressure.nunique())
print("u_in:", tr.u_in.describe().round(2).to_dict())
print("time_step max:", tr.time_step.max().round(3), "dt median:", tr.groupby("breath_id").time_step.diff().median().round(4))
# Baselines on inspiratory phase
m = tr.u_out==0
print("\nBaseline MAE (insp): global median =", (tr.pressure[m]-tr.pressure[m].median()).abs().mean().round(3))
med = tr[m].groupby(["R","C"]).pressure.transform("median")
print("Baseline MAE (insp): R,C median =", (tr.pressure[m]-med).abs().mean().round(3))
print("\ncorr with pressure (insp):")
s = tr[m].copy(); s["u_in_cumsum"]=tr.groupby("breath_id").u_in.cumsum()[m]
print(s[["u_in","u_in_cumsum","time_step","R","C"]].corrwith(s.pressure).round(3))
