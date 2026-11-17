"""Score LSTM out-of-fold predictions, test grid snapping, and build submission.csv from the fold median."""
import glob, sys
import numpy as np, pandas as pd
from seq_data import load

TAG = sys.argv[1] if len(sys.argv) > 1 else "lstm"
P_MIN, P_STEP, P_N = -1.8957443, 0.070302, 950  # pressure grid seen in train


def snap(p):
    return P_MIN + np.clip(np.round((p - P_MIN) / P_STEP), 0, P_N - 1) * P_STEP


d = load()
y, mask, R, C = d["y"], d["mask"], d["R"], d["C"]
oof_files = sorted(glob.glob(f"oof/{TAG}_oof_fold*.npy"))
test_files = sorted(glob.glob(f"oof/{TAG}_test_fold*.npy"))
print(f"folds found: {len(oof_files)} OOF, {len(test_files)} test")

# OOF: each file holds [breath index, preds for the first S steps]
idx, preds = [], []
for f in oof_files:
    a = np.load(f); idx.append(a[:, 0].astype(int)); preds.append(a[:, 1:])
idx, preds = np.concatenate(idx), np.concatenate(preds)
S = preds.shape[1]
yy, mm = y[idx, :S], mask[idx, :S]
err = np.abs(preds - yy) * mm
print(f"\nOOF MAE ({len(idx):,} breaths): {err.sum() / mm.sum():.4f}")
err_snap = np.abs(snap(preds) - yy) * mm
print(f"OOF MAE snapped to grid:     {err_snap.sum() / mm.sum():.4f}")
use_snap = err_snap.sum() < err.sum()

per = pd.DataFrame({"R": R[idx], "C": C[idx], "e": err.sum(1), "n": mm.sum(1)}).groupby(["R", "C"]).sum()
print("\nOOF MAE by R,C:\n" + (per.e / per.n).round(3).unstack().to_string())

# Test: median across folds, pad unscored steps (>= S) with 0
tp = np.median(np.stack([np.load(f) for f in test_files]), axis=0)
if use_snap:
    tp = snap(tp)
full = np.zeros(d["test_ids"].shape, dtype=np.float64); full[:, :S] = tp
sub = pd.DataFrame({"id": d["test_ids"].ravel(), "pressure": full.ravel()}).sort_values("id").reset_index(drop=True)
sub.to_csv("submission.csv", index=False)
print(f"\nwrote submission.csv (snapped={use_snap})")

ss = pd.read_csv("data/raw/sample_submission.csv")
print("CHECK columns match:", list(sub.columns) == list(ss.columns))
print("CHECK ids match in order:", sub["id"].equals(ss["id"]))
print("CHECK rows:", len(sub), "NaNs:", int(sub.pressure.isna().sum()))
scored = d["test_mask"].ravel()[np.argsort(d["test_ids"].ravel())] == 1
print("test pred (scored rows):", sub.pressure[scored].describe().round(2).to_dict())
print("train pressure (scored): ", pd.Series(y[mask == 1]).describe().round(2).to_dict())
