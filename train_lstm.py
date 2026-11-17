"""10-fold bidirectional LSTM. Saves each fold's best model, OOF and test predictions so runs can resume."""
import argparse, os, time
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from seq_data import load

p = argparse.ArgumentParser()
p.add_argument("--folds", type=int, default=10)
p.add_argument("--only", type=int, nargs="*", help="run just these fold indices")
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--patience", type=int, default=20)
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--batch", type=int, default=512)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--steps", type=int, default=80, help="keep only the first N steps (scored phase ends by step 32)")
p.add_argument("--tag", default="lstm")
args = p.parse_args()

dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
os.makedirs("models", exist_ok=True); os.makedirs("oof", exist_ok=True)
d = load()
S = args.steps
X, y, mask, Xt = d["X"][:, :S], d["y"][:, :S], d["mask"][:, :S], d["Xt"][:, :S]
X, y, mask, Xt = (np.ascontiguousarray(a) for a in (X, y, mask, Xt))


class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.lstm = nn.LSTM(n_in, args.hidden, args.layers, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(2 * args.hidden, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.lstm(x)[0]).squeeze(-1)


def predict(model, A):
    model.eval(); out = []
    if dev == "mps": torch.mps.empty_cache()
    with torch.no_grad():
        for i in range(0, len(A), 512):
            out.append(model(torch.from_numpy(A[i:i + 512]).to(dev)).cpu().numpy())
    return np.concatenate(out)


def masked_mae(pred, yy, mm):
    return float((np.abs(pred - yy) * mm).sum() / mm.sum())


# GroupKFold is deterministic; shuffle breaths first so folds aren't ordered by breath_id
perm = np.random.default_rng(0).permutation(len(X))
folds = [(perm[a], perm[b]) for a, b in GroupKFold(args.folds).split(perm, groups=perm)]

for k in (args.only if args.only is not None else range(args.folds)):
    out_test = f"oof/{args.tag}_test_fold{k}.npy"
    if os.path.exists(out_test):
        print(f"fold {k}: already done, skipping", flush=True); continue
    tri, vai = folds[k]
    model = Net(X.shape[2]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6)
    Xtr, ytr, mtr = (torch.from_numpy(a[tri]).to(dev) for a in (X, y, mask))  # keep train data on GPU
    tr_eval = np.random.default_rng(k).choice(tri, size=len(vai), replace=False)
    best, best_ep, t0 = np.inf, 0, time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); te = time.time()
        for idx in torch.randperm(len(tri), device=dev).split(args.batch):
            xb, yb, mb = Xtr[idx], ytr[idx], mtr[idx]
            loss = ((model(xb) - yb).abs() * mb).sum() / mb.sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        va = masked_mae(predict(model, X[vai]), y[vai], mask[vai])
        sched.step(va)
        if va < best:
            best, best_ep = va, ep
            torch.save(model.state_dict(), f"models/{args.tag}_fold{k}.pt")
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            trm = masked_mae(predict(model, X[tr_eval]), y[tr_eval], mask[tr_eval])
            print(f"fold {k} ep {ep:3d}  train {trm:.4f}  val {va:.4f}  gap {va - trm:.4f}  best {best:.4f}@{best_ep}  "
                  f"lr {opt.param_groups[0]['lr']:.1e}  {time.time() - te:.0f}s/ep"
                  + (f"  mem {torch.mps.driver_allocated_memory() / 2**30:.1f}GiB" if dev == "mps" else ""), flush=True)
        if ep - best_ep >= args.patience:
            print(f"fold {k}: early stop at ep {ep}", flush=True); break
    model.load_state_dict(torch.load(f"models/{args.tag}_fold{k}.pt"))
    oof = predict(model, X[vai])
    np.save(f"oof/{args.tag}_oof_fold{k}.npy", np.c_[vai, oof])  # breath index + 80 preds
    np.save(out_test, predict(model, Xt))
    del Xtr, ytr, mtr, model, opt
    if dev == "mps": torch.mps.empty_cache()
    print(f"fold {k}: DONE best val {best:.4f} @ ep {best_ep}  ({time.time() - t0:.0f}s)", flush=True)
