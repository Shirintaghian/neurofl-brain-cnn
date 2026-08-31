#!/usr/bin/env python3
"""Train the 3D CNN directly on a raw BIDS tree. No preprocessing step."""
import argparse, sys, time
import numpy as np, torch
sys.path.insert(0, "/site/experiment")
from raw_dataset import RawBIDSDataset, subject_split
from model import BrainCNN, make_loss, train_one_epoch, evaluate, baseline

ap = argparse.ArgumentParser()
ap.add_argument("bids_root")
ap.add_argument("--target", default="sex", choices=["age", "sex"])
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--shape", default="48,48,48")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
shape = tuple(int(x) for x in a.shape.split(","))

full = RawBIDSDataset(a.bids_root, target=a.target, shape=shape)
if len(full) == 0:
    sys.exit(f"no usable scans under {a.bids_root}")
tr_idx, va_idx = subject_split(full, seed=a.seed)
tr_ds = RawBIDSDataset(a.bids_root, a.target, shape, indices=tr_idx, train=True)
va_ds = RawBIDSDataset(a.bids_root, a.target, shape, indices=va_idx)
ytr, yva = tr_ds.labels, va_ds.labels

print(f"{len(full)} scans / {len(set(full.subjects))} subjects -> "
      f"train {len(tr_ds)} ({len(set(tr_ds.subjects))} subj), "
      f"val {len(va_ds)} ({len(set(va_ds.subjects))} subj)")
base = baseline(a.target, ytr, yva)
if a.target == "sex":
    print(f"train F/M {int((ytr==0).sum())}/{int((ytr==1).sum())} | "
          f"val F/M {int((yva==0).sum())}/{int((yva==1).sum())} | "
          f"majority baseline {100*base:.0f}%")
else:
    print(f"baseline (predict mean {ytr.mean():.1f}) val MAE {base:.2f} y")

tr = torch.utils.data.DataLoader(tr_ds, batch_size=a.batch, shuffle=True)
va = torch.utils.data.DataLoader(va_ds, batch_size=a.batch)

model = BrainCNN(task=a.target, age_offset=float(ytr.mean()))
loss_fn = make_loss(a.target, ytr)
print("parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

best, t0 = 0.0 if a.target == "sex" else float("inf"), time.time()
for ep in range(1, a.epochs + 1):
    loss = train_one_epoch(model, tr, opt, loss_fn); sched.step()
    m = evaluate(model, va)
    if a.target == "sex":
        cur = m["balanced_acc"]; best = max(best, cur)
        line = f"acc {100*m['acc']:.0f}%  bal-acc {100*cur:.0f}%  auc {m['auc']:.2f}"
    else:
        cur = m["mae"]; best = min(best, cur)
        line = f"MAE {cur:.2f}  RMSE {m['rmse']:.2f}"
    if ep % 5 == 0 or ep == 1:
        print(f"epoch {ep:3d}  loss {loss:.3f}  {line}")

el = time.time() - t0
if a.target == "sex":
    print(f"\nbest balanced acc {100*best:.0f}% vs majority baseline {100*base:.0f}%  ({el:.0f}s)")
    print("VERDICT:", "BEATS baseline" if best > base else "does NOT beat baseline")
else:
    print(f"\nbest val MAE {best:.2f} y vs baseline {base:.2f} y  ({el:.0f}s)")
    print("VERDICT:", "BEATS baseline" if best < base else "does NOT beat baseline")
