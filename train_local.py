#!/usr/bin/env python3
"""Train the CNN on one site's prepared data. Sanity check before federating."""
import argparse, json, sys
import numpy as np, torch
sys.path.insert(0, "/site/experiment")
from model import (BrainAgeCNN, VolumeDataset, load_split,
                   train_one_epoch, evaluate, baseline_mae)

ap = argparse.ArgumentParser()
ap.add_argument("data_dir"); ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=4); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
(Xtr, ytr), (Xva, yva), info = load_split(a.data_dir, seed=a.seed)
print("split:", json.dumps(info))
print(f"train ages {ytr.min():.0f}-{ytr.max():.0f} | val ages {yva.min():.0f}-{yva.max():.0f}")

tr = torch.utils.data.DataLoader(VolumeDataset(Xtr, ytr, True), batch_size=a.batch, shuffle=True)
va = torch.utils.data.DataLoader(VolumeDataset(Xva, yva), batch_size=a.batch)

model = BrainAgeCNN(age_offset=float(ytr.mean()))
print("parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

base = baseline_mae(ytr, yva)
print(f"baseline (predict train mean {ytr.mean():.1f}): val MAE {base:.2f} y\n")

best = float("inf")
for ep in range(1, a.epochs + 1):
    loss = train_one_epoch(model, tr, opt); sched.step()
    m = evaluate(model, va)
    best = min(best, m["mae"])
    if ep % 5 == 0 or ep == 1:
        print(f"epoch {ep:3d}  loss {loss:.3f}  val MAE {m['mae']:.2f}  RMSE {m['rmse']:.2f}")
print(f"\nbest val MAE {best:.2f} y vs baseline {base:.2f} y")
print("VERDICT:", "beats baseline" if best < base else "does NOT beat baseline")
