"""3D CNN for T1w brain volumes — regression (age) or binary classification (sex).

Deliberately small. The federated cohorts here are tens of scans, not
thousands, so capacity is the enemy: a wide 3D ResNet memorises 40 volumes
perfectly and tells you nothing. Four stride-2 blocks with GroupNorm, ~900k
parameters, global pooling instead of a large dense head.

GroupNorm rather than BatchNorm on purpose. BatchNorm statistics are computed
per batch and get averaged across sites during federated aggregation, which
leaks cohort information and destabilises training when each site holds a
handful of scans. GroupNorm is per-sample and carries no such state.
"""
from __future__ import annotations
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> SiLU, twice, residual, then downsample by 2."""

    def __init__(self, cin: int, cout: int, groups: int = 8):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(min(groups, cout), cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(min(groups, cout), cout)
        self.skip = nn.Conv3d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.silu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        x = F.silu(h + self.skip(x))
        return F.max_pool3d(x, 2)


class BrainCNN(nn.Module):
    """3D CNN over a single-channel volume.

    task="sex" -> one logit (BCE). task="age" -> scalar offset from age_offset.

    Predicting the residual from a fixed offset for regression keeps the head
    near zero at init, which matters when every site starts from identical
    weights and averages after one local epoch.
    """

    def __init__(self, widths=(16, 32, 64, 128), dropout=0.4,
                 task: str = "sex", age_offset: float = 25.0,
                 stem_stride: int = 2):
        super().__init__()
        self.task, self.age_offset = task, age_offset
        # Stride-2 stem: the wide-channel activations live at half resolution,
        # which is what keeps peak memory small enough for the training sandbox
        # (a full-resolution 16 x D^3 activation with its autograd buffers is
        # tens of MB and overflows a tightly-capped node). The INPUT volume is
        # still full resolution — only the feature maps are coarser, exactly as
        # a 3D ResNet stem does. Set stem_stride=1 to keep full-res features on
        # a node with plenty of RAM.
        self.stem = nn.Conv3d(1, widths[0], 3, stride=stem_stride,
                              padding=1, bias=False)
        blocks, cin = [], widths[0]
        for w in widths:
            blocks.append(ConvBlock(cin, w))
            cin = w
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(cin, 64), nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.dim() == 4:                      # (N, D, H, W) -> add channel
            x = x.unsqueeze(1)
        h = self.blocks(self.stem(x))
        h = F.adaptive_avg_pool3d(h, 1)
        out = self.head(h).squeeze(-1)
        return out if self.task == "sex" else out + self.age_offset


# Kept so older imports still resolve.
class BrainAgeCNN(BrainCNN):
    def __init__(self, *a, **kw):
        kw.setdefault("task", "age")
        super().__init__(*a, **kw)


# ── training ────────────────────────────────────────────────────────────────

def make_loss(task: str, labels: np.ndarray):
    """Loss for the task, with class imbalance handled for classification.

    Both cohorts here skew female (67% / 59%), so an unweighted BCE happily
    predicts "female" forever and scores at the majority baseline. pos_weight
    rebalances the positive (male) class.
    """
    if task == "sex":
        n_pos = float((labels == 1).sum())
        n_neg = float((labels == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return lambda p, y: bce(p, y)
    # Huber, not MSE: with cohorts this small one outlier dominates the gradient.
    return lambda p, y: F.smooth_l1_loss(p, y, beta=2.0)


def train_one_epoch(model, loader, opt, loss_fn=None, device="cpu"):
    model.train()
    if loss_fn is None:
        loss_fn = make_loss(getattr(model, "task", "age"), np.array([0.0, 1.0]))
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item() * len(y); n += len(y)
    return tot / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    """Regression: MAE / RMSE / R^2. Classification: acc / balanced acc / AUC."""
    model.eval()
    P, Y = [], []
    for x, y in loader:
        P.append(model(x.to(device)).cpu()); Y.append(y)
    if not P:
        return {"n": 0}
    p, t = torch.cat(P), torch.cat(Y)

    if getattr(model, "task", "age") == "sex":
        prob = torch.sigmoid(p).numpy()
        yt = t.numpy()
        pred = (prob > 0.5).astype(np.float32)
        acc = float((pred == yt).mean())
        # Balanced accuracy is the honest number on a skewed cohort: plain
        # accuracy rewards always guessing the majority class.
        accs = [float((pred[yt == c] == c).mean()) for c in (0.0, 1.0) if (yt == c).any()]
        bal = float(np.mean(accs)) if accs else float("nan")
        return {"acc": acc, "balanced_acc": bal, "auc": _auc(yt, prob), "n": len(yt)}

    err = p - t
    ss_res = (err ** 2).sum().item()
    ss_tot = ((t - t.mean()) ** 2).sum().item()
    return {"mae": err.abs().mean().item(),
            "rmse": (err ** 2).mean().sqrt().item(),
            "r2": (1 - ss_res / ss_tot) if ss_tot > 1e-6 else float("nan"),
            "n": len(t)}


def _auc(y, score):
    """ROC AUC via rank statistic; no sklearn dependency."""
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def baseline(task: str, train_y: np.ndarray, val_y: np.ndarray) -> float:
    """What "learned nothing" scores. Beat this or the images were ignored."""
    if task == "sex":
        return float(max((val_y == 0).mean(), (val_y == 1).mean()))
    return float(np.abs(val_y - float(np.mean(train_y))).mean())


def baseline_mae(train_y, val_y):
    return baseline("age", train_y, val_y)
