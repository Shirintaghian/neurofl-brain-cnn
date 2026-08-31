"""shared.py — model + data loading + train/eval for the NeuroFL brain CNN.

Equivalent to a tabular modeller's shared.py, but for 3D imaging: it holds the
3D CNN (BrainCNN), the raw-BIDS volume loader (RawBIDSDataset), and the
train/evaluate helpers. client_app_raw.py and server_app.py import from here.
"""
import csv, pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib

# ============================ MODEL ============================
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


# ====================== DATA (raw BIDS) =======================
def scan_bids(root) -> list[tuple[str, pathlib.Path]]:
    """(subject, T1w path) for every readable anatomical scan.

    Handles both flat (sub-XX/anat) and session (sub-XX/ses-YY/anat) layouts.
    Dangling git-annex symlinks are skipped: they glob fine and fail on open,
    which would otherwise crash mid-epoch rather than at startup.
    """
    root = pathlib.Path(root)
    out = []
    for pat in ("sub-*/anat/*T1w.nii.gz", "sub-*/ses-*/anat/*T1w.nii.gz"):
        for f in sorted(root.glob(pat)):
            if not f.is_file():
                continue
            sub = next(p for p in f.parts if p.startswith("sub-"))
            out.append((sub, f))
    return out


def read_participants(root) -> dict:
    root = pathlib.Path(root)
    meta = {}
    tsv = root / "participants.tsv"
    if not tsv.is_file():
        return meta
    with tsv.open(encoding="utf-8-sig") as f:      # some BIDS files carry a BOM
        for r in csv.DictReader(f, delimiter="\t"):
            pid = (r.get("participant_id") or "").strip()
            if pid:
                meta[pid] = {
                    "age": (r.get("age") or "").strip(),
                    # `sex` in most datasets, `gender` in others (e.g. ds000102)
                    "sex": (r.get("sex") or r.get("gender") or "").strip().upper(),
                    "group": (r.get("group") or "").strip(),
                }
    return meta


def to_volume(path, out_shape=(64, 64, 64), fast=False) -> np.ndarray:
    """One NIfTI -> normalised float32 volume.

    fast=False (default): full-resolution trilinear resample — best fidelity.
    fast=True: a coarse strided pre-downsample before the resize, several times
    faster at negligible quality cost for these target shapes, and lighter on
    memory-constrained nodes.
    """
    # Load as float32 from the start. Sandbox RAM is tight, so every step below
    # is written to avoid float64 intermediates and extra full-size copies — a
    # single (176,256,256) float64 temporary is ~88 MiB and overflows the jail.
    #
    # nibabel's ArrayProxy applies its scale/offset in float64 and returns a
    # float64 array; np.asanyarray(..., dtype=float32) then copies, so the peak
    # is a float64 whole-volume temporary. Downsample the volume with nibabel's
    # in-image resampling FIRST (reads at reduced resolution) is not available
    # without extra deps, so instead read the un-scaled native array via
    # get_unscaled() (usually int16, ~half the RAM) and apply the scale factor
    # ourselves in float32.
    # Sandbox RAM is very tight (a single full-res float32 volume, ~44 MiB, can
    # fail to allocate), so DOWNSAMPLE FIRST and do every heavy op (clip, crop,
    # z-score) on the small out_shape volume. Peak memory is then one native
    # int16 array (~22 MiB) briefly, never a full-res float32/float64 temporary.
    img = nib.load(str(path))
    try:
        raw = np.asarray(img.dataobj.get_unscaled())   # native dtype, e.g. int16
        sl = img.dataobj.slope
        inter = img.dataobj.inter
    except Exception:
        raw = np.asanyarray(img.dataobj)               # last resort
        sl, inter = None, None

    # Optional coarse strided pre-downsample before the (expensive) trilinear
    # resize: take every other voxel per axis so interpolate runs on ~1/8 the
    # data. Only when fast=True; full resolution is the default.
    if fast:
        step = 2 if min(raw.shape) >= 2 * min(out_shape) else 1
        if step > 1:
            raw = raw[::step, ::step, ::step]

    # Resize the (already-coarse) array to out_shape. torch converts to float32
    # inside interpolate; the result is tiny (64^3).
    t = torch.from_numpy(np.ascontiguousarray(raw)).to(torch.float32)[None, None]
    del raw
    t = torch.nn.functional.interpolate(
        t, size=tuple(out_shape), mode="trilinear", align_corners=False)
    vol = t[0, 0].numpy().copy()                        # small: out_shape
    del t

    # Apply the scale/offset now, on the small volume.
    if sl not in (None, 1.0):
        vol *= np.float32(sl)
    if inter not in (None, 0.0):
        vol += np.float32(inter)

    # Clip bright outliers, then scale to [0, 1] over the (small) volume.
    hi = float(np.percentile(vol, 99.5))
    if hi > 0:
        np.clip(vol, 0.0, hi, out=vol)
        vol /= hi

    brain = vol[vol > 0.01]                              # per-scan z-score
    if brain.size:
        vol = (vol - brain.mean()) / (brain.std() + 1e-6)
    return np.ascontiguousarray(vol, dtype=np.float32)


class RawBIDSDataset(torch.utils.data.Dataset):
    """Volumes straight from a BIDS tree.

    target: "age" (regression) or "sex" (binary classification).
    """

    def __init__(self, root, target="age", shape=(64, 64, 64),
                 indices=None, train=False, cache=True, fast=False):
        self.root = pathlib.Path(root)
        self.shape, self.target, self.train = tuple(shape), target, train
        self.fast = fast
        self.cache = {} if cache else None

        part = read_participants(self.root)
        items = []
        for sub, f in scan_bids(self.root):
            m = part.get(sub)
            if not m:
                continue
            if target == "age":
                try:
                    label = float(m["age"])
                except (KeyError, ValueError):
                    continue
            elif target == "sex":
                s = m.get("sex", "")
                if s not in ("M", "F"):
                    continue
                label = 1.0 if s == "M" else 0.0
            else:
                raise ValueError(f"unknown target {target!r}")
            items.append((sub, f, label))

        if indices is not None:
            items = [items[i] for i in indices]
        self.items = items
        self.subjects = [s for s, _, _ in items]
        self.labels = np.array([l for _, _, l in items], dtype=np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sub, f, label = self.items[i]
        if self.cache is not None and i in self.cache:
            vol = self.cache[i]
        else:
            vol = to_volume(f, self.shape, fast=self.fast)
            if self.cache is not None:
                self.cache[i] = vol
        x = torch.from_numpy(vol.copy())
        if self.train:
            if torch.rand(1).item() < 0.5:      # brains are near-symmetric
                x = torch.flip(x, dims=[0])
            x = x + 0.02 * torch.randn_like(x)
        return x, torch.tensor(label, dtype=torch.float32)


def subject_split(dataset, val_fraction=0.25, seed=0):
    """Indices split BY SUBJECT, so no subject appears on both sides.

    Splitting by scan would put sub-01's two sessions in train and val and
    report memorisation as accuracy.
    """
    subs = np.array(dataset.subjects)
    uniq = np.unique(subs)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_fraction)))
    val = set(uniq[:n_val].tolist())
    is_val = np.array([s in val for s in subs])
    return np.where(~is_val)[0], np.where(is_val)[0]

