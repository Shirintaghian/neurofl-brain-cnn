# NeuroFL 3D CNN — raw BIDS brain imaging

A federated 3D CNN for brain **age regression** or **sex classification**,
trained directly on raw BIDS T1w volumes. Ported to run on the **NeuroFL
platform** (the modern `flwr.serverapp` Message API) and verified end-to-end on
the open-data clients (`nodeap`, `auditory_fmri`).

Each site points its SuperNode at its raw dataset; volumes are built in the
DataLoader at read time — **no preprocessing step and no derived dataset**.

---

## Files

Four files, mirroring a standard NeuroFL app (`shared` + client + server +
config). `shared.py` plays the role a tabular app's `shared.py` does — it just
also carries the 3D CNN and the raw-BIDS volume loader, which are bigger than a
CSV model, so they live here rather than in the ClientApp.

| File | Role |
|---|---|
| `shared.py` | **The model + data + train/eval** — `BrainCNN` (3D residual CNN, GroupNorm, ~905k params), `RawBIDSDataset` (reads a raw BIDS tree; resize → clip → z-score per scan), and the `train_one_epoch` / `evaluate` / `make_loss` helpers. |
| `client_app.py` | ClientApp — `@app.train` / `@app.evaluate`; reads the site's raw BIDS tree and calls into `shared`. |
| `server_app.py` | ServerApp — FedAvg over the sites (`@app.main` / `strategy.start`). |
| `pyproject.toml` | Flower app config (target, shape, rounds, hyperparameters). |

To replicate: point each site's node at its raw dataset, set the knobs in
`pyproject.toml` (see **Config**), build the app into a FAB, and submit it
targeting your clients. Nothing else is needed.

---

## Run on NeuroFL

1. Build the app into a FAB and submit it, targeting your clients. On the
   modeller UI, upload this app and select the client datasets to train on.
2. Rounds, target, resolution and speed are controlled in
   `pyproject.toml` → `[tool.flwr.app.config]` (see **Config** below) — the
   platform passes `learning_rate` / `batch_size` / `local_epochs` from the
   submission, but reserves `num_rounds` / `target_clients`, so **rounds are set
   by `num-server-rounds` here.**

### Run locally (sanity check, no federation)

Everything you need is in `shared.py`, so you can smoke-test the model on one
dataset before submitting — the same idea as a tabular `shared.py` benchmark:

```python
import torch
from shared import (RawBIDSDataset, subject_split, BrainCNN,
                    make_loss, train_one_epoch, evaluate)

root, target = "/data/nodeap", "sex"          # or "age"
full = RawBIDSDataset(root, target=target, shape=(64, 64, 64))
tr_idx, va_idx = subject_split(full, seed=0)
tr = torch.utils.data.DataLoader(
    RawBIDSDataset(root, target, (64, 64, 64), indices=tr_idx, train=True),
    batch_size=4, shuffle=True)
va = torch.utils.data.DataLoader(
    RawBIDSDataset(root, target, (64, 64, 64), indices=va_idx), batch_size=4)

model = BrainCNN(task=target)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = make_loss(target, full.labels[tr_idx])
for epoch in range(40):
    train_one_epoch(model, tr, opt, loss_fn=loss_fn)
print(evaluate(model, va))
```

---

## Config (`[tool.flwr.app.config]`)

| Key | Default | Meaning |
|---|---|---|
| `num-server-rounds` | `10` | federated rounds (set here — the platform reserves `num_rounds`) |
| `target` | `"age"` | `"age"` (regression) or `"sex"` (binary classification) |
| `shape` | `"64,64,64"` | volume the CNN sees; smaller = faster, less detail |
| `local_epochs` | `2` | local epochs per round (overridable at submit time) |
| `batch_size` | `4` | (overridable at submit time) |
| `learning_rate` | `3e-4` | (overridable at submit time) |
| `min-clients` | `2` | minimum sites for a round |
| `fast_load` | `"0"` | **volume-loading speed/quality trade-off — see below** |

### `fast_load` — the two loading options

Each scan is decompressed and resampled to `shape` at read time. Two modes:

- **`fast_load = "0"` (default) — full resolution.** A full-resolution trilinear
  resample of every scan. Best fidelity. Slower per round, especially on large
  cohorts and CPU-only nodes. **Use this for real results.**
- **`fast_load = "1"` — fast.** A coarse strided pre-downsample (every other
  voxel) before the resize. Several times faster and lighter on memory, at a
  negligible quality cost for these target shapes. Handy for a quick smoke test
  or memory-tight nodes.

Both modes keep peak RAM low (they never hold a full-resolution `float64`
volume), which matters inside the training sandbox.

---

## Measured results (federated, this platform)

A 10-round FedAvg run over the two open-data imaging clients (`nodeap` +
`auditory_fmri`), target **sex**, full resolution (`shape=64`, `stem-stride=1`,
`batch_size=4`), both sites reporting every round, zero failures:

| Round | val_acc | val_balanced_acc | val_auc | baseline_acc |
|------:|--------:|-----------------:|--------:|-------------:|
| 1  | 0.61 | 0.50 | 0.98 | 0.61 |
| 4  | 0.61 | 0.50 | 0.85 | 0.61 |
| 7  | 0.61 | 0.50 | 0.96 | 0.61 |
| 10 | **0.72** | **0.64** | **0.92** | 0.61 |

**Read this honestly:** AUC is high throughout (~0.85–0.98) — the model ranks the
two classes well from the raw scans from round 1. Threshold accuracy lags early
(the head bias takes a few rounds to move on a small cohort), then by the final
round the model **beats the majority baseline on both accuracy (0.72 > 0.61) and
balanced accuracy (0.64 > 0.50)**. On a validation fold of ~18 subjects, expect
the accuracy numbers to wobble round-to-round; AUC is the more stable signal here.

For reference, the same architecture scored ~AUC 0.88 on nodeap sex and AUC 1.00
on ds000228 child-vs-adult in separate local runs.

---

## Which datasets to try it on

This is an **imaging** model: it needs a raw BIDS tree with T1w scans
(`sub-*/anat/*T1w.nii.gz`) and a `participants.tsv` carrying the label column
(`age` for regression, `sex`/`gender` for classification). It does **not** apply
to tabular / FreeSurfer-IDP datasets — those are a different pipeline.

Good candidates on the open-data clients:

| Dataset | Subjects | Best target | Why |
|---|---|---|---|
| `nodeap` | ~48 (48 T1w) | **sex** | Enough subjects for a stable split; sex is a learnable, balanced-ish signal (measured AUC ~0.9). Age range may be narrow, so age regression is weaker here. |
| `auditory_fmri` | ~14 (17 T1w) | pairs with nodeap | Small on its own; most useful as the **second site** in a 2-node run, which is exactly how it was validated. |
| `ds000228` | ~150+ | **age** (child vs adult) | Wide age range with a strong structural signal — the model hit AUC ~1.0 on child-vs-adult here. The clearest "it works" demo. |

**What to expect, and why.** With small per-site cohorts, **AUC is the reliable
metric** — it reads the model's ranking and is stable. Threshold **accuracy** on a
tiny validation fold (e.g. ~18 subjects) wobbles round-to-round and can sit at the
majority baseline for the first few rounds before the head calibrates; that's a
small-sample effect, not a failure. Judge a run by AUC and by whether accuracy
beats `baseline_acc` by the final rounds. A dataset with a wide, well-sampled
label (age on ds000228) gives the cleanest results; a narrow or tiny one
(age on nodeap, anything on auditory_fmri alone) will look noisier.

---

## Design notes

- **GroupNorm, not BatchNorm.** BatchNorm statistics get averaged across sites
  during FedAvg, which leaks cohort information and destabilises training when
  each site holds tens of scans. GroupNorm is per-sample and carries no state.
- **Subject-wise splits.** Some subjects have several sessions. Splitting by
  scan puts the same brain on both sides and reports memorisation as accuracy.
- **Huber loss for age; class-weighted BCE for sex.** At these cohort sizes a
  single outlier dominates an MSE gradient, and both cohorts skew female so an
  unweighted BCE just predicts the majority.
- **Every metric ships with its baseline.** `baseline_mae` (age) /
  `baseline_acc` (sex) is what "learned nothing" scores; `beats_baseline` says
  whether the model learned anything from the images. A model that does not beat
  baseline has not learned, whatever the loss curve looks like.

---

## Data requirements

A BIDS tree with `participants.tsv` (needs `age`, and `sex`/`gender` for the
classification target) and `sub-*/anat/*T1w.nii.gz`, flat or session-organised.

If the dataset is DataLad/git-annex, content must be fetched first
(`datalad get sub-*/anat/*T1w.nii.gz`) — unfetched files are dangling symlinks
that glob fine and fail on open.
