"""Flower ClientApp: 3D CNN trained directly on each site's RAW BIDS tree
(NeuroFL Message API).

No preprocessing step and no derived dataset — the SuperNode points at the raw
dataset and volumes are built in the DataLoader at read time.

Dataset location comes from the SuperNode's node-config ``data-path``, so a
site trains only on the dataset its own SuperNode was given.

run_config keys (all optional):
    target        "age" (default) | "sex"
    local-epochs  default 2
    batch-size    default 4
    learning-rate default 3e-4        (NeuroFL injects this key; ``lr`` also read)
    shape         default "64,64,64"
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import RawBIDSDataset, subject_split
from shared import BrainCNN, make_loss, train_one_epoch, evaluate

app = ClientApp()

_TRAINING_LOG_TAG = "NEUROFL_TRAINING_LOG"


# ── helpers ──────────────────────────────────────────────────────────────────

def _config(context: Context) -> dict:
    """Merge per-dataset node_config OVER the app-wide run_config.

    The SuperNode sets node_config per dataset (data-path, site), so a
    multi-dataset node's clients each read their OWN data-path.
    """
    cfg = dict(context.run_config)
    cfg.update(dict(getattr(context, "node_config", {}) or {}))
    return cfg


def _client_id(context: Context, cfg: dict) -> str:
    return (
        os.getenv("CLIENT_ID")
        or str(cfg.get("site") or "")
        or str(context.node_id)
    )


def _round_from(msg: Message) -> int:
    try:
        return int(msg.content["config"]["server-round"])
    except (KeyError, TypeError, ValueError):
        pass
    try:
        return int(getattr(msg.metadata, "group_id", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _set_params(model, params):
    sd = model.state_dict()
    model.load_state_dict(
        {k: torch.tensor(np.asarray(v)) for k, v in zip(sd.keys(), params)},
        strict=True,
    )


def _get_params(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]


def _find_bids(start: pathlib.Path) -> pathlib.Path:
    """Accept either the dataset dir or a parent holding exactly one."""
    if list(start.glob("sub-*")):
        return start
    subdirs = [d for d in sorted(start.iterdir())
               if d.is_dir() and list(d.glob("sub-*"))]
    if not subdirs:
        raise FileNotFoundError(
            f"no BIDS dataset (sub-*) under {start}. This model reads raw BIDS; "
            f"if the site serves prepared .npy arrays, use client_app.py instead."
        )
    return subdirs[0]


def _emit_training_log(client_id, server_round, num_examples, metrics):
    record = {
        "client_id": client_id,
        "round": int(server_round),
        "num_examples": int(num_examples),
        "metrics": {
            k: (float(v) if isinstance(v, (int, float, bool)) else v)
            for k, v in (metrics or {}).items()
        },
        "timestamp": time.time(),
    }
    print(f"{_TRAINING_LOG_TAG} {json.dumps(record)}", flush=True)


def _build(context: Context):
    """Load the site's raw BIDS data and build model + loaders once per call.

    Returns (model, train_loader, val_loader, ytr, yva, target, epochs, lr, site).
    """
    cfg = _config(context)
    site = _client_id(context, cfg)

    root = _find_bids(pathlib.Path(
        cfg.get("data-path") or os.getenv("NEUROFL_DATA_PATH", "/data")))

    target = str(cfg.get("target", "age"))
    shape = tuple(int(x) for x in str(cfg.get("shape", "64,64,64")).split(","))
    # NeuroFL injects underscore keys (local_epochs, batch_size, learning_rate);
    # keep the hyphen/`lr` forms as fallbacks for stock Flower.
    epochs = int(cfg.get("local_epochs", cfg.get("local-epochs", 2)))
    bs = int(cfg.get("batch_size", cfg.get("batch-size", 4)))
    lr = float(cfg.get("learning_rate", cfg.get("learning-rate", cfg.get("lr", 3e-4))))
    fast = str(cfg.get("fast_load", cfg.get("fast-load", "0"))).lower() in ("1", "true", "yes")

    full = RawBIDSDataset(root, target=target, shape=shape, fast=fast)
    if len(full) == 0:
        raise RuntimeError(f"no usable scans in {root} for target {target!r}")
    tr_idx, va_idx = subject_split(full, seed=0)

    tr_ds = RawBIDSDataset(root, target, shape, indices=tr_idx, train=True, fast=fast)
    va_ds = RawBIDSDataset(root, target, shape, indices=va_idx, fast=fast)
    ytr, yva = tr_ds.labels, va_ds.labels

    tr = torch.utils.data.DataLoader(tr_ds, batch_size=bs, shuffle=True)
    va = torch.utils.data.DataLoader(va_ds, batch_size=bs)

    stem_stride = int(cfg.get("stem_stride", cfg.get("stem-stride", 2)))
    model = BrainCNN(task=target,
                     age_offset=float(ytr.mean()) if target == "age" else 0.0,
                     stem_stride=stem_stride)

    print(f"[{site}] raw BIDS {root} | {len(full)} scans "
          f"({len(set(full.subjects))} subj) -> train {len(tr_ds)}/val {len(va_ds)} "
          f"| target={target} shape={shape} stem_stride={stem_stride} "
          f"bs={bs} epochs={epochs}", flush=True)
    return model, tr, va, ytr, yva, target, epochs, lr, site


def _metrics(model, va, ytr, yva, target):
    if target == "age":
        e = evaluate(model, va)
        base = float(np.abs(yva - ytr.mean()).mean())
        return {
            "val_mae": e.get("mae", float("nan")),
            "val_rmse": e.get("rmse", float("nan")),
            "baseline_mae": base,
            "beats_baseline": bool(e.get("mae", 1e9) < base),
        }
    e = evaluate(model, va)
    base = float(max((yva == 0).mean(), (yva == 1).mean()))
    return {
        "val_acc": e.get("acc", float("nan")),
        "val_balanced_acc": e.get("balanced_acc", float("nan")),
        "val_auc": e.get("auc", float("nan")),
        "baseline_acc": base,
        "beats_baseline": bool(e.get("acc", 0.0) > base),
    }


# ── Message API handlers ─────────────────────────────────────────────────────

@app.train()
def train(msg: Message, context: Context) -> Message:
    _cfg = _config(context)
    print(f"[cnn.train] config: shape={_cfg.get('shape')} "
          f"stem_stride={_cfg.get('stem_stride', _cfg.get('stem-stride'))} "
          f"batch_size={_cfg.get('batch_size')} target={_cfg.get('target')}",
          flush=True)
    arrays_in = msg.content["arrays"].to_numpy_ndarrays()
    model, tr, va, ytr, yva, target, epochs, lr, site = _build(context)

    _set_params(model, arrays_in)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = make_loss(target, ytr)
    loss = 0.0
    for _ in range(epochs):
        loss = train_one_epoch(model, tr, opt, loss_fn=loss_fn)

    num_examples = len(ytr)
    m = _metrics(model, va, ytr, yva, target)
    metrics = {"train_loss": float(loss), **m}

    _emit_training_log(site, _round_from(msg), num_examples, metrics)

    # MetricRecord accepts numbers only; drop the bool 'beats_baseline'.
    numeric = {k: float(v) for k, v in metrics.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(_get_params(model)),
            "metrics": MetricRecord({**numeric, "num-examples": float(num_examples)}),
        }),
        reply_to=msg,
    )


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    arrays_in = msg.content["arrays"].to_numpy_ndarrays()
    model, tr, va, ytr, yva, target, epochs, lr, site = _build(context)

    _set_params(model, arrays_in)
    m = _metrics(model, va, ytr, yva, target)
    # Primary loss: MAE for age, error-rate for sex.
    loss = float(m["val_mae"]) if target == "age" else float(1.0 - m["val_acc"])
    num_examples = len(yva)

    numeric = {k: float(v) for k, v in m.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return Message(
        content=RecordDict({
            "metrics": MetricRecord(
                {**numeric, "loss": loss, "num-examples": float(num_examples)}),
        }),
        reply_to=msg,
    )
