"""Flower ClientApp: federated 3D CNN over each site's own prepared volumes.

Reads the derived dataset produced by prepare_raw.py — images.npy/targets.npy —
NOT the raw BIDS tree and not the old flattened train_data.npy.

The dataset directory comes from the SuperNode's node-config `data-path`, so a
site trains only on the dataset its own SuperNode was given.
"""
from __future__ import annotations
import json, os, pathlib, sys
import numpy as np, torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (BrainAgeCNN, VolumeDataset, load_split,
                   train_one_epoch, evaluate, baseline_mae)

def _params(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]

def _set_params(model, params):
    sd = model.state_dict()
    model.load_state_dict(
        {k: torch.tensor(v) for k, v in zip(sd.keys(), params)}, strict=True)

class BrainClient(NumPyClient):
    def __init__(self, ctx: Context):
        cfg = ctx.node_config
        data_dir = cfg.get("data-path") or os.getenv("NEUROFL_DATA_PATH", "/data")
        self.site = str(cfg.get("site", os.getenv("VM_NAME", "unknown")))

        d = pathlib.Path(data_dir)
        # A site may be given the dataset root or its parent; accept either.
        if not (d / "images.npy").exists():
            found = sorted(d.glob("*/images.npy"))
            if not found:
                raise FileNotFoundError(
                    f"no images.npy under {d}. Run prepare_raw.py first — this "
                    f"model trains on prepared volumes, not raw BIDS.")
            d = found[0].parent

        (Xtr, ytr), (Xva, yva), self.split = load_split(d, seed=0)
        self.baseline = baseline_mae(ytr, yva)
        self.n_train = len(ytr)

        rc = ctx.run_config
        bs = int(rc.get("batch-size", 4))
        self.epochs = int(rc.get("local-epochs", 2))
        self.lr = float(rc.get("lr", 3e-4))

        self.tr = torch.utils.data.DataLoader(
            VolumeDataset(Xtr, ytr, True), batch_size=bs, shuffle=True)
        self.va = torch.utils.data.DataLoader(
            VolumeDataset(Xva, yva), batch_size=bs)
        self.model = BrainAgeCNN(age_offset=float(np.mean(ytr)))
        print(f"[{self.site}] {json.dumps(self.split)} baseline MAE {self.baseline:.2f}y",
              flush=True)

    def fit(self, parameters, config):
        _set_params(self.model, parameters)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-2)
        for _ in range(self.epochs):
            loss = train_one_epoch(self.model, self.tr, opt)
        m = evaluate(self.model, self.va)
        # One tagged line per round; the watcher harvests these into the
        # Run History tab.
        print("NEUROFL_TRAINING_LOG " + json.dumps({
            "client_id": self.site, "round": int(config.get("server_round", 0)),
            "num_examples": self.n_train,
            "metrics": {"train_loss": float(loss), "val_mae": m["mae"],
                        "val_rmse": m["rmse"], "baseline_mae": self.baseline,
                        "beats_baseline": bool(m["mae"] < self.baseline)},
        }), flush=True)
        return _params(self.model), self.n_train, {"train_loss": float(loss)}

    def evaluate(self, parameters, config):
        _set_params(self.model, parameters)
        m = evaluate(self.model, self.va)
        return float(m["mae"]), m["n"], {
            "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"],
            "baseline_mae": self.baseline,
            "beats_baseline": bool(m["mae"] < self.baseline),
        }

app = ClientApp(client_fn=lambda ctx: BrainClient(ctx).to_client())
