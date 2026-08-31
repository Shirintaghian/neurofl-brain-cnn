"""Flower ServerApp — FedAvg over the sites' 3D CNNs (NeuroFL Message API).

Pairs with client_app_raw.py. Uses the modern flwr.serverapp API
(``@app.main`` + ``strategy.start(grid=...)``) that the NeuroFL platform
runs. When the platform's strategy wrapper is importable, it is applied so the
run reports metrics + per-client contribution scoring; otherwise the plain
FedAvg runs, so this app also works on a stock Flower deployment.

All sites start from identical weights, initialised centrally here.
"""
from __future__ import annotations

import os
import sys

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import BrainCNN

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    # NeuroFL injects `num_rounds` (underscore); `num-server-rounds` is the
    # app-native key. Read whichever is present.
    num_rounds = int(cfg.get("num_rounds", cfg.get("num-server-rounds", 10)))
    target = str(cfg.get("target", "age"))
    min_clients = int(cfg.get("min-clients", 2))
    # Must match the clients' stem_stride, or the initial state_dict shapes
    # won't line up with the model the clients build.
    stem_stride = int(cfg.get("stem_stride", cfg.get("stem-stride", 2)))

    # Identical initial weights for every site: initialise centrally and ship
    # the arrays as the strategy's starting point.
    init = BrainCNN(task=target, age_offset=25.0 if target == "age" else 0.0,
                    stem_stride=stem_stride)
    initial = [v.cpu().numpy() for v in init.state_dict().values()]

    strategy = FedAvg(
        fraction_train=1.0,          # small federations: use every site
        fraction_evaluate=1.0,
        min_train_nodes=min_clients,
        min_evaluate_nodes=min_clients,
        min_available_nodes=min_clients,
    )

    # Apply the NeuroFL platform's metrics + contribution-scoring callbacks when
    # available; harmless no-op fallback keeps this runnable on stock Flower.
    try:
        from server.neurofl_strategy_wrapper import wrap_strategy_for_neurofl
        strategy = wrap_strategy_for_neurofl(strategy, context)
    except Exception:
        pass

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(initial),
        num_rounds=num_rounds,
    )

    print(
        f"[server] finished. final tensors="
        f"{len(result.arrays.to_numpy_ndarrays())}",
        flush=True,
    )
