#!/usr/bin/env python3
"""Run PyTorch FedAvg on chronological smoke-sensor partitions and retain evidence."""
import argparse
import configparser
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from flwr.app import ConfigRecord, RecordDict

from smoke import ROOT, FEATURES, LABEL, source_data, split_data, save_json, sha256
from flower.frameworks import get_adapter
from flower.frameworks.base import ClientDatasets, HoldOutDataset
from flower.local_experiment import local_task_identity
from flower.strategies import build_strategy


def scores(y, probability):
    prediction = probability >= .5
    return {"accuracy": float(accuracy_score(y, prediction)), "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "f1": float(f1_score(y, prediction)), "roc_auc": float(roc_auc_score(y, probability)),
            "logloss": float(log_loss(y, probability)), "confusion_matrix": confusion_matrix(y, prediction).tolist()}


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    os.environ["WANDB_MODE"] = "disabled"
    frame, provenance = source_data(args.cache)
    rows = split_data(frame, args.clients, args.seed, "iid")
    # Reserve the latter test period for the server, with another temporal embargo.
    test = rows[rows.partition == "test"]
    boundary = int(test.iloc[len(test) // 2].UTC)
    rows.loc[(rows.partition == "test") & (rows.UTC >= boundary), "partition"] = "holdout"
    rows.loc[(rows.partition == "test") & (rows.UTC >= boundary - 60), "partition"] = "embargo"
    for partition in ("validation", "test"):
        indices = rows.index[rows.partition == partition]
        for client, subset in enumerate(np.array_split(indices, args.clients)):
            rows.loc[subset, "client"] = client
    for earlier, later in zip(("train", "validation", "test"), ("validation", "test", "holdout")):
        assert rows.loc[rows.partition == later, "UTC"].min() - rows.loc[rows.partition == earlier, "UTC"].max() >= 60
    assert rows.row_id.is_unique
    assert not rows.duplicated(FEATURES).any()
    partitions = rows[["row_id", "UTC", LABEL, "partition", "client"]]
    partitions.to_csv(output / "partitions.csv", index=False)
    partitions.groupby(["partition", "client", LABEL]).size().to_csv(output / "partition-counts.csv")
    scaler = StandardScaler().fit(rows.loc[rows.partition == "train", FEATURES])
    scenario = json.loads((ROOT / "examples/smoke-pytorch.json").read_text())
    scenario["training"]["server_rounds"] = args.rounds
    scenario["aggregation"]["min_available_clients"] = args.clients
    scenario["pytorch"]["model"].update(seed=args.seed, feature_mean=scaler.mean_.tolist(), feature_scale=scaler.scale_.tolist())
    save_json(output / "scenario.json", scenario)
    adapter = get_adapter("pytorch")
    model = adapter.build_model_from_scenario(scenario)
    config = adapter.make_config_record(scenario)
    arrays = adapter.initial_arrays(model)
    datasets = []
    for client in range(args.clients):
        share = output / f"client-{client}" / "share"
        share.mkdir(parents=True)
        values = []
        for partition, filename in (("train", "train"), ("validation", "eval"), ("test", "test")):
            subset = rows[(rows.partition == partition) & (rows.client == client)]
            if subset.empty:
                raise ValueError("empty participant partition")
            subset[FEATURES + [LABEL]].to_parquet(share / f"{filename}.parquet", index=False)
            values += [adapter.adapt_features(subset[FEATURES].to_numpy(), model.aide_config), subset[LABEL].to_numpy()]
        datasets.append(ClientDatasets(*values))
    heldout = rows[rows.partition == "holdout"]
    if set(heldout[LABEL]) != {0, 1}:
        raise ValueError("server holdout must contain both labels")
    server = output / "server" / "share"
    server.mkdir(parents=True)
    heldout[FEATURES + [LABEL]].to_parquet(server / "hold-out.parquet", index=False)
    holdout = HoldOutDataset(adapter.adapt_features(heldout[FEATURES].to_numpy(), model.aide_config), heldout[LABEL].to_numpy())
    ini = configparser.ConfigParser()
    ini["serverapp"] = {"scenario": json.dumps(scenario)}
    with (server / "config.ini").open("w") as file:
        ini.write(file)
    class Grid:
        def get_node_ids(self):
            return list(range(1, args.clients + 1))
    strategy = build_strategy(scenario["aggregation"], scenario_name=scenario["name"], model=model,
                              adapter=adapter, share_dir=str(server), server_rounds=args.rounds)
    initial = dict(adapter.evaluate_centralised(model, arrays, holdout))
    metrics = []
    with local_task_identity():
        for round_nr in range(1, args.rounds + 1):
            requests = list(strategy.configure_train(round_nr, arrays, config, Grid()))
            replies, local_results = [], []
            for request in requests:
                data = datasets[request.metadata.dst_node_id - 1]
                local = adapter.restore_client_model(config)
                result = adapter.train(local, arrays, data, request.content["config"], 0)
                local_results.append((result.arrays, len(data.x_train)))
                replies.append(request.create_reply(RecordDict({"arrays": result.arrays, "metrics": result.metrics,
                                        "history": ConfigRecord({"history": json.dumps(result.history)})})))
            arrays, train_metrics = strategy.aggregate_train(round_nr, replies)
            # Independently check sample weighting, including every named tensor.
            for key in arrays:
                expected = sum(a[key].numpy() * (n / sum(count for _, count in local_results)) for a, n in local_results)
                np.testing.assert_allclose(arrays[key].numpy(), expected, rtol=1e-5, atol=1e-7)
            replies = []
            for request in strategy.configure_evaluate(round_nr, arrays, config, Grid()):
                data = datasets[request.metadata.dst_node_id - 1]
                result = adapter.evaluate(adapter.restore_client_model(config), arrays, data, config)
                replies.append(request.create_reply(RecordDict({"metrics": result})))
            client_metrics = strategy.aggregate_evaluate(round_nr, replies)
            central = adapter.evaluate_centralised(model, arrays, holdout)
            metrics.append({"round": round_nr, "train": dict(train_metrics), "client_test": dict(client_metrics), "server_holdout": dict(central)})
    import wandb
    wandb.finish()
    probability = adapter.predict_proba(model, heldout[FEATURES].to_numpy())[:, 1]
    restored = adapter.load_model(server / scenario["name"])
    np.testing.assert_array_equal(probability, adapter.predict_proba(restored, heldout[FEATURES].to_numpy())[:, 1])
    predictions = heldout[["row_id", "UTC", LABEL]].copy()
    predictions["probability"] = probability
    predictions.to_csv(output / "predictions.csv", index=False)
    baseline = np.full(len(heldout), float(rows.loc[rows.partition == "train", LABEL].mean() >= .5))
    summary = {"initial_holdout": initial, "final_holdout": scores(heldout[LABEL], probability),
               "majority_baseline": scores(heldout[LABEL], baseline), "reload_predictions_identical": True,
               "weighted_aggregation_verified": True, "clients": args.clients, "rounds": args.rounds,
               "scope": "Local multi-client experiment; not cluster deployment or privacy verification"}
    save_json(output / "summary.json", summary)
    (output / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
    save_json(output / "provenance.json", provenance)
    manifest = {"executed_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
                "python": sys.version, "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
                "features": FEATURES, "label": LABEL, "normalization": "training rows only; shared public moments",
                "split": "chronological 50/20/30; 60-second embargo; last half of test reserved for server with another embargo",
                "limitations": "Simulated participants from one sensor dataset; no independent sites; no DP, SecAgg+ or FHE"}
    save_json(output / "manifest.json", manifest)
    source = output / "source"
    source.mkdir()
    for path in [Path(__file__), ROOT / "experiments/wp4/smoke.py", ROOT / "flower/flower/frameworks/pytorch_adapter.py",
                 ROOT / "flower/flower/strategies.py", ROOT / "flower/flower/client_app.py", ROOT / "flower/flower/server_app.py"]:
        shutil.copy2(path, source / path.name)
    (source / "changes.diff").write_bytes(subprocess.check_output(["git", "diff"], cwd=ROOT))
    save_json(output / "checksums.json", {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/aide-wp4-smoke-data"))
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
