#!/usr/bin/env python3
"""Measure TensorFlow FedAvg utility with native Flower central DP on public smoke data."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import tensorflow as tf
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, FedAvg
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score

from smoke import FEATURES, LABEL, save_json, sha256, source_data, split_data
from flower.frameworks.tensorflow_adapter import TensorFlowAdapter
from flower.strategies import build_strategy
from flower.local_experiment import local_task_identity

# Fixed unit conversion, independent of all participant samples and labels.
SCALES = np.array([100, 100, 10000, 10000, 100000, 100000, 1000, 100, 100, 100, 100, 100], dtype=np.float32)


def features(frame):
    return np.clip(frame[FEATURES].to_numpy(dtype=np.float32) / SCALES, -10, 10)


def model():
    result = tf.keras.Sequential([tf.keras.Input((len(FEATURES),)),
                                  tf.keras.layers.Dense(8, activation="relu"),
                                  tf.keras.layers.Dense(1, activation="sigmoid")])
    result.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.05), loss="binary_crossentropy")
    return result


def evaluate(model, frame, prefix):
    probability = model(features(frame), training=False).numpy().ravel()
    y = frame[LABEL].to_numpy()
    predicted = probability >= 0.5
    return {f"{prefix}_{key}": float(value) for key, value in {
        "logloss": log_loss(y, probability, labels=[0, 1]),
        "f1": f1_score(y, predicted, zero_division=0),
        "accuracy": accuracy_score(y, predicted),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "roc_auc": roc_auc_score(y, probability),
    }.items()}


class LocalGrid:
    def get_node_ids(self):
        return [1, 2, 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/aide-wp4-smoke-data"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    source, provenance = source_data(args.cache)
    prepared = split_data(source, 3, args.seed, "iid")
    prepared[["row_id", "UTC", LABEL, "partition", "client"]].to_csv(args.output / f"partitions-seed-{args.seed}.csv", index=False)
    save_json(args.output / "provenance.json", provenance)
    save_json(args.output / "experiment.json", {
        "scope": "local public-data diagnostic; central trusted-server client-level DP model releases",
        "evaluation": "public benchmark validation/test metrics are outside private client release accounting",
        "preprocessing": "fixed unit scales and [-10,10] clipping; no learned global statistics",
        "scales": SCALES.tolist(), "versions": {name: importlib.metadata.version(name) for name in ["flwr", "tensorflow", "numpy"]},
    })
    source_dir = args.output / "source"
    source_dir.mkdir()
    for path in [Path(__file__), Path(__file__).with_name("smoke.py"),
                 Path(__file__).parents[2] / "flower/flower/local_experiment.py",
                 Path(__file__).parents[2] / "flower/flower/privacy.py",
                 Path(__file__).parents[2] / "flower/flower/strategies.py",
                 Path(__file__).parents[2] / "flower/flower/frameworks/tensorflow_adapter.py"]:
        shutil.copy2(path, source_dir / path.name)
    validation = prepared[prepared.partition == "validation"]
    test = prepared[prepared.partition == "test"]
    adapter = TensorFlowAdapter()
    for noise in [None, 0.0, 0.5, 1.0]:
        accounted = noise is not None and noise > 0
        method = ("tensorflow-fedavg" if noise is None else
                  "tensorflow-clipped-fedavg" if noise == 0 else f"tensorflow-dp-noise-{noise}")
        directory = args.output / f"{method}-seed-{args.seed}"
        directory.mkdir()
        config = {"method": method, "seed": args.seed, "rounds": args.rounds, "clients": 3,
                  "local_epochs": 1, "noise_multiplier": noise, "clipping_norm": 0.2,
                  "target_delta": 1e-5, "model": "12-8-1 dense ReLU/sigmoid SGD0.05",
                  "aggregation": "uniform-client FedAvg", "public_benchmark_evaluation": True}
        save_json(directory / "config.json", config)
        tf.keras.utils.set_random_seed(args.seed)
        global_model = model()
        initial = global_model.get_weights()
        arrays = ArrayRecord(initial)
        save_json(directory / "model-scenario.json", json.loads(global_model.to_json()))
        if noise is None:
            strategy = FedAvg(min_available_nodes=3, fraction_evaluate=0.0)
        elif noise == 0:
            strategy = DifferentialPrivacyServerSideFixedClipping(
                strategy=FedAvg(min_available_nodes=3, fraction_evaluate=0.0),
                noise_multiplier=0.0, clipping_norm=0.2, num_sampled_clients=3)
        else:
            strategy = build_strategy({"strategy": "fedavg", "fraction_fit": 1.0,
                "fraction_evaluate": 1.0, "min_available_clients": 3,
                "privacy": {"mode": "central_dp", "noise_multiplier": noise, "clipping_norm": 0.2,
                            "num_sampled_clients": 3, "population_size": 3, "target_delta": 1e-5}},
                scenario_name=method, model=global_model, adapter=adapter,
                share_dir=str(directory), server_rounds=args.rounds)
        started = time.perf_counter()
        with (directory / "metrics.jsonl").open("w") as log:
            for rnd in range(1, args.rounds + 1):
                requests = list(strategy.configure_train(rnd, arrays, ConfigRecord(), LocalGrid()))
                replies = []
                for request in requests:
                    client = request.metadata.dst_node_id - 1
                    frame = prepared[prepared.client == client]
                    x, y = features(frame), frame[LABEL].to_numpy(dtype=np.float32)
                    local = model()
                    local.set_weights(arrays.to_numpy_ndarrays())
                    indices = np.random.default_rng(args.seed + 1000 * rnd + client).permutation(len(y))
                    for start in range(0, len(indices), 256):
                        batch = indices[start:start + 256]
                        local.train_on_batch(x[batch], y[batch])
                    replies.append(request.create_reply(RecordDict({"arrays": ArrayRecord(local.get_weights()),
                                                        "metrics": MetricRecord({"num-examples": 1})})))
                if accounted:
                    # Keras initialization seeds NumPy globally. Re-seed from OS
                    # entropy before native Flower Gaussian noise, never the run seed.
                    np.random.seed(None)
                arrays, _ = strategy.aggregate_train(rnd, replies)
                global_model.set_weights(arrays.to_numpy_ndarrays())
                row = {"round": rnd, **evaluate(global_model, validation, "validation")}
                if accounted:
                    spent = strategy.privacy_spent()
                    row.update(privacy_epsilon=spent.epsilon, privacy_delta=spent.delta)
                log.write(json.dumps(row) + "\n")
                log.flush()
        summary = {**evaluate(global_model, test, "test"), "selected_round": args.rounds,
                   "elapsed_seconds": time.perf_counter() - started,
                   "scope": "public benchmark utility; trusted central server, uniform client weights, no sampling amplification"}
        y_test = test[LABEL].to_numpy()
        majority = int(prepared.loc[prepared.partition == "train", LABEL].mode().iloc[0])
        dummy = np.full(len(y_test), majority)
        predictions = global_model(features(test), training=False).numpy().ravel() >= 0.5
        summary.update(
            test_num_examples=len(y_test), validation_num_examples=len(validation),
            train_num_examples=int((prepared.partition == "train").sum()),
            client_train_counts={str(i): int((prepared.client == i).sum()) for i in range(3)},
            majority_baseline_label=majority,
            majority_baseline_test_accuracy=float(accuracy_score(y_test, dummy)),
            majority_baseline_test_balanced_accuracy=float(balanced_accuracy_score(y_test, dummy)),
            majority_baseline_test_f1=float(f1_score(y_test, dummy, zero_division=0)),
            test_confusion_matrix=confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
            test_confusion_matrix_labels=[0, 1],
        )
        if accounted:
            spent = strategy.privacy_spent()
            summary.update(privacy_epsilon=spent.epsilon, privacy_delta=spent.delta, private_model_releases=spent.num_releases)
        global_model.save(directory / "model.keras")
        restored = tf.keras.models.load_model(directory / "model.keras")
        np.testing.assert_allclose(restored(features(test.iloc[:100])).numpy(), global_model(features(test.iloc[:100])).numpy())
        save_json(directory / "summary.json", summary)
        save_json(directory / "checksums.json", {p.name: sha256(p) for p in directory.iterdir() if p.is_file()})
        print(json.dumps({"run": method, **summary}), flush=True)
    save_json(args.output / "manifest.json", {str(p.relative_to(args.output)): sha256(p)
              for p in sorted(args.output.rglob("*")) if p.is_file()})


if __name__ == "__main__":
    with local_task_identity():
        main()
