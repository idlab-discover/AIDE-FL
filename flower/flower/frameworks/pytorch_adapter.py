"""CPU binary classifiers with named, trainable-only Flower parameters."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord
from safetensors.torch import load_file, save_file

from flower.frameworks.base import FrameworkAdapter, TrainResult


def validate_model_config(config):
    allowed = {"kind", "n_features", "hidden_sizes", "seed", "feature_mean", "feature_scale"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("invalid pytorch model configuration")
    if config.get("kind") not in ("logistic_regression", "mlp"):
        raise ValueError("pytorch supports logistic_regression and mlp binary classifiers")
    n = config.get("n_features")
    if type(n) is not int or n < 1:
        raise ValueError("pytorch n_features must be a positive integer")
    seed = config.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("pytorch seed must be an integer in [0, 2**32)")
    hidden = config.get("hidden_sizes", [])
    if not isinstance(hidden, list) or any(type(v) is not int or v < 1 for v in hidden):
        raise ValueError("hidden_sizes must contain positive integers")
    if (config["kind"] == "mlp") != bool(hidden):
        raise ValueError("mlp requires hidden_sizes; logistic_regression forbids them")
    mean, scale = config.get("feature_mean"), config.get("feature_scale")
    if mean is not None or scale is not None:
        if mean is None or scale is None:
            raise ValueError("feature_mean and feature_scale must be supplied together")
        if np.shape(mean) != (n,) or np.shape(scale) != (n,):
            raise ValueError("normalization must match n_features")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(np.asarray(scale) <= 0):
            raise ValueError("normalization must be finite with positive scales")
    return config


class PyTorchAdapter(FrameworkAdapter):
    name = "pytorch"

    def model_config_from_scenario(self, scenario):
        return validate_model_config(scenario.get("pytorch", {}).get("model"))

    def build_model_from_scenario(self, scenario):
        return self.build_model(self.model_config_from_scenario(scenario))

    def build_model(self, config):
        validate_model_config(config)
        widths = [config["n_features"], *config.get("hidden_sizes", []), 1]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.get("seed", 42))
            layers = []
            for index, (left, right) in enumerate(zip(widths, widths[1:])):
                layers.append(nn.Linear(left, right))
                if index < len(widths) - 2:
                    layers.append(nn.ReLU())
            model = nn.Sequential(*layers)
        model.aide_config = dict(config)
        return model

    def _parameters(self, model):
        # No implicit FedBN or counter policy: reject all buffers, including floats.
        if list(model.named_buffers()):
            raise ValueError("pytorch FedAvg does not support state_dict buffers")
        parameters = dict(model.named_parameters())
        if not parameters or list(model.state_dict()) != list(parameters):
            raise ValueError("pytorch state must consist only of named parameters")
        if any(not p.requires_grad or p.dtype != torch.float32 for p in parameters.values()):
            raise ValueError("pytorch FedAvg requires trainable float32 parameters")
        return dict(sorted(parameters.items()))

    def initial_arrays(self, model):
        return ArrayRecord(torch_state_dict={k: v.detach().cpu().clone() for k, v in self._parameters(model).items()})

    def validate_arrays(self, model, arrays):
        expected = self._parameters(model)
        if list(arrays) != list(expected):
            raise ValueError("pytorch parameter names or ordering differ")
        for name, parameter in expected.items():
            value = arrays[name].numpy()
            if value.shape != tuple(parameter.shape) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"invalid pytorch parameter shape, dtype or value: {name}")

    def _apply_arrays(self, model, arrays):
        self.validate_arrays(model, arrays)
        model.load_state_dict(arrays.to_torch_state_dict(), strict=True)

    def make_config_record(self, scenario):
        from flower.privacy import validate_privacy
        validate_privacy(scenario["aggregation"], self.name)
        if scenario["aggregation"]["strategy"] != "fedavg":
            raise ValueError("pytorch currently supports FedAvg only")
        options = scenario["pytorch"].get("optimizer", {})
        if set(options) - {"kind", "learning_rate", "weight_decay"} or options.get("kind", "adam") not in ("adam", "sgd"):
            raise ValueError("pytorch optimizer must be adam or sgd")
        lr, decay = options.get("learning_rate", .001), options.get("weight_decay", 0.)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in (lr, decay)) or lr <= 0 or decay < 0:
            raise ValueError("invalid pytorch learning_rate or weight_decay")
        training = scenario["training"]
        if any(type(training[k]) is not int or training[k] < 1 for k in ("local_epochs", "batch_size")):
            raise ValueError("local_epochs and batch_size must be positive integers")
        return ConfigRecord({"framework": self.name, "model_config": json.dumps(self.model_config_from_scenario(scenario), sort_keys=True),
                             "label_column": scenario["label_column"], "local_epochs": training["local_epochs"],
                             "batch_size": training["batch_size"], "optimizer": options.get("kind", "adam"),
                             "learning_rate": float(lr), "weight_decay": float(decay)})

    def client_state_record(self, config):
        return ConfigRecord({k: config[k] for k in ("framework", "model_config", "label_column")})

    def state_reload_required(self, state_config, config):
        return state_config is None or dict(state_config) != dict(self.client_state_record(config))

    def get_model_config_from_record(self, config):
        return validate_model_config(json.loads(config["model_config"]))

    def restore_client_model(self, config):
        return self.build_model(self.get_model_config_from_record(config))

    def adapt_features(self, x, model_config):
        validate_model_config(model_config)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != model_config["n_features"] or not np.isfinite(x).all():
            raise ValueError("pytorch requires finite tabular features matching n_features")
        if "feature_mean" in model_config:
            x = (x - np.asarray(model_config["feature_mean"], dtype=np.float32)) / np.asarray(model_config["feature_scale"], dtype=np.float32)
        return x

    def _data(self, x, y):
        x, y = np.array(x, dtype=np.float32, copy=True), np.array(y, dtype=np.float32, copy=True).reshape(-1, 1)
        if len(x) == 0 or len(x) != len(y) or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("pytorch requires nonempty finite data and binary labels 0/1")
        return torch.from_numpy(x), torch.from_numpy(y)

    def _evaluate(self, model, x, y):
        x, y = self._data(x, y)
        model.eval()
        with torch.no_grad():
            logits = model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y).item()
            accuracy = ((logits >= 0) == (y >= .5)).float().mean().item()
        return {"loss": loss, "accuracy": accuracy}

    def train(self, model, arrays, datasets, config, verbose=None):
        self._apply_arrays(model, arrays)
        if config.get("proximal-mu", 0) or config.get("privacy-mode"):
            raise ValueError("pytorch supports non-private FedAvg only")
        x, y = self._data(datasets.x_train, datasets.y_train)
        optimizer_type = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[config["optimizer"]]
        optimizer = optimizer_type(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        generator = torch.Generator().manual_seed(model.aide_config.get("seed", 42) + int(config.get("server-round", 0)))
        history = {"loss": [], "val_loss": []}
        for _ in range(config["local_epochs"]):
            model.train()
            order = torch.randperm(len(x), generator=generator)
            total = 0.
            for indices in order.split(config["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(model(x[indices]), y[indices])
                loss.backward()
                optimizer.step()
                total += loss.item() * len(indices)
            history["loss"].append(total / len(x))
            history["val_loss"].append(self._evaluate(model, datasets.x_eval, datasets.y_eval)["loss"])
        result = self.initial_arrays(model)
        self.validate_arrays(model, result)
        return TrainResult(result, MetricRecord({"num-examples": len(x), "train-loss": history["loss"][-1],
                                                "train-val-loss": history["val_loss"][-1]}), history)

    def evaluate(self, model, arrays, datasets, config):
        self._apply_arrays(model, arrays)
        return MetricRecord({"num-examples": len(datasets.x_test), **self._evaluate(model, datasets.x_test, datasets.y_test)})

    def evaluate_centralised(self, model, arrays, dataset):
        self._apply_arrays(model, arrays)
        return MetricRecord(self._evaluate(model, dataset.x_hold_out, dataset.y_hold_out))

    def save_server_artifacts(self, model, arrays, share_dir, scenario_name, server_round, server_rounds):
        self._apply_arrays(model, arrays)
        root = Path(share_dir)
        self.save_model(model, root / f"{scenario_name}_round_{server_round}")
        if server_round == server_rounds:
            self.save_model(model, root / scenario_name)

    def save_model(self, model, prefix):
        prefix = Path(prefix)
        self._parameters(model)
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(prefix) + ".safetensors")
        Path(str(prefix) + ".model.json").write_text(json.dumps({"format_version": 1, "framework": self.name, "model": model.aide_config}, indent=2) + "\n")

    def load_model(self, prefix):
        metadata = json.loads(Path(str(prefix) + ".model.json").read_text())
        if metadata.get("format_version") != 1 or metadata.get("framework") != self.name:
            raise ValueError("unsupported pytorch artifact format")
        model = self.build_model(metadata["model"])
        self._apply_arrays(model, ArrayRecord(torch_state_dict=load_file(str(prefix) + ".safetensors")))
        model.eval()
        return model

    def predict_proba(self, model, raw_features):
        x = self.adapt_features(raw_features, model.aide_config)
        model.eval()
        with torch.no_grad():
            positive = torch.sigmoid(model(torch.from_numpy(x))).numpy().reshape(-1)
        return np.column_stack((1 - positive, positive))
