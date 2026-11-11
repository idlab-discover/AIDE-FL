import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flower"))
from flower.frameworks import get_adapter
from flower.frameworks.base import ClientDatasets, HoldOutDataset
from flower.local_experiment import local_task_identity


def scenario(kind="mlp"):
    return {"framework": "pytorch", "label_column": "label", "training": {"local_epochs": 3, "batch_size": 8},
            "aggregation": {"strategy": "fedavg"}, "pytorch": {
                "model": {"kind": kind, "n_features": 2, "hidden_sizes": [8] if kind == "mlp" else [], "seed": 7},
                "optimizer": {"kind": "sgd", "learning_rate": .1}}}


@pytest.mark.parametrize("kind", ["mlp", "logistic_regression"])
def test_training_aggregation_evaluation_and_reload(kind, tmp_path):
    adapter = get_adapter("pytorch")
    spec = scenario(kind)
    model = adapter.build_model_from_scenario(spec)
    arrays = adapter.initial_arrays(model)
    config = adapter.make_config_record(spec)
    x = np.array([[-2, -1], [-1, -2], [1, 2], [2, 1]] * 16, dtype=np.float32)
    y = np.array([0, 0, 1, 1] * 16)
    data = ClientDatasets(x, y, x, y, x, y)
    before = adapter.evaluate_centralised(model, arrays, HoldOutDataset(x, y))["loss"]
    models = [adapter.restore_client_model(config) for _ in range(2)]
    results = []
    for index, local in enumerate(models):
        local_config = ConfigRecord(dict(config))
        local_config["learning_rate"] *= index + 1
        results.append(adapter.train(local, arrays, data, local_config, 0))
    class Grid:
        def get_node_ids(self):
            return [1, 2]
    with local_task_identity():
        strategy = FedAvg(min_available_nodes=2)
        messages = strategy.configure_train(1, arrays, config, Grid())
        replies = [m.create_reply(RecordDict({"arrays": r.arrays, "metrics": MetricRecord({"num-examples": n})}))
                   for m, r, n in zip(messages, results, [1, 3])]
        averaged, _ = strategy.aggregate_train(1, replies)
    for key in arrays:
        np.testing.assert_allclose(averaged[key].numpy(), (results[0].arrays[key].numpy() + 3 * results[1].arrays[key].numpy()) / 4)
    assert adapter.evaluate(model, averaged, data, config)["loss"] < before
    adapter.save_server_artifacts(model, averaged, str(tmp_path), "test", 1, 1)
    restored = adapter.load_model(tmp_path / "test")
    np.testing.assert_array_equal(adapter.predict_proba(model, x), adapter.predict_proba(restored, x))
    assert (tmp_path / "test_round_1.safetensors").exists()


@pytest.mark.parametrize("buffer", [torch.tensor(0), torch.zeros(2)])
def test_buffers_never_enter_fedavg(buffer):
    adapter = get_adapter("pytorch")
    model = adapter.build_model_from_scenario(scenario())
    model.register_buffer("counter", buffer)
    with pytest.raises(ValueError, match="buffers"):
        adapter.initial_arrays(model)


def test_frozen_parameters_are_not_averaged():
    adapter = get_adapter("pytorch")
    model = adapter.build_model_from_scenario(scenario())
    next(model.parameters()).requires_grad_(False)
    with pytest.raises(ValueError, match="trainable"):
        adapter.initial_arrays(model)


@pytest.mark.parametrize("corruption", ["name", "order", "shape", "dtype", "nan"])
def test_parameter_layout_is_checked_before_loading(corruption):
    adapter = get_adapter("pytorch")
    model = adapter.build_model_from_scenario(scenario())
    arrays = adapter.initial_arrays(model)
    key = next(iter(arrays))
    if corruption == "name":
        arrays["wrong"] = arrays.pop(key)
    elif corruption == "order":
        arrays[key] = arrays.pop(key)
    else:
        value = arrays[key].numpy()
        value = value.reshape(-1)[:1] if corruption == "shape" else value.astype(np.float64) if corruption == "dtype" else value * np.nan
        arrays[key] = Array(value)
    with pytest.raises(ValueError, match="parameter"):
        adapter.validate_arrays(model, arrays)


@pytest.mark.parametrize("change", [{"kind": "cnn"}, {"n_features": True}, {"hidden_sizes": []}, {"seed": -1},
                                    {"feature_mean": [0, 0]}, {"feature_mean": [0, 0], "feature_scale": [1, 0]}])
def test_invalid_model_configuration(change):
    adapter = get_adapter("pytorch")
    spec = scenario()
    spec["pytorch"]["model"].update(change)
    with pytest.raises(ValueError):
        adapter.build_model_from_scenario(spec)


def test_normalization_and_configuration_survive_checkpoint(tmp_path):
    adapter = get_adapter("pytorch")
    spec = scenario()
    spec["pytorch"]["model"].update(feature_mean=[1, 2], feature_scale=[2, 4])
    model = adapter.build_model_from_scenario(spec)
    config = adapter.make_config_record(spec)
    assert not adapter.state_reload_required(adapter.client_state_record(config), config)
    adapter.save_model(model, tmp_path / "model")
    restored = adapter.load_model(tmp_path / "model")
    assert restored.aide_config == model.aide_config
    np.testing.assert_array_equal(adapter.adapt_features([[3, 6]], model.aide_config), [[1, 1]])


def test_privacy_and_other_strategies_rejected():
    adapter = get_adapter("pytorch")
    spec = scenario()
    spec["aggregation"]["privacy"] = {"mode": "central_dp"}
    with pytest.raises(ValueError, match="TensorFlow"):
        adapter.make_config_record(spec)
    del spec["aggregation"]["privacy"]
    spec["aggregation"]["strategy"] = "fedprox"
    with pytest.raises(ValueError, match="FedAvg"):
        adapter.make_config_record(spec)


def test_scenario_schema_and_rendering():
    import importlib.util
    import jsonschema
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("render", root / "render_scenario.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rendered = module.render_scenario(root / "examples/smoke-pytorch.json")
    schema = json.loads((root / "examples/scenario.schema.json").read_text())
    jsonschema.validate(rendered, schema)
    get_adapter("pytorch").make_config_record(rendered)
    rendered["pytorch"]["model"]["batchnorm"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(rendered, schema)


def test_production_aggregation_rejects_corrupt_reply(tmp_path, monkeypatch):
    import wandb
    from flower.strategies import build_strategy
    monkeypatch.setattr(wandb, "init", lambda **kwargs: None)
    adapter = get_adapter("pytorch")
    spec = scenario()
    model = adapter.build_model_from_scenario(spec)
    arrays = adapter.initial_arrays(model)
    strategy = build_strategy({"strategy": "fedavg", "fraction_fit": 1., "fraction_evaluate": 1., "min_available_clients": 1},
                              scenario_name="test", model=model, adapter=adapter, share_dir=str(tmp_path), server_rounds=1)
    class Grid:
        def get_node_ids(self):
            return [1]
    with local_task_identity():
        request = next(iter(strategy.configure_train(1, arrays, ConfigRecord(), Grid())))
        key = next(iter(arrays))
        arrays[key] = Array(arrays[key].numpy().astype(np.float64))
        reply = request.create_reply(RecordDict({"arrays": arrays, "metrics": MetricRecord({"num-examples": 2})}))
        with pytest.raises(ValueError, match="dtype"):
            strategy.aggregate_train(1, [reply])
    assert not list(tmp_path.iterdir())
