import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flower"))

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flower.privacy import make_private_strategy, validate_privacy
from flower.experimental_fhe import (
    aggregate_encrypted, create_private_context, decrypt_weights, encrypt_weights,
    public_context_bytes,
)


def privacy_config():
    return dict(mode="central_dp", noise_multiplier=1.0, clipping_norm=1.0,
                num_sampled_clients=2, population_size=2, target_delta=1e-5)


class TwoClients:
    def get_node_ids(self):
        return [1, 2]


def strategy():
    return make_private_strategy(privacy_config(), dict(
        fraction_train=1.0, min_available_nodes=2, weighted_by_key="num-examples"))


def private_round(dp, round_nr=1, values=(100.0, 200.0)):
    messages = list(dp.configure_train(round_nr, ArrayRecord([np.array([0.0])]),
                                      ConfigRecord(), TwoClients()))
    assert all(message.content["config"]["privacy-mode"] == "central_dp" for message in messages)
    return [message.create_reply(RecordDict({
        "arrays": ArrayRecord([np.array([value])]),
        "metrics": MetricRecord({"num-examples": node_id * 100, "private-loss": 42.0}),
        "history": ConfigRecord({"history": "private details"}),
    })) for node_id, (message, value) in enumerate(zip(messages, values), 1)]


def test_native_dp_clips_noises_accounts_and_removes_private_metrics():
    dp = strategy()
    replies = private_round(dp)
    arrays, metrics = dp.aggregate_train(1, replies)
    assert arrays is not None and metrics is None
    assert abs(arrays.to_numpy_ndarrays()[0][0]) < 5
    spent = dp.privacy_spent()
    assert spent.epsilon > 0 and spent.delta == 1e-5 and spent.num_releases == 1
    assert all(set(reply.content) == {"arrays", "metrics"} for reply in replies)
    assert all(dict(reply.content["metrics"]) == {"num-examples": 1} for reply in replies)
    assert list(dp.configure_evaluate(1, arrays, ConfigRecord(), TwoClients())) == []


def test_private_round_missing_client_is_not_released():
    dp = strategy()
    replies = private_round(dp)
    with pytest.raises(RuntimeError, match="aborted"):
        dp.aggregate_train(1, replies[:1])
    assert dp.privacy_spent().num_releases == 0


def test_private_round_rejects_duplicate_sender_before_accounting():
    dp = strategy()
    replies = private_round(dp)
    with pytest.raises(RuntimeError, match="distinct sampled clients"):
        dp.aggregate_train(1, [replies[0], replies[0]])
    assert dp.privacy_spent().num_releases == 0


def test_private_round_rejects_reordered_parameter_keys():
    from flwr.app import Array

    dp = strategy()
    initial = ArrayRecord({"first": Array(np.array([1.])), "second": Array(np.array([2.]))})
    messages = list(dp.configure_train(1, initial, ConfigRecord(), TwoClients()))
    replies = [message.create_reply(RecordDict({
        "arrays": ArrayRecord({"second": Array(np.array([3.])), "first": Array(np.array([4.]))}),
        "metrics": MetricRecord({"num-examples": 1}),
    })) for message in messages]
    with pytest.raises(ValueError, match="array layout"):
        dp.aggregate_train(1, replies)
    assert dp.privacy_spent().num_releases == 0


def test_private_round_rejects_nan_before_accounting():
    dp = strategy()
    replies = private_round(dp, values=(float("nan"), 1.0))
    with pytest.raises(ValueError, match="non-finite"):
        dp.aggregate_train(1, replies)
    assert dp.privacy_spent().num_releases == 0


def test_privacy_budget_refuses_first_round_before_any_release():
    from flwr.serverapp.exception import PrivacyBudgetExhausted

    config = privacy_config()
    config["max_epsilon"] = 0.01
    dp = make_private_strategy(config, dict(fraction_train=1.0, min_available_nodes=2))
    with pytest.raises(PrivacyBudgetExhausted):
        private_round(dp)
    assert dp.privacy_spent().num_releases == 0


def test_runtime_saves_only_the_noisy_released_model(tmp_path, monkeypatch):
    from flower.strategies import build_strategy
    import wandb

    monkeypatch.setattr(wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(wandb, "log", lambda *args, **kwargs: None)

    class Adapter:
        name = "tensorflow"
        saved = []

        def save_server_artifacts(self, **kwargs):
            self.saved.append(kwargs["arrays"].to_numpy_ndarrays())

    adapter = Adapter()
    dp = build_strategy(dict(strategy="fedavg", fraction_fit=1.0, fraction_evaluate=1.0,
                             min_available_clients=2, privacy=privacy_config()),
                        scenario_name="private", model=None, adapter=adapter,
                        share_dir=str(tmp_path), server_rounds=1)
    arrays, _ = dp.aggregate_train(1, private_round(dp))
    assert len(adapter.saved) == 1
    np.testing.assert_array_equal(adapter.saved[0][0], arrays.to_numpy_ndarrays()[0])
    assert (tmp_path / "private_privacy.json").exists()
    assert not (tmp_path / "private_fitted_validated_metrics.json").exists()


@pytest.mark.parametrize("key,value", [("noise_multiplier", 0), ("clipping_norm", float("nan")),
                                      ("num_sampled_clients", True), ("population_size", 1),
                                      ("target_delta", 0.5), ("max_epsilon", -1)])
def test_invalid_dp_configuration(key, value):
    config = privacy_config()
    config[key] = value
    with pytest.raises(ValueError):
        validate_privacy({"strategy": "fedavg", "privacy": config}, "tensorflow")


def test_private_strategy_rejects_non_float_model_metadata():
    dp = strategy()
    with pytest.raises(ValueError, match="floating-point"):
        dp.configure_train(1, ArrayRecord([np.array([0, 1])]), ConfigRecord(), TwoClients())


def test_encrypted_fedavg_uses_public_only_aggregation_and_matches_plaintext():
    pytest.importorskip("tenseal")
    private = create_private_context()
    public = public_context_bytes(private)
    weights = [[np.array([[1., -2.], [3., 4.]], dtype=np.float32), np.array([0.25])],
               [np.array([[4., 1.], [0., -1.]], dtype=np.float32), np.array([-0.5])]]
    updates = [encrypt_weights(public, arrays, count) for arrays, count in zip(weights, [2, 7])]
    combined = aggregate_encrypted(public, updates)
    actual = decrypt_weights(private, combined)
    for i in range(2):
        np.testing.assert_allclose(actual[i], (weights[0][i] * 2 + weights[1][i] * 7) / 9, atol=1e-6)
        assert actual[i].dtype == weights[0][i].dtype
    with pytest.raises(ValueError, match="public-only"):
        aggregate_encrypted(private.serialize(save_secret_key=True), updates)
    with pytest.raises(ValueError, match="floating-point"):
        encrypt_weights(public, [np.array([0, 1])], 1)
    with pytest.raises(ValueError, match="at least two"):
        aggregate_encrypted(public, updates[:1])
    other_private = create_private_context()
    with pytest.raises(ValueError, match="different private context"):
        decrypt_weights(other_private, combined)
    other = encrypt_weights(public_context_bytes(other_private), weights[0], 2)
    with pytest.raises(ValueError, match="different public context"):
        aggregate_encrypted(public, [updates[0], other])


def test_native_secagg_masks_individual_weights_and_matches_fedavg():
    from flower.experimental_secagg import aggregate_secagg

    updates = [([np.array([float(i), 0.1, -0.5])], i) for i in [1, 2, 3]]
    actual, protocol = aggregate_secagg(updates)
    expected = sum(arrays[0] * count for arrays, count in updates) / 6
    np.testing.assert_allclose(actual[0], expected, atol=2e-5)
    assert protocol["stages"] == 4 and protocol["message_replies"] == 12
    assert protocol["plaintext_array_values_in_replies"] == 0
    with pytest.raises(ValueError, match="at least three"):
        aggregate_secagg(updates[:2])
