import sys
from pathlib import Path

import numpy as np
import pytest
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flower"))

from flower.frameworks.base import ClientDatasets, HoldOutDataset
from flower.frameworks.scikit_learn_adapter import ScikitLearnAdapter
from flower.local_experiment import local_task_identity


def model(adapter, classes):
    return adapter.build_model(dict(penalty="l2", solver="saga", multi_class="auto",
                                    n_features=2, classes=classes))


@pytest.mark.parametrize("classes", [[0, 1], ["clear", "smoke"]])
def test_weighted_fedavg_does_not_change_class_labels(classes, tmp_path):
    adapter = ScikitLearnAdapter()
    global_model = model(adapter, classes)
    weights = adapter.initial_arrays(global_model)
    assert len(weights) == 2

    class Grid:
        def get_node_ids(self):
            return [1, 2, 3]

    with local_task_identity():
        strategy = FedAvg(fraction_train=1, min_available_nodes=3)
        requests = list(strategy.configure_train(1, weights, ConfigRecord(), Grid()))
        counts = [11, 17, 23]
        replies = [request.create_reply(RecordDict({
            "arrays": ArrayRecord([np.array([[value, -value]]), np.array([0.])]),
            "metrics": MetricRecord({"num-examples": count}),
        })) for request, count, value in zip(requests, counts, [0.5, 1., 1.5], strict=True)]
        aggregate, _ = strategy.aggregate_train(1, replies)

    x = np.array([[-1., 1.], [1., -1.]])
    y = np.array([classes[0], classes[1]])
    metrics = adapter.evaluate_centralised(global_model, aggregate,
                                           HoldOutDataset(x_hold_out=x, y_hold_out=y))
    assert metrics["accuracy"] == 1.
    np.testing.assert_array_equal(global_model.classes_, classes)
    np.testing.assert_allclose(global_model.coef_, [[57 / 51, -57 / 51]])
    adapter.save_server_artifacts(global_model, aggregate, str(tmp_path), "smoke", 1, 1)
    import joblib
    restored = joblib.load(tmp_path / "smoke.joblib")
    np.testing.assert_array_equal(restored.classes_, classes)
    np.testing.assert_array_equal(restored.predict(x), y)


def test_training_rejects_a_partition_missing_configured_class():
    adapter = ScikitLearnAdapter()
    local = model(adapter, [0, 1, 2])
    x = np.array([[0., 1.], [1., 0.]])
    y = np.array([0, 1])
    datasets = ClientDatasets(x_train=x, y_train=y, x_eval=x, y_eval=y, x_test=x, y_test=y)
    with pytest.raises(ValueError, match="all configured classes"):
        adapter.train(local, adapter.initial_arrays(local), datasets, ConfigRecord({"local_epochs": 1}), 0)
