from __future__ import annotations

import json
import os
from inspect import signature
from typing import Any

import joblib
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

from flower.frameworks.base import ClientDatasets, FrameworkAdapter, HoldOutDataset, TrainResult


class ScikitLearnAdapter(FrameworkAdapter):
    @property
    def name(self) -> str:
        return "scikit_learn"

    def model_config_from_scenario(self, scenario: dict) -> dict:
        scikit_learn = scenario.get("scikit_learn")
        if not isinstance(scikit_learn, dict):
            raise ValueError("scenario is missing scikit_learn block")

        model = scikit_learn.get("model")
        if not isinstance(model, dict):
            raise ValueError("scenario scikit_learn block is missing model")

        kind = model.get("kind")
        if kind != "logistic_regression":
            raise ValueError(f"unsupported scikit_learn model kind: {kind}")

        n_features = model.get("n_features")
        if not isinstance(n_features, int) or n_features < 1:
            raise ValueError("scenario scikit_learn model is missing a valid n_features value")

        classes = model.get("classes")
        if not isinstance(classes, list) or len(classes) < 2:
            raise ValueError("scenario scikit_learn model is missing at least two classes")

        penalty = model.get("penalty", "l2")
        solver = model.get("solver", "saga")
        multi_class = model.get("multi_class", "auto")

        return {
            "kind": kind,
            "n_features": n_features,
            "classes": classes,
            "penalty": penalty,
            "solver": solver,
            "multi_class": multi_class,
        }

    def build_model_from_scenario(self, scenario: dict):
        return self.build_model(self.model_config_from_scenario(scenario))

    def build_model(self, model_config: dict):
        kwargs = {
            "penalty": model_config["penalty"],
            "max_iter": 1,
            "warm_start": True,
            "solver": model_config["solver"],
        }
        if "multi_class" in signature(LogisticRegression).parameters:
            kwargs["multi_class"] = model_config["multi_class"]

        model = LogisticRegression(**kwargs)
        self._set_model_params(
            model,
            self._zero_arrays(model_config["n_features"], model_config["classes"]),
        )
        return model

    def initial_arrays(self, model) -> ArrayRecord:
        return ArrayRecord(self._get_model_params(model))

    def make_config_record(self, scenario: dict) -> ConfigRecord:
        training = scenario["training"]
        return ConfigRecord(
            {
                "framework": self.name,
                "model_format": "scikit_logistic_regression",
                "parameter_record_kind": "scikit-logreg-parameters",
                "checkpoint_format": "joblib",
                "final_artifact_format": "joblib",
                "model_config": json.dumps(self.model_config_from_scenario(scenario)),
                "local_epochs": int(training["local_epochs"]),
                "batch_size": int(training["batch_size"]),
                "label_column": scenario["label_column"],
            }
        )

    def state_reload_required(self, state_config: ConfigRecord | None, config: ConfigRecord) -> bool:
        if state_config is None:
            return True

        for key in ("framework", "model_format", "model_config", "label_column"):
            if state_config.get(key) != config.get(key):
                return True

        return False

    def client_state_record(self, config: ConfigRecord) -> ConfigRecord:
        return ConfigRecord(
            {
                "framework": config["framework"],
                "model_format": config["model_format"],
                "model_config": config["model_config"],
                "label_column": config["label_column"],
            }
        )

    def restore_client_model(self, config: ConfigRecord):
        model_config = self.get_model_config_from_record(config)
        return self.build_model(model_config)

    def get_model_config_from_record(self, config: ConfigRecord) -> dict:
        return json.loads(config["model_config"])

    def adapt_features(self, x: np.ndarray, model_config: dict) -> np.ndarray:
        expected_columns = int(model_config["n_features"])
        if x.shape[1] != expected_columns:
            raise ValueError(
                f"dataset has {x.shape[1]} feature columns but scikit_learn model expects {expected_columns}"
            )
        return x

    def train(self, model, arrays: ArrayRecord, datasets: ClientDatasets, config: ConfigRecord, verbose: Any) -> TrainResult:
        del verbose
        self._set_model_params(model, arrays.to_numpy_ndarrays())
        model.max_iter = int(config["local_epochs"])
        model.fit(datasets.x_train, datasets.y_train)

        y_train_pred_proba = model.predict_proba(datasets.x_train)
        y_train_pred = model.predict(datasets.x_train)
        metrics = MetricRecord(
            {
                "num-examples": len(datasets.x_train),
                "train_logloss": float(log_loss(datasets.y_train, y_train_pred_proba, labels=model.classes_)),
                "train_accuracy": float(accuracy_score(datasets.y_train, y_train_pred)),
            }
        )

        return TrainResult(
            arrays=ArrayRecord(self._get_model_params(model)),
            metrics=metrics,
        )

    def evaluate(self, model, arrays: ArrayRecord, datasets: ClientDatasets, config: ConfigRecord) -> MetricRecord:
        del config
        self._set_model_params(model, arrays.to_numpy_ndarrays())
        return self._evaluate_metrics(model, datasets.x_test, datasets.y_test, num_examples=len(datasets.x_test))

    def evaluate_centralised(self, model, arrays: ArrayRecord, dataset: HoldOutDataset) -> MetricRecord:
        self._set_model_params(model, arrays.to_numpy_ndarrays())
        return self._evaluate_metrics(model, dataset.x_hold_out, dataset.y_hold_out)

    def save_server_artifacts(
        self,
        model,
        arrays: ArrayRecord,
        share_dir: str,
        scenario_name: str,
        server_round: int,
        server_rounds: int,
    ) -> None:
        self._set_model_params(model, arrays.to_numpy_ndarrays())

        checkpoint = os.path.join(share_dir, f"{scenario_name}_round_{server_round}.joblib")
        joblib.dump(model, checkpoint)
        print(f"Saved checkpoint to {checkpoint}")

        if server_round != server_rounds:
            return

        final_model_path = os.path.join(share_dir, f"{scenario_name}.joblib")
        joblib.dump(model, final_model_path)
        print(f"Saved final scikit-learn model to {final_model_path}")

    def _evaluate_metrics(self, model, x, y, num_examples: int | None = None) -> MetricRecord:
        y_pred = model.predict(x)
        y_pred_proba = model.predict_proba(x)

        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
            "logloss": float(log_loss(y, y_pred_proba, labels=model.classes_)),
        }
        if num_examples is not None:
            metrics["num-examples"] = int(num_examples)

        return MetricRecord(metrics)

    def _get_model_params(self, model) -> list[np.ndarray]:
        return [
            np.asarray(model.coef_, dtype=np.float64),
            np.asarray(model.intercept_, dtype=np.float64),
            np.asarray(model.classes_),
        ]

    def _set_model_params(self, model, ndarrays: list[np.ndarray]) -> None:
        coef, intercept, classes = ndarrays
        model.classes_ = np.asarray(classes)
        model.coef_ = np.asarray(coef, dtype=np.float64)
        model.intercept_ = np.asarray(intercept, dtype=np.float64)
        model.n_features_in_ = model.coef_.shape[1]

    def _zero_arrays(self, n_features: int, classes: list[Any]) -> list[np.ndarray]:
        class_values = np.asarray(classes)
        n_classes = len(class_values)
        if n_classes == 2:
            coef_shape = (1, n_features)
            intercept_shape = (1,)
        else:
            coef_shape = (n_classes, n_features)
            intercept_shape = (n_classes,)

        return [
            np.zeros(coef_shape, dtype=np.float64),
            np.zeros(intercept_shape, dtype=np.float64),
            class_values,
        ]
