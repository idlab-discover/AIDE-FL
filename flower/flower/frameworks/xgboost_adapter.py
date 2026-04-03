from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from flower.frameworks.base import ClientDatasets, FrameworkAdapter, HoldOutDataset, TrainResult


class XGBoostAdapter(FrameworkAdapter):
    @property
    def name(self) -> str:
        return "xgboost"

    def model_config_from_scenario(self, scenario: dict) -> dict:
        xgboost = scenario.get("xgboost")
        if not isinstance(xgboost, dict):
            raise ValueError("scenario is missing xgboost block")

        model = xgboost.get("model")
        if not isinstance(model, dict):
            raise ValueError("scenario xgboost block is missing model")

        if model.get("kind") != "booster":
            raise ValueError(f"unsupported xgboost model kind: {model.get('kind')}")

        n_features = model.get("n_features")
        if not isinstance(n_features, int) or n_features < 1:
            raise ValueError("scenario xgboost model is missing a valid n_features value")

        params = xgboost.get("params")
        if not isinstance(params, dict):
            raise ValueError("scenario xgboost block is missing params")
        if params.get("objective") != "binary:logistic":
            raise ValueError(f"unsupported xgboost objective: {params.get('objective')}")

        return {
            "kind": model["kind"],
            "n_features": n_features,
            "params": params,
        }

    def build_model_from_scenario(self, scenario: dict):
        return self.build_model(self.model_config_from_scenario(scenario))

    def build_model(self, model_config: dict):
        return xgb.Booster(params=model_config["params"])

    def initial_arrays(self, model) -> ArrayRecord:
        del model
        return ArrayRecord([np.frombuffer(b"", dtype=np.uint8)])

    def make_config_record(self, scenario: dict) -> ConfigRecord:
        training = scenario["training"]
        model_config = self.model_config_from_scenario(scenario)
        return ConfigRecord(
            {
                "framework": self.name,
                "model_format": "xgboost_booster",
                "parameter_record_kind": "xgboost-booster-json",
                "checkpoint_format": "xgboost_json",
                "final_artifact_format": "xgboost_json",
                "model_config": json.dumps({"kind": model_config["kind"], "n_features": model_config["n_features"]}),
                "params": json.dumps(model_config["params"]),
                "local_epochs": int(training["local_epochs"]),
                "batch_size": int(training["batch_size"]),
                "label_column": scenario["label_column"],
            }
        )

    def state_reload_required(self, state_config: ConfigRecord | None, config: ConfigRecord) -> bool:
        if state_config is None:
            return True

        for key in ("framework", "model_format", "model_config", "params", "label_column"):
            if state_config.get(key) != config.get(key):
                return True

        return False

    def client_state_record(self, config: ConfigRecord) -> ConfigRecord:
        return ConfigRecord(
            {
                "framework": config["framework"],
                "model_format": config["model_format"],
                "model_config": config["model_config"],
                "params": config["params"],
                "label_column": config["label_column"],
            }
        )

    def restore_client_model(self, config: ConfigRecord):
        params = self.get_params_from_record(config)
        return xgb.Booster(params=params)

    def get_model_config_from_record(self, config: ConfigRecord) -> dict:
        return json.loads(config["model_config"])

    def get_params_from_record(self, config: ConfigRecord) -> dict:
        return json.loads(config["params"])

    def adapt_features(self, x: np.ndarray, model_config: dict) -> np.ndarray:
        expected_columns = int(model_config["n_features"])
        if x.shape[1] != expected_columns:
            raise ValueError(
                f"dataset has {x.shape[1]} feature columns but xgboost model expects {expected_columns}"
            )
        return x

    def train(self, model, arrays: ArrayRecord, datasets: ClientDatasets, config: ConfigRecord, verbose: Any) -> TrainResult:
        del verbose
        params = self.get_params_from_record(config)
        num_local_round = int(config["local_epochs"])
        train_matrix = xgb.DMatrix(datasets.x_train, label=datasets.y_train)

        raw_model = self._extract_booster_bytes(arrays)
        if raw_model:
            model.load_model(bytearray(raw_model))
            trained_model = self._local_boost(model, num_local_round, train_matrix)
        else:
            trained_model = xgb.train(params, train_matrix, num_boost_round=num_local_round)

        return TrainResult(
            arrays=self._arrays_from_booster(trained_model),
            metrics=MetricRecord({"num-examples": len(datasets.x_train)}),
        )

    def evaluate(self, model, arrays: ArrayRecord, datasets: ClientDatasets, config: ConfigRecord) -> MetricRecord:
        del config
        return self._evaluate(model, self._extract_booster_bytes(arrays), datasets.x_test, datasets.y_test, num_examples=len(datasets.x_test))

    def evaluate_centralised(self, model, arrays: ArrayRecord, dataset: HoldOutDataset) -> MetricRecord:
        return self._evaluate(model, self._extract_booster_bytes(arrays), dataset.x_hold_out, dataset.y_hold_out)

    def save_server_artifacts(
        self,
        model,
        arrays: ArrayRecord,
        share_dir: str,
        scenario_name: str,
        server_round: int,
        server_rounds: int,
    ) -> None:
        booster = self._load_booster(model, self._extract_booster_bytes(arrays))

        checkpoint = os.path.join(share_dir, f"{scenario_name}_round_{server_round}.json")
        booster.save_model(checkpoint)
        print(f"Saved checkpoint to {checkpoint}")

        if server_round != server_rounds:
            return

        final_model_path = os.path.join(share_dir, f"{scenario_name}.json")
        booster.save_model(final_model_path)
        print(f"Saved final XGBoost model to {final_model_path}")

    def _arrays_from_booster(self, booster: xgb.Booster) -> ArrayRecord:
        raw_model = booster.save_raw(raw_format="json")
        return ArrayRecord([np.frombuffer(raw_model, dtype=np.uint8)])

    def _extract_booster_bytes(self, arrays: ArrayRecord) -> bytes:
        ndarrays = arrays.to_numpy_ndarrays()
        if len(ndarrays) != 1:
            raise ValueError(f"expected a single booster payload, got {len(ndarrays)} arrays")
        return ndarrays[0].tobytes()

    def _load_booster(self, model: xgb.Booster, raw_model: bytes) -> xgb.Booster:
        if raw_model:
            model.load_model(bytearray(raw_model))
        return model

    def _local_boost(self, booster: xgb.Booster, num_local_round: int, train_matrix: xgb.DMatrix) -> xgb.Booster:
        for _ in range(num_local_round):
            booster.update(train_matrix, booster.num_boosted_rounds())

        end_round = booster.num_boosted_rounds()
        return booster[end_round - num_local_round : end_round]

    def _evaluate(
        self,
        model: xgb.Booster,
        raw_model: bytes,
        x: np.ndarray,
        y: np.ndarray,
        num_examples: int | None = None,
    ) -> MetricRecord:
        booster = self._load_booster(model, raw_model)
        matrix = xgb.DMatrix(x, label=y)
        predictions = booster.predict(matrix)
        prediction_labels = (predictions >= 0.5).astype(int)

        metrics = {
            "auc": float(roc_auc_score(y, predictions)),
            "accuracy": float(accuracy_score(y, prediction_labels)),
            "logloss": float(log_loss(y, predictions, labels=[0, 1])),
        }
        if num_examples is not None:
            metrics["num-examples"] = int(num_examples)
        return MetricRecord(metrics)
