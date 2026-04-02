from __future__ import annotations

import json
import os

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord
from safetensors.numpy import save_file
from tensorflow import keras

from flower.frameworks.base import ClientDatasets, FrameworkAdapter, HoldOutDataset, TrainResult


class TensorFlowAdapter(FrameworkAdapter):
    @property
    def name(self) -> str:
        return "tensorflow"

    def model_config_from_scenario(self, scenario: dict) -> dict:
        tensorflow = scenario.get("tensorflow")
        if not isinstance(tensorflow, dict):
            raise ValueError("scenario is missing tensorflow block")

        model = tensorflow.get("model")
        if not isinstance(model, dict):
            raise ValueError("scenario tensorflow block is missing model")

        model_config = model.get("config")
        if not isinstance(model_config, dict):
            raise ValueError("scenario tensorflow model is missing config")

        return model_config

    def build_model_from_scenario(self, scenario: dict):
        return self.build_model(self.model_config_from_scenario(scenario))

    def build_model(self, model_config: dict):
        return keras.models.model_from_json(json.dumps(model_config))

    def initial_arrays(self, model) -> ArrayRecord:
        return ArrayRecord(model.get_weights())

    def make_config_record(self, scenario: dict) -> ConfigRecord:
        training = scenario["training"]
        return ConfigRecord(
            {
                "framework": self.name,
                "model_format": "keras_json",
                "parameter_record_kind": "tensorflow-weights",
                "checkpoint_format": "keras_weights_h5",
                "final_artifact_format": "safetensors",
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
        model_data = json.loads(config["model_config"])
        return self.build_model(model_data)

    def get_model_config_from_record(self, config: ConfigRecord) -> dict:
        return json.loads(config["model_config"])

    def get_model_input_shape(self, model_config: dict) -> list:
        layers = model_config["config"]["layers"]
        if not layers:
            raise ValueError("model does not define any layers")

        input_config = layers[0]["config"]
        input_shape = input_config.get("batch_shape")
        if input_shape is None:
            input_shape = input_config.get("batch_input_shape")
        if input_shape is None:
            raise ValueError("model input layer is missing batch_shape")

        return input_shape

    def adapt_features(self, x: np.ndarray, model_config: dict) -> np.ndarray:
        input_shape = self.get_model_input_shape(model_config)
        if len(input_shape) == 2:
            return x

        if len(input_shape) == 3 and input_shape[2] == 1:
            return np.expand_dims(x, axis=-1)

        raise ValueError(f"unsupported model input shape: {input_shape}")

    def _apply_arrays(self, model, arrays: ArrayRecord) -> None:
        model.set_weights(arrays.to_numpy_ndarrays())

    def _metrics_from_evaluation(self, evaluation: dict, num_examples: int | None = None) -> MetricRecord:
        metrics = {}
        if num_examples is not None:
            metrics["num-examples"] = int(num_examples)

        metrics["loss"] = float(evaluation.pop("loss"))
        for key, value in evaluation.items():
            metrics[key] = float(value)

        return MetricRecord(metrics)

    def train(self, model, arrays: ArrayRecord, datasets: ClientDatasets, config: ConfigRecord, verbose) -> TrainResult:
        self._apply_arrays(model, arrays)
        history = model.fit(
            x=datasets.x_train,
            y=datasets.y_train,
            epochs=int(config["local_epochs"]),
            batch_size=int(config["batch_size"]),
            validation_data=(datasets.x_eval, datasets.y_eval),
            verbose=verbose,
        )

        history_metrics = history.history
        metrics = MetricRecord(
            {
                "num-examples": len(datasets.x_train),
                "train-loss": float(history_metrics["loss"][-1]),
                "train-val-loss": float(history_metrics.get("val_loss", [0.0])[-1]),
            }
        )

        return TrainResult(
            arrays=ArrayRecord(model.get_weights()),
            metrics=metrics,
            history=history_metrics,
        )

    def evaluate(self, model, arrays: ArrayRecord, datasets: ClientDatasets, config: ConfigRecord) -> MetricRecord:
        del config
        self._apply_arrays(model, arrays)
        evaluation = model.evaluate(datasets.x_test, datasets.y_test, verbose=0, return_dict=True)
        return self._metrics_from_evaluation(evaluation, num_examples=len(datasets.x_test))

    def evaluate_centralised(self, model, arrays: ArrayRecord, dataset: HoldOutDataset) -> MetricRecord:
        self._apply_arrays(model, arrays)
        evaluation = model.evaluate(dataset.x_hold_out, dataset.y_hold_out, verbose=0, return_dict=True)
        return self._metrics_from_evaluation(evaluation)

    def save_server_artifacts(
        self,
        model,
        arrays: ArrayRecord,
        share_dir: str,
        scenario_name: str,
        server_round: int,
        server_rounds: int,
    ) -> None:
        self._apply_arrays(model, arrays)

        checkpoint = os.path.join(share_dir, f"{scenario_name}_round_{server_round}.weights.h5")
        model.save_weights(checkpoint)
        print(f"Saved checkpoint to {checkpoint}")

        if server_round != server_rounds:
            return

        final_weights_path = os.path.join(share_dir, f"{scenario_name}.weights.h5")
        model.save_weights(final_weights_path)
        print(f"Saved final TensorFlow weights to {final_weights_path}")

        weights = arrays.to_numpy_ndarrays()
        weights_dict = {
            f"weight_{i}": w.astype(np.float32) if not np.issubdtype(w.dtype, np.floating) else w
            for i, w in enumerate(weights)
        }
        safetensors_path = os.path.join(share_dir, f"{scenario_name}.safetensors")
        save_file(weights_dict, safetensors_path)
        print(f"Saved final TensorFlow safetensors to {safetensors_path}")
