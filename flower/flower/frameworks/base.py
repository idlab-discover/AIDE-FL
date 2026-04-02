from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from flwr.app import ArrayRecord, ConfigRecord, MetricRecord


@dataclass
class ClientDatasets:
    x_train: Any
    y_train: Any
    x_eval: Any
    y_eval: Any
    x_test: Any
    y_test: Any


@dataclass
class HoldOutDataset:
    x_hold_out: Any
    y_hold_out: Any


@dataclass
class TrainResult:
    arrays: ArrayRecord
    metrics: MetricRecord
    history: dict[str, Any] | None = None


class FrameworkAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def model_config_from_scenario(self, scenario: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def build_model_from_scenario(self, scenario: dict) -> Any:
        raise NotImplementedError

    @abstractmethod
    def initial_arrays(self, model: Any) -> ArrayRecord:
        raise NotImplementedError

    @abstractmethod
    def make_config_record(self, scenario: dict) -> ConfigRecord:
        raise NotImplementedError

    @abstractmethod
    def state_reload_required(self, state_config: ConfigRecord | None, config: ConfigRecord) -> bool:
        raise NotImplementedError

    @abstractmethod
    def client_state_record(self, config: ConfigRecord) -> ConfigRecord:
        raise NotImplementedError

    @abstractmethod
    def restore_client_model(self, config: ConfigRecord) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_model_config_from_record(self, config: ConfigRecord) -> dict:
        raise NotImplementedError

    @abstractmethod
    def adapt_features(self, x, model_config: dict):
        raise NotImplementedError

    @abstractmethod
    def train(self, model: Any, arrays: ArrayRecord, datasets: ClientDatasets, config: ConfigRecord, verbose: Any) -> TrainResult:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model: Any, arrays: ArrayRecord, datasets: ClientDatasets, config: ConfigRecord) -> MetricRecord:
        raise NotImplementedError

    @abstractmethod
    def evaluate_centralised(self, model: Any, arrays: ArrayRecord, dataset: HoldOutDataset) -> MetricRecord:
        raise NotImplementedError

    @abstractmethod
    def save_server_artifacts(
        self,
        model: Any,
        arrays: ArrayRecord,
        share_dir: str,
        scenario_name: str,
        server_round: int,
        server_rounds: int,
    ) -> None:
        raise NotImplementedError
