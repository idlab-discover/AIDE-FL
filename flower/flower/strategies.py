import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

import wandb
from flwr.app import MetricRecord
from flwr.serverapp.strategy import Bulyan, FedAvg, FedXgbBagging, FedXgbCyclic
from flwr.serverapp.strategy.fedavg import aggregate_arrayrecords

from flower.frameworks.base import FrameworkAdapter

ARRAYS_KEY = "arrays"
CONFIG_KEY = "config"
METRICS_KEY = "metrics"
HISTORY_KEY = "history"
WEIGHTED_BY_KEY = "num-examples"


def aggregate_numeric_metrics(contents, weighted_by_key: str) -> MetricRecord:
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(float)
    total_weight = 0.0

    for content in contents:
        metrics = content.metric_records.get(METRICS_KEY)
        if metrics is None:
            continue

        weight = float(metrics.get(weighted_by_key, 0))
        if weight <= 0:
            continue
        total_weight += weight

        for key, value in metrics.items():
            if key == weighted_by_key or isinstance(value, list):
                continue
            metric_sums[key] += float(value) * weight
            metric_counts[key] += weight

    aggregated = {}
    if total_weight > 0:
        aggregated[weighted_by_key] = int(total_weight)

    for key, total in metric_sums.items():
        aggregated[key] = total / metric_counts[key]

    return MetricRecord(aggregated)


class SaveModelStrategyMixin:
    def __init__(
        self,
        scenario_name: str,
        model: Any,
        adapter: FrameworkAdapter,
        share_dir: str,
        server_rounds: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scenario_name = scenario_name
        self.model = model
        self.adapter = adapter
        self.share_dir = share_dir
        self.server_rounds = server_rounds
        self.fitted_validated_metrics = {}
        self.distributed_metrics = {}

        date_name = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        wandb.init(project="aide-flower-deployment", name=f"{scenario_name}-{date_name}")

    def _save_train_results(self, server_round, arrays, metrics):
        if arrays is None:
            return arrays, metrics

        self.adapter.save_server_artifacts(
            model=self.model,
            arrays=arrays,
            share_dir=self.share_dir,
            scenario_name=self.scenario_name,
            server_round=server_round,
            server_rounds=self.server_rounds,
        )

        if metrics is not None:
            wandb.log(
                {f"train/{key}": value for key, value in metrics.items() if key != WEIGHTED_BY_KEY},
                step=server_round,
            )

        return arrays, metrics

    def _record_train_histories(self, server_round, valid_replies, metrics):
        histories = []
        for reply in valid_replies:
            history = reply.content.config_records.get(HISTORY_KEY)
            if history is None:
                continue
            history_json = history.get(HISTORY_KEY)
            if history_json:
                histories.append(json.loads(history_json))

        self.fitted_validated_metrics[server_round] = {
            "aggregated": dict(metrics) if metrics is not None else {},
            "histories": histories,
        }
        self._write_json(
            f"{self.scenario_name}_fitted_validated_metrics.json",
            self.fitted_validated_metrics,
        )

    def aggregate_evaluate(self, server_round, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)
        metrics = None
        if valid_replies:
            reply_contents = [reply.content for reply in valid_replies]
            metrics = self.evaluate_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        client_metrics = []
        for reply in valid_replies:
            metric_record = reply.content.metric_records.get(METRICS_KEY)
            if metric_record is not None:
                client_metrics.append(dict(metric_record))

        self.distributed_metrics[server_round] = {
            "aggregated": dict(metrics) if metrics is not None else {},
            "clients": client_metrics,
        }
        self._write_json(
            f"{self.scenario_name}_distributed_metrics.json",
            self.distributed_metrics,
        )

        if metrics is not None:
            wandb.log(
                {f"distributed/{key}": value for key, value in metrics.items() if key != WEIGHTED_BY_KEY},
                step=server_round,
            )

        return metrics

    def _write_json(self, filename: str, payload):
        path = os.path.join(self.share_dir, filename)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)
        print(f"Updated {filename} in {path}")


class SaveModelFedAvg(SaveModelStrategyMixin, FedAvg):
    def aggregate_train(self, server_round, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        arrays = metrics = None
        if valid_replies:
            reply_contents = [reply.content for reply in valid_replies]
            arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
            metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        self._record_train_histories(server_round, valid_replies, metrics)
        return self._save_train_results(server_round, arrays, metrics)


class SaveModelBulyan(SaveModelStrategyMixin, Bulyan):
    def aggregate_train(self, server_round, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        arrays, metrics = super().aggregate_train(server_round, valid_replies)
        self._record_train_histories(server_round, valid_replies, metrics)
        return self._save_train_results(server_round, arrays, metrics)


class SaveXGBoostBagging(FedXgbBagging):
    def __init__(
        self,
        scenario_name: str,
        model: Any,
        adapter: FrameworkAdapter,
        share_dir: str,
        server_rounds: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scenario_name = scenario_name
        self.model = model
        self.adapter = adapter
        self.share_dir = share_dir
        self.server_rounds = server_rounds
        self.distributed_metrics = {}

        date_name = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        wandb.init(project="aide-flower-deployment", name=f"{scenario_name}-{date_name}")

    def aggregate_train(self, server_round, replies):
        arrays, metrics = super().aggregate_train(server_round, replies)
        if arrays is not None:
            self.adapter.save_server_artifacts(
                model=self.model,
                arrays=arrays,
                share_dir=self.share_dir,
                scenario_name=self.scenario_name,
                server_round=server_round,
                server_rounds=self.server_rounds,
            )

        if metrics is not None:
            wandb.log(
                {f"train/{key}": value for key, value in metrics.items() if key != WEIGHTED_BY_KEY},
                step=server_round,
            )
        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)
        metrics = None
        if valid_replies:
            reply_contents = [reply.content for reply in valid_replies]
            metrics = self.evaluate_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        client_metrics = []
        for reply in valid_replies:
            metric_record = reply.content.metric_records.get(METRICS_KEY)
            if metric_record is not None:
                client_metrics.append(dict(metric_record))

        self.distributed_metrics[server_round] = {
            "aggregated": dict(metrics) if metrics is not None else {},
            "clients": client_metrics,
        }
        self._write_json(
            f"{self.scenario_name}_distributed_metrics.json",
            self.distributed_metrics,
        )

        if metrics is not None:
            wandb.log(
                {f"distributed/{key}": value for key, value in metrics.items() if key != WEIGHTED_BY_KEY},
                step=server_round,
            )
        return metrics

    def _write_json(self, filename: str, payload):
        path = os.path.join(self.share_dir, filename)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)
        print(f"Updated {filename} in {path}")


class SaveXGBoostCyclic(FedXgbCyclic):
    def __init__(
        self,
        scenario_name: str,
        model: Any,
        adapter: FrameworkAdapter,
        share_dir: str,
        server_rounds: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scenario_name = scenario_name
        self.model = model
        self.adapter = adapter
        self.share_dir = share_dir
        self.server_rounds = server_rounds
        self.distributed_metrics = {}

        date_name = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        wandb.init(project="aide-flower-deployment", name=f"{scenario_name}-{date_name}")

    def aggregate_train(self, server_round, replies):
        arrays, metrics = super().aggregate_train(server_round, replies)
        if arrays is not None:
            self.adapter.save_server_artifacts(
                model=self.model,
                arrays=arrays,
                share_dir=self.share_dir,
                scenario_name=self.scenario_name,
                server_round=server_round,
                server_rounds=self.server_rounds,
            )

        if metrics is not None:
            wandb.log(
                {f"train/{key}": value for key, value in metrics.items() if key != WEIGHTED_BY_KEY},
                step=server_round,
            )
        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)
        metrics = None
        if valid_replies:
            reply_contents = [reply.content for reply in valid_replies]
            metrics = self.evaluate_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        client_metrics = []
        for reply in valid_replies:
            metric_record = reply.content.metric_records.get(METRICS_KEY)
            if metric_record is not None:
                client_metrics.append(dict(metric_record))

        self.distributed_metrics[server_round] = {
            "aggregated": dict(metrics) if metrics is not None else {},
            "clients": client_metrics,
        }
        self._write_json(
            f"{self.scenario_name}_distributed_metrics.json",
            self.distributed_metrics,
        )

        if metrics is not None:
            wandb.log(
                {f"distributed/{key}": value for key, value in metrics.items() if key != WEIGHTED_BY_KEY},
                step=server_round,
            )
        return metrics

    def _write_json(self, filename: str, payload):
        path = os.path.join(self.share_dir, filename)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)
        print(f"Updated {filename} in {path}")


def build_strategy(
    aggregation: dict,
    *,
    scenario_name: str,
    model: Any,
    adapter: FrameworkAdapter,
    share_dir: str,
    server_rounds: int,
):
    strategy_name = aggregation["strategy"]
    num_malicious_nodes = int(aggregation.get("num_malicious_nodes", 0))
    base_kwargs = {
        "fraction_train": aggregation["fraction_fit"],
        "fraction_evaluate": aggregation["fraction_evaluate"],
        "min_available_nodes": aggregation["min_available_clients"],
        "arrayrecord_key": ARRAYS_KEY,
        "configrecord_key": CONFIG_KEY,
        "weighted_by_key": WEIGHTED_BY_KEY,
        "train_metrics_aggr_fn": aggregate_numeric_metrics,
        "evaluate_metrics_aggr_fn": aggregate_numeric_metrics,
    }
    fedavg_kwargs = {
        **base_kwargs,
        "min_train_nodes": 1,
        "min_evaluate_nodes": 1,
    }

    if strategy_name == "fedavg":
        return SaveModelFedAvg(
            scenario_name=scenario_name,
            model=model,
            adapter=adapter,
            share_dir=share_dir,
            server_rounds=server_rounds,
            **fedavg_kwargs,
        )

    if strategy_name == "bulyan":
        return SaveModelBulyan(
            scenario_name=scenario_name,
            model=model,
            adapter=adapter,
            share_dir=share_dir,
            server_rounds=server_rounds,
            num_malicious_nodes=num_malicious_nodes,
            **fedavg_kwargs,
        )

    if strategy_name == "fedxgbbagging":
        return SaveXGBoostBagging(
            scenario_name=scenario_name,
            model=model,
            adapter=adapter,
            share_dir=share_dir,
            server_rounds=server_rounds,
            **base_kwargs,
        )

    if strategy_name == "fedxgbcyclic":
        return SaveXGBoostCyclic(
            scenario_name=scenario_name,
            model=model,
            adapter=adapter,
            share_dir=share_dir,
            server_rounds=server_rounds,
            **base_kwargs,
        )

    raise ValueError(f"unsupported aggregation strategy: {strategy_name}")
