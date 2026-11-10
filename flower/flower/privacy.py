"""Accounted central DP for floating-point FedAvg models.

The server is trusted and sees client updates. Only the noisy model releases are
accounted. Private client metrics are deliberately not collected in this mode.
"""

import math
import numpy as np

from flwr.app import MetricRecord
from flwr.serverapp import (
    NeighboringRelation,
    PrivacyConfig,
    RdpAccountant,
    SamplingMethod,
)
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, FedAvg


def validate_privacy(aggregation, framework):
    config = aggregation.get("privacy")
    if config is None:
        return None
    if not isinstance(config, dict) or config.get("mode") != "central_dp":
        raise ValueError("only privacy mode central_dp is supported by this runtime")
    allowed = {"mode", "noise_multiplier", "clipping_norm", "num_sampled_clients",
               "population_size", "target_delta", "max_epsilon"}
    if set(config) - allowed:
        raise ValueError("unknown privacy settings: " + ", ".join(sorted(set(config) - allowed)))
    if aggregation["strategy"] != "fedavg" or framework != "tensorflow":
        raise ValueError("central_dp requires TensorFlow FedAvg floating-point weights")
    for key in ("noise_multiplier", "clipping_norm", "target_delta"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"privacy.{key} must be positive and finite")
    for key in ("num_sampled_clients", "population_size"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"privacy.{key} must be an integer of at least two")
    if config["target_delta"] >= 1 / config["population_size"]:
        raise ValueError("privacy.target_delta must be smaller than 1/population_size")
    if config["num_sampled_clients"] > config["population_size"]:
        raise ValueError("sample size cannot exceed population size")
    if "max_epsilon" in config:
        value = config["max_epsilon"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("privacy.max_epsilon must be positive and finite")
    return config


class PrivateFedAvg(DifferentialPrivacyServerSideFixedClipping):
    def configure_train(self, server_round, arrays, config, grid):
        for array in arrays.values():
            value = array.numpy()
            if value.dtype.kind != "f" or not np.isfinite(value).all():
                raise ValueError("central_dp requires finite floating-point model arrays")
        config["privacy-mode"] = "central_dp"
        messages = list(super().configure_train(server_round, arrays, config, grid))
        self.sampled_node_ids = {message.metadata.dst_node_id for message in messages}
        if len(self.sampled_node_ids) != self.num_sampled_clients:
            raise ValueError("central_dp requires distinct sampled client identities")
        return messages

    def aggregate_train(self, server_round, replies):
        # Do not trust a client to suppress non-private metrics or histories.
        replies = list(replies)
        senders = [reply.metadata.src_node_id for reply in replies]
        if len(set(senders)) != len(senders) or set(senders) != self.sampled_node_ids:
            raise RuntimeError("private round aborted: replies must match distinct sampled clients")
        for reply in replies:
            if not reply.has_error():
                records = reply.content.array_records
                if set(records) != {"arrays"} or list(records["arrays"]) != list(self.current_arrays):
                    raise ValueError("private model reply must match the configured array layout")
                for key, array in records["arrays"].items():
                    value = array.numpy()
                    expected = self.current_arrays[key].numpy()
                    if value.shape != expected.shape or value.dtype != expected.dtype or not np.isfinite(value).all():
                        raise ValueError("private model reply contains mismatched or non-finite weights")
                for key in list(reply.content.config_records):
                    del reply.content[key]
                for key in list(reply.content.metric_records):
                    del reply.content[key]
                reply.content["metrics"] = MetricRecord({"num-examples": 1})
        arrays, _ = super().aggregate_train(server_round, replies)
        if arrays is None:
            raise RuntimeError("private round aborted: all configured clients must reply successfully")
        return arrays, None

    def configure_evaluate(self, server_round, arrays, config, grid):
        return []

    def aggregate_evaluate(self, server_round, replies):
        return None


def make_private_strategy(config, fedavg_kwargs, strategy_type=PrivateFedAvg, **kwargs_for_wrapper):
    kwargs = dict(fedavg_kwargs)
    kwargs.update(
        min_train_nodes=config["num_sampled_clients"],
        fraction_evaluate=0.0,
        train_metrics_aggr_fn=lambda contents, key: MetricRecord(),
    )
    accountant = RdpAccountant(PrivacyConfig(
        target_delta=config["target_delta"],
        population_size=config["population_size"],
        neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE,
        sampling_method=SamplingMethod.NO_AMPLIFICATION,
        max_epsilon=config.get("max_epsilon"),
    ))
    return strategy_type(
        strategy=FedAvg(**kwargs),
        noise_multiplier=config["noise_multiplier"],
        clipping_norm=config["clipping_norm"],
        num_sampled_clients=config["num_sampled_clients"],
        accountant=accountant,
        **kwargs_for_wrapper,
    )
