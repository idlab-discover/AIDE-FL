"""Local protocol experiment using Flower's native SecAgg+ workflow and mod.

This runs real protocol stages with isolated client Context state and in-process
message delivery. It is not a network deployment or a general-purpose MPC engine.
"""

import copy
import math

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, RecordDict
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import parameters_to_ndarrays
from flwr.common.secure_aggregation.secaggplus_constants import RECORD_KEY_CONFIGS, Key, Stage
from flwr.compat.common.recorddict_compat import arrayrecord_to_parameters
from flwr.server import LegacyContext
from flwr.server.client_manager import SimpleClientManager
from flwr.server.compat.grid_client_proxy import GridClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.workflow import SecAggPlusWorkflow

from flower.local_experiment import local_task_identity


class _WeightsClient(NumPyClient):
    def __init__(self, arrays, count):
        self.arrays = arrays
        self.count = count

    def fit(self, parameters, config):
        return self.arrays, self.count, {}


class _LocalProtocolGrid:
    def __init__(self, updates, drop_before_upload):
        self.contexts = {i: Context(1, i, {}, RecordDict(), {}) for i in range(1, len(updates) + 1)}
        self.apps = {}
        self.exchanges = []
        self.drop_before_upload = drop_before_upload
        for i, (arrays, count) in enumerate(updates, 1):
            self.apps[i] = self._app(arrays, count)

    @staticmethod
    def _app(arrays, count):
        def client_fn(context: Context):
            return _WeightsClient(arrays, count).to_client()
        return ClientApp(client_fn=client_fn, mods=[secaggplus_mod])

    def send_and_receive(self, messages, timeout=None):
        replies = []
        for message in messages:
            # Serialization boundaries must not share mutable RecordDict instances.
            message = copy.deepcopy(message)
            node_id = message.metadata.dst_node_id
            stage = message.content.config_records[RECORD_KEY_CONFIGS][Key.STAGE]
            if node_id in self.drop_before_upload and stage in (Stage.COLLECT_MASKED_VECTORS, Stage.UNMASK):
                continue
            reply = self.apps[node_id](message, self.contexts[node_id])
            if reply.has_error():
                raise RuntimeError(reply.error.reason)
            if any(record for record in reply.content.array_records.values()):
                raise RuntimeError("SecAgg client exposed a nonempty model ArrayRecord")
            self.exchanges.append({"node_id": node_id, "stage": stage, "array_values": 0})
            replies.append(copy.deepcopy(reply))
        return replies


def aggregate_secagg(updates, *, clipping_range=8.0, drop_before_upload=()):
    """Aggregate using SecAgg+, optionally dropping one-based node IDs after key sharing."""
    updates = list(updates)
    if len(updates) < 3:
        raise ValueError("the SecAgg+ experiment requires at least three clients")
    if isinstance(clipping_range, bool) or not isinstance(clipping_range, (int, float)) or not math.isfinite(clipping_range) or clipping_range <= 0:
        raise ValueError("clipping_range must be positive and finite")
    dropped = set(drop_before_upload)
    if any(isinstance(node, bool) or not isinstance(node, int) or not 1 <= node <= len(updates) for node in dropped):
        raise ValueError("drop_before_upload must contain valid one-based node IDs")
    if len(updates) - len(dropped) < 3:
        raise ValueError("the experiment requires at least three surviving clients")
    # Each encoded weighted coordinate is below 2**22. Keep the sum below
    # Flower's default 2**32 modulus, including its encoded weight coordinate.
    if len(updates) >= 2**10:
        raise ValueError("too many clients for the SecAgg quantization modulus")
    reference = [np.asarray(array) for array in updates[0][0]]
    if not reference or any(array.size == 0 for array in reference):
        raise ValueError("SecAgg requires nonempty model arrays")
    for arrays, count in updates:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("client sample counts must be positive integers")
        if len(arrays) != len(reference):
            raise ValueError("client model layouts differ")
        for array, expected in zip(arrays, reference, strict=True):
            array = np.asarray(array)
            if array.shape != expected.shape or array.dtype.kind != "f" or not np.isfinite(array).all():
                raise ValueError("SecAgg requires matching finite floating-point weights")
            if np.any(np.abs(array) > clipping_range):
                raise ValueError("weights exceed SecAgg quantization clipping_range")
    grid = _LocalProtocolGrid(updates, dropped)
    manager = SimpleClientManager()
    for node_id in grid.contexts:
        manager.register(GridClientProxy(node_id, grid, 1))
    strategy = FedAvg(fraction_fit=1.0, fraction_evaluate=0.0,
                      min_fit_clients=len(updates), min_available_clients=len(updates))
    context = LegacyContext(Context(1, 0, {}, RecordDict(), {}), strategy=strategy,
                            client_manager=manager)
    context.state["config"] = ConfigRecord({"current_round": 1})
    context.state["parameters"] = ArrayRecord([np.zeros_like(array) for array in reference])
    workflow = SecAggPlusWorkflow(num_shares=len(updates), reconstruction_threshold=2,
                                 max_weight=float(max(count for _, count in updates)),
                                 clipping_range=clipping_range)
    with local_task_identity():
        workflow(grid, context)
    expected = {Stage.SETUP: len(updates), Stage.SHARE_KEYS: len(updates),
                Stage.COLLECT_MASKED_VECTORS: len(updates) - len(dropped),
                Stage.UNMASK: len(updates) - len(dropped)}
    actual = {stage: sum(exchange["stage"] == stage for exchange in grid.exchanges) for stage in Stage.all()}
    if actual != expected:
        raise RuntimeError("SecAgg+ did not complete all four stages")
    arrays = parameters_to_ndarrays(arrayrecord_to_parameters(context.state["parameters"], True))
    return arrays, {"clients": len(updates), "stages": 4, "message_replies": len(grid.exchanges),
                    "surviving_clients": len(updates) - len(dropped),
                    "dropped_before_upload": sorted(dropped), "replies_per_stage": actual,
                    "plaintext_array_values_in_replies": 0,
                    "transport": "in-process native Flower SecAgg+ protocol"}
