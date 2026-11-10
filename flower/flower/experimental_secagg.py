"""Local protocol experiment using Flower's native SecAgg+ workflow and mod.

This runs real protocol stages with isolated client Context state and in-process
message delivery. It is not a network deployment or a general-purpose MPC engine.
"""

import copy

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, RecordDict
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import parameters_to_ndarrays
from flwr.compat.common.recorddict_compat import arrayrecord_to_parameters
from flwr.server import LegacyContext
from flwr.server.client_manager import SimpleClientManager
from flwr.server.compat.grid_client_proxy import GridClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.workflow import SecAggPlusWorkflow


class _WeightsClient(NumPyClient):
    def __init__(self, arrays, count):
        self.arrays = arrays
        self.count = count

    def fit(self, parameters, config):
        return self.arrays, self.count, {}


class _LocalProtocolGrid:
    def __init__(self, updates):
        self.contexts = {i: Context(1, i, {}, RecordDict(), {}) for i in range(1, len(updates) + 1)}
        self.apps = {}
        self.exchanges = []
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
            reply = self.apps[node_id](message, self.contexts[node_id])
            if reply.has_error():
                raise RuntimeError(reply.error.reason)
            if any(record for record in reply.content.array_records.values()):
                raise RuntimeError("SecAgg client exposed a nonempty model ArrayRecord")
            self.exchanges.append({"node_id": node_id, "array_values": 0})
            replies.append(copy.deepcopy(reply))
        return replies


def aggregate_secagg(updates, *, clipping_range=8.0):
    """Return the aggregate and protocol exchange counts for three or more clients."""
    updates = list(updates)
    if len(updates) < 3:
        raise ValueError("the SecAgg+ experiment requires at least three clients")
    reference = [np.asarray(array) for array in updates[0][0]]
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
    grid = _LocalProtocolGrid(updates)
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
    workflow(grid, context)
    if len(grid.exchanges) != 4 * len(updates):
        raise RuntimeError("SecAgg+ did not complete all four stages")
    arrays = parameters_to_ndarrays(arrayrecord_to_parameters(context.state["parameters"], True))
    return arrays, {"clients": len(updates), "stages": 4, "message_replies": len(grid.exchanges),
                    "plaintext_array_values_in_replies": 0,
                    "transport": "in-process native Flower SecAgg+ protocol"}
