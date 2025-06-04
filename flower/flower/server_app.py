"""flower: A Flower / TensorFlow app."""

import json

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from flower.task import *
from flower.strategies import SaveModelFedAvg


def handle_fit_metrics(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Handle metrics from fit method in clients"""
    
    histories = []
    for _, m in metrics:
        history_str = m["history"]
        history = json.loads(history_str)
        histories.append(history)
        
    return {"histories": histories}


def handle_eval_metrics(metrics: list[tuple[int, Metrics]]) -> Metrics: # One Metrics object per client
    """Handle metrics from evaluate method in clients"""
    
    eval_metrics = []
    for _, m in metrics:
        metrics_str = m["metrics"]
        metrics_json = json.loads(metrics_str)
        eval_metrics.append(metrics_json)
        
    return {"metrics": eval_metrics}


def server_fn(context: Context):
    model, fraction_fit, fraction_evaluate, min_available_clients, server_rounds = load_server_config()

    # Read from config
    num_rounds = server_rounds

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(model.get_weights()) # Currently no initial parameters. Could be randomized 

    # Define strategy
    strategy = strategy = SaveModelFedAvg(
        model=model,
        share_dir="share",
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=min_available_clients,
        initial_parameters=parameters, # If not defined, the server gets its initial parameters from a random client
        fit_metrics_aggregation_fn=handle_fit_metrics,
        evaluate_metrics_aggregation_fn=handle_eval_metrics, # Defines logic used to handle metrics passed by client's evaluate function (e.g. accuracy)
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
