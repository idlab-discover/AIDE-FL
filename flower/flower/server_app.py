"""flower: A Flower / TensorFlow app."""

import json
import wandb

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, Server
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from flower.task import *
from flower.strategies import SaveModelFedAvg


class CustomServer(Server):
    def fit(self, num_rounds, timeout):
        history, elapsed = super().fit(num_rounds, timeout)
        
        # Notify wandb when server is done fitting
        wandb.finish()
        
        return history, elapsed


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


def make_on_fit_config(model_data: dict, local_epochs: int, batch_size: int):
    
    def on_fit_config(server_round: int) -> Metrics:
        # Only send client config on the first round, else, client will use latest config it received
        if server_round > 1:
            print("Let clients reuse their local config")
            return {} # Would return None but is changed to {} by Flower either way
        
        model_data_str = json.dumps(model_data) # Send model dict as string
        
        print("Sending config to clients")
        return {"model_data": model_data_str, "local_epochs": local_epochs, "batch_size": batch_size}
    
    return on_fit_config


def server_fn(context: Context):
    scenario_name, model_data, fraction_fit, fraction_evaluate, min_available_clients, server_rounds, local_epochs, batch_size = load_server_config()

    # Load model from json
    model = keras.models.model_from_json(json.dumps(model_data))

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(model.get_weights()) # Currently no initial parameters. Could be randomized 

    # Define strategy
    strategy = SaveModelFedAvg(
        scenario_name=scenario_name,
        model=model,
        share_dir="share",
        server_rounds=server_rounds,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=min_available_clients,
        initial_parameters=parameters, # If not defined, the server gets its initial parameters from a random client
        fit_metrics_aggregation_fn=handle_fit_metrics, # Which function to use to handle metrics passed by client's fit function (e.g. history metrics for each epoch)
        evaluate_metrics_aggregation_fn=handle_eval_metrics, # Which function to use to handle metrics passed by client's evaluate function (e.g. accuracy)
        on_fit_config_fn=make_on_fit_config(model_data, local_epochs, batch_size), # Which function to use to relay a config from server to all clients
    )
    config = ServerConfig(num_rounds=server_rounds)

    return ServerAppComponents(server=CustomServer(client_manager=SimpleClientManager(), strategy=strategy), config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
