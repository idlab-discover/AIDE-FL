"""flower: A Flower / TensorFlow app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from flower.task import *


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics: # One Metrics object per client
    """A function that aggregates metrics"""
    
    accuracies = [n_tests * m["accuracy"] for n_tests, m in metrics]
    total_tests = sum(n_tests for n_tests, _ in metrics)
    
    return {"accuracy": sum(accuracies) / total_tests} # Returns weighted accuracy of all accuracies calculated per client on the aggregated model with their proper evaluate dataset


def server_fn(context: Context):
    model, fraction_fit, fraction_evaluate, min_available_clients, server_rounds = load_server_config()

    # Read from config
    num_rounds = server_rounds

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(model.get_weights()) # Currently no initial parameters. Could be randomized 

    # Define strategy
    strategy = strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=min_available_clients,
        initial_parameters=parameters, # If not defined, the server gets its initial parameters from a random client
        evaluate_metrics_aggregation_fn=weighted_average, # Defines logic used to handle metrics passed by client's evaluate function (e.g. accuracy)
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
