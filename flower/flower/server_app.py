"""flower: A Flower / TensorFlow app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from flower.task import *


def server_fn(context: Context):
    model, model_config, fraction_fit, fraction_evaluate, min_available_clients, server_rounds = load_server_config()

    # Read from config
    num_rounds = server_rounds

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(load_model(model, model_config).get_weights())

    # Define strategy
    strategy = strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=min_available_clients,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
