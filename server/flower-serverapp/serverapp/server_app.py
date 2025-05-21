"""flwr-keras: A Flower / TensorFlow app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from serverapp.task import *


def server_fn(context: Context):
    reload_config()

    # Read from config
    num_rounds = gl_server_rounds

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define strategy
    strategy = strategy = FedAvg(
        fraction_fit=gl_fraction_fit,
        fraction_evaluate=gl_fraction_evaluate,
        min_available_clients=gl_min_available_clients,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
