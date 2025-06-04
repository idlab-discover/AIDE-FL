"""flower: A Flower / TensorFlow app."""

import json

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from flower.task import *


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, verbose
    ):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        
        history_str = json.dumps(history.history)
        
        return self.model.get_weights(), len(self.x_train), {"history": history_str}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        metrics = self.model.evaluate(self.x_test, self.y_test, verbose=0, return_dict=True)
        loss = metrics.pop("loss")
        metrics_str = json.dumps(metrics)
        return loss, len(self.x_test), {"metrics": metrics_str}


model = model_config = local_epochs = batch_size = None
def client_fn(context: Context):
    global model, model_config, local_epochs, batch_size
    
    # Reload config if it changed
    config_changed = client_config_changed()
    if config_changed:
        model, model_config, local_epochs, batch_size = load_client_config()

    data = load_data(config_changed)
    epochs = local_epochs
    batch_size = batch_size
    verbose = context.run_config.get("verbose")

    # Return Client instance
    return FlowerClient(
        model, data, epochs, batch_size, verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
