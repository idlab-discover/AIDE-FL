"""flower: A Flower / TensorFlow app."""

import json

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from flower.task import *

model = local_epochs = batch_size = None
# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, datasets, verbose
    ):
        self.x_train, self.y_train, self.x_eval, self.y_eval, self.x_test, self.y_test = datasets
        self.verbose = verbose

    def fit(self, parameters, config):
        global model, local_epochs, batch_size
        
        # Set parameters if given by server, else, use previous
        if config:
            print("Received new parameters from server")
            local_epochs = config["local_epochs"]
            batch_size = config["batch_size"]
            model_data_str = config["model_data"] # Model is sent as json string
            
            model = keras.models.model_from_json(model_data_str)
        else: # When {} is received
            print("No new parameters received from server. Using cached parameters")
            
        model.set_weights(parameters)
        
        history = model.fit(
            x=self.x_train,
            y=self.y_train,
            epochs=local_epochs,
            batch_size=batch_size,
            validation_data=(self.x_eval, self.y_eval),
            verbose=self.verbose,
        )
        
        history_str = json.dumps(history.history)
        
        return model.get_weights(), len(self.x_train), {"history": history_str}

    def evaluate(self, parameters, config):
        global model, local_epochs, batch_size
        
        model.set_weights(parameters)
        metrics = model.evaluate(self.x_test, self.y_test, verbose=0, return_dict=True)
        loss = metrics.pop("loss")
        metrics_str = json.dumps(metrics)
        return loss, len(self.x_test), {"metrics": metrics_str}


def client_fn(context: Context):
    # Reload config to see whether current round is start of a new training run
    config_changed = client_config_changed()

    datasets = load_data(config_changed)
    verbose = context.run_config.get("verbose")

    # Return Client instance
    return FlowerClient(
        datasets, verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
