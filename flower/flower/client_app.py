"""flower: A Flower / TensorFlow app."""

import json

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from flower.task import *

model = local_epochs = batch_size = None
x_train = y_train = x_eval = y_eval = x_test = y_test = None

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, verbose
    ):
        self.verbose = verbose

    def fit(self, parameters, config):
        global model, local_epochs, batch_size
        global x_train, y_train, x_eval, y_eval, x_test, y_test
        
        # Reset parameters and datasets if server sends a config, else, use previous
        # Server sending a config denotes the start of a new training run, meaning a (potentially) new scenario has been provided having new parameters and diff dataset
        if config:
            print("Received new parameters from server")
            local_epochs = config["local_epochs"]
            batch_size = config["batch_size"]
            model_data_str = config["model_data"] # Model is sent as json string
            
            model = keras.models.model_from_json(model_data_str)
            
            x_train, y_train, x_eval, y_eval, x_test, y_test = load_data(config["label_column"])
        else: # When {} is received
            print("No new parameters received from server. Using cached parameters")
            
        model.set_weights(parameters)
        
        history = model.fit(
            x=x_train,
            y=y_train,
            epochs=local_epochs,
            batch_size=batch_size,
            validation_data=(x_eval, y_eval),
            verbose=self.verbose,
        )
        
        history_str = json.dumps(history.history)
        
        return model.get_weights(), len(x_train), {"history": history_str}

    def evaluate(self, parameters, config):
        global model, local_epochs, batch_size
        
        model.set_weights(parameters)
        metrics = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
        loss = metrics.pop("loss")
        metrics_str = json.dumps(metrics)
        return loss, len(x_test), {"metrics": metrics_str}


def client_fn(context: Context):
    verbose = context.run_config.get("verbose")

    # Return Client instance
    return FlowerClient(verbose).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
