from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays

import tensorflow as tf
import os
import json


class SaveModelFedAvg(FedAvg):
    def __init__(self, model: tf.keras.Model, share_dir, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.share_dir = share_dir
        self.all_metrics = {}
    
    def aggregate_fit(self, server_round, results, failures):
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        
        if parameters_aggregated is not None:
            # Set aggregated weights to the model
            weights = parameters_to_ndarrays(parameters_aggregated)
            self.model.set_weights(weights)
            
            # Save model checkpoint
            path = os.path.join(self.share_dir, f"round_{server_round}.weights.h5") # Must end with .weights.h5
            self.model.save_weights(path)
            print(f"Saved checkpoint to {path}")
        
        return parameters_aggregated, metrics_aggregated
    
    
    def aggregate_evaluate(self, server_round, results, failures):
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        metrics = {"loss": loss_aggregated, **metrics_aggregated}
        
        self.all_metrics[server_round] = metrics
        
        # Save metrics as json
        path = os.path.join(self.share_dir, "aggregated_metrics.json")
        with open(path, 'w') as file:
            json.dump(self.all_metrics, file, indent=4)
        print(f"Updated metrics in {path}")
        
        return loss_aggregated, metrics_aggregated