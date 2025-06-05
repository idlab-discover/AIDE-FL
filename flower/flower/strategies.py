from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays

import tensorflow as tf
import os
import json
import wandb
from datetime import datetime
from collections import defaultdict


class SaveModelFedAvg(FedAvg):
    def __init__(self, scenario_name: str, model: tf.keras.Model, share_dir: str, server_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.share_dir = share_dir
        self.server_rounds = server_rounds
        self.all_metrics = {}
        
        date_name = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        wandb.init(project="aide-flower-deployment", name=f"{scenario_name}-{date_name}")
    
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
        
        # Log to wandb
        client_metrics_list = metrics_aggregated["metrics"]
        metric_sums = defaultdict(float)
        metric_counts = defaultdict(int)
        
        # Log per-client metrics
        for client_id, metrics in enumerate(client_metrics_list):
            for metric_name, metric_value in metrics.items():
                wandb.log({f"client_{client_id}/{metric_name}": metric_value}, step=server_round)
                metric_sums[metric_name] += metric_value
                metric_counts[metric_name] += 1
        
        # Log average per metric
        avg_metrics = {f"avg/{i}": metric_sums[i] / metric_counts[i] for i in metric_sums}
        wandb.log({"loss": loss_aggregated, **avg_metrics}, step=server_round)
        
        if server_round == self.server_rounds:
            wandb.finish()
        
        return loss_aggregated, metrics_aggregated