from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays

import tensorflow as tf
import os
import json
import wandb
import numpy as np
from safetensors.numpy import save_file
from datetime import datetime
from collections import defaultdict


class SaveModelFedAvg(FedAvg):
    def __init__(self, scenario_name: str, model: tf.keras.Model, share_dir: str, server_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.scenario_name = scenario_name
        self.model = model
        self.share_dir = share_dir
        self.server_rounds = server_rounds
        self.fitted_validated_metrics = {}
        self.distributed_metrics = {}
        self.centralised_metrics = {}
        
        date_name = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        wandb.init(project="aide-flower-deployment", name=f"{scenario_name}-{date_name}")
    
    def aggregate_fit(self, server_round, results, failures):
        """Add functionality to default FedAvg aggregate_fit function that saves checkpoints as h5 files and the final model as safetensors."""
        
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        
        if parameters_aggregated is None:
            return parameters_aggregated, metrics_aggregated
        
        self.fitted_validated_metrics[server_round] = metrics_aggregated.get("histories")
        
        # Save metrics as json
        path = os.path.join(self.share_dir, f"{self.scenario_name}_fitted_validated_metrics.json")
        with open(path, 'w') as file:
            json.dump(self.fitted_validated_metrics, file, indent=4)
        print(f"Updated fitted_validated metrics in {path}")
        
        # Set aggregated weights to the model
        weights = parameters_to_ndarrays(parameters_aggregated)
        self.model.set_weights(weights)
        
        # Save model checkpoint
        if server_round == self.server_rounds:
            
            # Safe last aggregated model in safetensors
            weights_dict = {
                f"weight_{i}": w.astype(np.float32) if not np.issubdtype(w.dtype, np.floating) else w for i, w in enumerate(weights)
            }
            
            st_path = os.path.join(self.share_dir, f"{self.scenario_name}.safetensors")
            save_file(weights_dict, st_path)
            print(f"Saved checkpoint to {st_path}")
        
        path = os.path.join(self.share_dir, f"{self.scenario_name}_round_{server_round}.weights.h5") # Must end with .weights.h5
        self.model.save_weights(path)
        print(f"Saved checkpoint to {path}")
        
        return parameters_aggregated, metrics_aggregated
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Add functionality to default FedAvg aggregate_evaluate function that saves distributed evaluation metrics to json and logs them to wandb."""

        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        metrics = {"avg_loss": loss_aggregated, **metrics_aggregated}
        
        self.distributed_metrics[server_round] = metrics
        
        # Save metrics as json
        path = os.path.join(self.share_dir, f"{self.scenario_name}_distributed_metrics.json")
        with open(path, 'w') as file:
            json.dump(self.distributed_metrics, file, indent=4)
        print(f"Updated distributed metrics in {path}")
        
        # There are no "metrics" if all clients fail. Then nothing to log to wandb
        if not metrics_aggregated.get("distributed_metrics"): # get("metrics") returns None if not present instead of KeyError in case of ["metrics"]
            return loss_aggregated, metrics_aggregated
        
        # Log to wandb
        client_metrics_list = metrics_aggregated["distributed_metrics"]
        metric_sums = defaultdict(float)
        metric_counts = defaultdict(int)
        
        # Log per-client metrics
        for client_id, metrics in enumerate(client_metrics_list):
            for metric_name, metric_value in metrics.items():
                wandb.log({f"client_{client_id}/{metric_name}": metric_value}, step=server_round)
                metric_sums[metric_name] += metric_value
                metric_counts[metric_name] += 1
        
        # Log average per metric
        avg_metrics = {f"distributed/avg_{i}": metric_sums[i] / metric_counts[i] for i in metric_sums}
        wandb.log({"distributed/loss": loss_aggregated, **avg_metrics}, step=server_round)
        
        return loss_aggregated, metrics_aggregated
    
    def evaluate(self, server_round, parameters):
        """Add functionality to default FedAvg evaluate function that saves centralised evaluation metrics to json and logs them to wandb."""

        eval_res = super().evaluate(server_round, parameters)
        
        if eval_res is None:
            return None
        
        loss, str_metrics = eval_res
        dict_metrics = json.loads(str_metrics.get("centralised_metrics")) # Flower does not support setting a dict type as value so had to be sent as a string

        metrics_combined = {"loss": loss, **dict_metrics}
        self.centralised_metrics[server_round] = metrics_combined
        
        # Save centralised metrics as json
        path = os.path.join(self.share_dir, f"{self.scenario_name}_centralised_metrics.json")
        with open(path, 'w') as file:
            json.dump(self.centralised_metrics, file, indent=4)
        print(f"Updated centralised metrics in {path}")
        
        # There are no "metrics" if all clients fail. Then nothing to log to wandb
        if not str_metrics.get("centralised_metrics"):
            return loss, str_metrics
        
        # Log to wandb
        combined = {f"centralised/{i}": dict_metrics[i] for i in dict_metrics}
        wandb.log({"centralised/loss": loss, **combined}, step=server_round)
        
        return loss, str_metrics