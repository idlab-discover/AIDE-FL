import flwr as fl
from typing import List, Tuple, Optional, Dict, Union, Any
from flwr.common import Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
import numpy as np
from ..model.model import get_model
import os
from typing import List, Tuple, Optional, Dict, Union

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, num_rounds=5, learning_rate=0.001, batch_size=16, local_epochs=1, momentum=0.9, s=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.momentum = momentum
        self.s = s
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            model = get_model()
            model.set_weights(aggregated_ndarrays)
                     
            current_file_dir = os.path.dirname(os.path.realpath(__file__))

            checkpoint_dir = os.path.join(current_file_dir, "model_checkpoints")

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            model_save_path = os.path.join(checkpoint_dir, f"model_round_{server_round}.keras")

            try:
                model.save(model_save_path)
                print(f"Model successfully saved to {model_save_path}")
            except Exception as e:
                print(f"Failed to save the model: {e}")

        return aggregated_parameters, aggregated_metrics

    def fit_config(self, server_round: int) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "local_epochs": self.local_epochs,
            "momentum": self.momentum,
            "s": self.s,
            "server_round": server_round
        }