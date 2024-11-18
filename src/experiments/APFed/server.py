"""
APFed Server Implementation using Flower Framework
Handles server initialization, model evaluation, and graceful restarts
"""

import argparse
import os
import glob
import gc
import time
import signal
from pathlib import Path
from typing import Dict, List, Optional
import flwr as fl
from flwr.common.logger import log
from logging import INFO, DEBUG
import tensorflow as tf

from ...aide_fl.experiments.apfed.config import UNSEEN_DATA, list_datasets
from .tools.preprocess import preprocess_queensland_data
from .tools.misc import download_all_datasets_from_kaggle
from .custom_strategies.save_model_strategy import SaveModelStrategy
from .model.model import get_model

class APFedServer:
    def __init__(
        self,
        server_address: str = "0.0.0.0:8080",
        rounds: int = 5,
        min_num_clients: int = 3,
        learning_rate: float = 0.0001,
        batch_size: int = 128,
        local_epochs: int = 1,
    ):
        self.server_address = server_address
        self.rounds = rounds
        self.min_num_clients = min_num_clients
        self.running = True
        self.strategy = SaveModelStrategy(
            learning_rate=learning_rate,
            batch_size=batch_size,
            local_epochs=local_epochs,
            momentum=0.9,
            s=0.2,
            min_fit_clients=min_num_clients,
            evaluate_metrics_aggregation_fn=self.average_metrics,
        )
        self.strategy.on_fit_config_fn = self.strategy.fit_config
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        """Handle graceful shutdown on SIGTERM/SIGINT"""
        log(INFO, "Received shutdown signal, cleaning up...")
        self.running = False

    def average_metrics(self, metrics: List[Dict]) -> Dict:
        """Average metrics across all clients"""
        if not metrics:
            return {}
            
        metric_keys = ["accuracy", "recall", "precision", "AUROC_SCORE"]
        averaged = {key: sum(m[key] for _, m in metrics) / len(metrics) 
                   for key in metric_keys}
        
        # Log metrics to file
        with Path("avg_scores.txt").open("a") as f:
            f.write(f"{averaged}\n\n")
        
        return averaged

    def evaluate_global_model(self, model: tf.keras.Model, dataset_list: List[str], 
                            output_file: str) -> None:
        """Evaluate the global model on all datasets"""
        try:
            unseen_filename = Path(UNSEEN_DATA).stem
            with open(output_file, 'a') as f:
                for dataset_path in dataset_list:
                    try:
                        log(DEBUG, f"Evaluating {dataset_path}")
                        _, _, X_y_test_set = preprocess_queensland_data(dataset_path, True)          
                        
                        metrics = model.evaluate(X_y_test_set[0], X_y_test_set[1], 
                                              verbose=0)
                        
                        filename = Path(dataset_path).stem
                        for metric_name, value in zip(model.metrics_names, metrics):
                            f.write(f"{filename} {metric_name}: {value} -> unseen {unseen_filename}\n")
                            log(DEBUG, f"Test {metric_name}: {value}")
                            
                    except Exception as e:
                        log(DEBUG, f"Error evaluating {dataset_path}: {e}")
                        f.write(f"Error evaluating {dataset_path}: {e}\n")

        except IOError as e:
            log(DEBUG, f"Error writing to {output_file}: {e}")

    def cleanup_checkpoints(self) -> None:
        """Clean up model checkpoints"""
        checkpoint_dir = Path(__file__).parent / "custom_strategies" / "model_checkpoints"
        try:
            for checkpoint in checkpoint_dir.glob("*"):
                checkpoint.unlink()
                log(DEBUG, f"Removed checkpoint: {checkpoint}")
        except Exception as e:
            log(DEBUG, f"Error cleaning checkpoints: {e}")

    def run(self) -> None:
        """Main server loop with graceful restart capability"""
        while self.running:
            try:
                log(INFO, f"Starting Flower server on {self.server_address}")
                
                # Start the Flower server
                fl.server.start_server(
                    server_address=self.server_address,
                    config=fl.server.ServerConfig(num_rounds=self.rounds),
                    strategy=self.strategy,
                )

                # Download datasets if needed
                log(INFO, "Downloading datasets from Kaggle")
                download_all_datasets_from_kaggle()

                # Evaluate final model
                checkpoint_dir = Path(__file__).parent / "custom_strategies" / "model_checkpoints"
                latest_checkpoint = max(checkpoint_dir.glob("*"), 
                                     key=os.path.getctime, default=None)
                
                if latest_checkpoint:
                    log(DEBUG, f"Loading model from: {latest_checkpoint}")
                    model = tf.keras.models.load_model(latest_checkpoint, safe_mode=False)
                    self.evaluate_global_model(model, list_datasets, "scores.txt")
                
                self.cleanup_checkpoints()
                
                if not self.running:
                    break
                    
                log(INFO, "Server round complete, waiting before restart...")
                time.sleep(30)  # Configurable delay between restarts
                
            except Exception as e:
                log(DEBUG, f"Server error: {e}")
                if self.running:
                    time.sleep(5)  # Brief delay before retry
                    continue
                break

def main() -> None:
    parser = argparse.ArgumentParser(description="APFed Flower Server")
    parser.add_argument("--server", type=str, default="0.0.0.0:8080",
                       help="gRPC server address")
    parser.add_argument("--rounds", type=int, default=5,
                       help="Number of rounds of federated learning")
    parser.add_argument("--min_num_clients", type=int, default=3,
                       help="Minimum number of available clients required")
    
    args = parser.parse_args()
    
    server = APFedServer(
        server_address=args.server,
        rounds=args.rounds,
        min_num_clients=args.min_num_clients
    )
    server.run()

if __name__ == "__main__":
    main()
