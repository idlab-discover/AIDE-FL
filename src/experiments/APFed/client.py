"""
APFed Client Implementation using Flower Framework
Handles client training, evaluation, and graceful restarts
"""

import argparse
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import tensorflow as tf
import flwr as fl
from flwr.common.logger import log
from logging import INFO, DEBUG

from ...aide_fl.experiments.apfed.config import list_datasets, NUM_CLIENTS
from .tools.preprocess import preprocess_queensland_data
from .tools.misc import download_dataset_from_kaggle
from .model.model import get_model

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    batch_size: int
    local_epochs: int
    learning_rate: float
    momentum: float
    s: float  # Subset fraction for adaptive personalization

class APFedClient(fl.client.NumPyClient):
    """Flower client implementing APFed algorithm"""
    
    def __init__(self, cid: int, trainset: Tuple, valset: Tuple):
        if not 0 <= cid < NUM_CLIENTS:
            raise ValueError(f"Client ID must be between 0 and {NUM_CLIENTS-1}")
            
        self.cid = cid
        self.x_train, self.y_train = trainset
        self.x_val, self.y_val = valset
        self.model = get_model()
        
    def adaptive_personalized_update(
        self,
        model: tf.keras.Model,
        global_params: List[tf.Tensor],
        x_train: tf.Tensor,
        y_train: tf.Tensor,
        config: TrainingConfig,
        epochs: Optional[int] = None
    ) -> List[tf.Tensor]:
        """Implement Algorithm 2: Adaptive Personalized Update"""
        # Sample random subset
        subset_size = int(len(x_train) * config.s)
        indices = tf.random.shuffle(tf.range(len(x_train)))
        x_train_subset = tf.gather(x_train, indices[:subset_size])
        y_train_subset = tf.gather(y_train, indices[:subset_size])

        # Prepare dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            (x_train_subset, y_train_subset)
        ).batch(config.batch_size).cache().prefetch(tf.data.AUTOTUNE)

        # Set weights and compile
        model.set_weights(global_params)
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(config.learning_rate),
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(curve='ROC', name='auc_roc')]
        )
        
        if epochs:
            model.fit(dataset, epochs=1, verbose=0)
        else:
            for batch_x, batch_y in dataset:
                model.train_on_batch(batch_x, batch_y)

        return model.get_weights()
    
    def train_local_model(
        self,
        model: tf.keras.Model,
        x_train: tf.Tensor,
        y_train: tf.Tensor,
        config: TrainingConfig
    ) -> List[tf.Tensor]:
        """Implement Algorithm 3: Local Model Training"""
        # Initialize momentum
        v = [tf.zeros_like(w) for w in model.weights]

        # Prepare dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)
        ).batch(config.batch_size).cache().prefetch(tf.data.AUTOTUNE)

        loss_object = tf.keras.losses.BinaryCrossentropy()

        for epoch in range(config.local_epochs):
            log(DEBUG, f"Starting epoch {epoch+1}/{config.local_epochs}")
            for step, (x_batch, y_batch) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    predictions = model(x_batch, training=True)
                    loss = loss_object(y_batch, predictions)
                
                grads = tape.gradient(loss, model.trainable_variables)
                
                # Update with momentum
                for i, grad in enumerate(grads):
                    v[i] = config.momentum * v[i] + grad
                    model.weights[i].assign_sub(config.learning_rate * v[i])
                
                if step % 100 == 0:
                    log(DEBUG, f"Epoch {epoch+1}, Step {step}, Loss: {loss.numpy():.4f}")

        return model.get_weights()

    def get_parameters(self, config: Dict = None) -> List[tf.Tensor]:
        """Get model parameters"""
        return self.model.get_weights()

    def set_parameters(self, params: List[tf.Tensor]) -> None:
        """Set model parameters"""
        self.model.set_weights(params)
        
    def fit(
        self,
        parameters: List[tf.Tensor],
        config: Dict
    ) -> Tuple[List[tf.Tensor], int, Dict]:
        """Train model with local dataset"""
        log(INFO, "Starting training round")
        
        # Extract config
        training_config = TrainingConfig(
            batch_size=config["batch_size"],
            local_epochs=config["local_epochs"],
            learning_rate=config["learning_rate"],
            momentum=config["momentum"],
            s=config["s"]
        )
        current_round = config["server_round"]
        
        log(INFO, f"Training round {current_round}")
        
        # Update model with global parameters
        self.set_parameters(parameters)

        # Adaptive Personalized Update (Algorithm 2) for rounds > 1
        if current_round > 1:
            log(INFO, "Executing Algorithm 2: Adaptive Personalized Update")
            self.model.set_weights(
                self.adaptive_personalized_update(
                    self.model,
                    parameters,
                    self.x_train,
                    self.y_train,
                    training_config,
                    epochs=training_config.local_epochs if current_round > 2 else None
                )
            )

        # Local model training (Algorithm 3)
        log(INFO, "Executing Algorithm 3: Local Model Training")
        self.model.set_weights(
            self.train_local_model(
                self.model,
                self.x_train,
                self.y_train,
                training_config
            )
        )
        
        return self.get_parameters(), len(self.x_train), {}

    def evaluate(
        self,
        parameters: List[tf.Tensor],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate model with local validation dataset"""
        log(INFO, "Starting evaluation")
        self.set_parameters(parameters)
        
        loss, acc, precision, recall, auroc = self.model.evaluate(
            self.x_val,
            self.y_val,
            verbose=0
        )
        
        return loss, len(self.x_val), {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "AUROC_SCORE": auroc
        }

class APFedClientManager:
    """Manages APFed client lifecycle and restarts"""
    
    def __init__(
        self,
        cid: int,
        server_address: str = "0.0.0.0:8080",
        restart_delay: int = 30
    ):
        self.cid = cid
        self.server_address = server_address
        self.restart_delay = restart_delay
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)
    
    def handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        log(INFO, "Received shutdown signal, cleaning up...")
        self.running = False
    
    def prepare_dataset(self) -> Tuple[Tuple, Tuple]:
        """Prepare training and validation datasets"""
        return preprocess_queensland_data(list_datasets[self.cid])
    
    def run(self) -> None:
        """Main client loop with restart capability"""
        while self.running:
            try:
                log(INFO, f"Starting client {self.cid}")
                
                # Download dataset if needed
                log(INFO, "Retrieving dataset from Kaggle")
                download_dataset_from_kaggle(self.cid)
                
                # Prepare datasets
                log(INFO, "Preprocessing dataset")
                train_set, val_set, _ = self.prepare_dataset()
                
                # Start Flower client
                fl.client.start_numpy_client(
                    server_address=self.server_address,
                    client=APFedClient(
                        cid=self.cid,
                        trainset=train_set,
                        valset=val_set
                    )
                )
                
                if not self.running:
                    break
                    
                log(INFO, "Client disconnected, waiting before restart...")
                time.sleep(self.restart_delay)
                
            except Exception as e:
                log(DEBUG, f"Client error: {e}")
                if self.running:
                    time.sleep(5)  # Brief delay before retry
                    continue
                break

def main() -> None:
    """Entry point for the client"""
    parser = argparse.ArgumentParser(description="APFed Flower Client")
    parser.add_argument("--cid", type=int, required=True,
                       help="Client ID in range [0, 3]")
    parser.add_argument("--server", type=str, default="0.0.0.0:8080",
                       help="gRPC server address")
    parser.add_argument("--restart-delay", type=int, default=30,
                       help="Delay in seconds before restart attempts")
    
    args = parser.parse_args()
    
    client_manager = APFedClientManager(
        cid=args.cid,
        server_address=args.server,
        restart_delay=args.restart_delay
    )
    client_manager.run()

if __name__ == "__main__":
    main()
    
    