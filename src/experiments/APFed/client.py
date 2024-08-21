import argparse
import flwr as fl
import tensorflow as tf
from tensorflow import keras as keras
from flwr.common.logger import log
from logging import INFO
from .tools.preprocess import preprocess_queensland_data
from .tools.misc import download_dataset_from_kaggle
from .model.model import get_model
from .config import list_datasets, NUM_CLIENTS
import tensorflow.keras.metrics as metrics
from tensorflow.keras.optimizers import Adam

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--cid",
    type=int,
    help=f"Client identifier",
)
parser.add_argument(
    "--server",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (default '0.0.0.0:8080')",
)

def prepare_dataset(cid : int):
    X_y_train_set, X_y_val_set, _ = preprocess_queensland_data(list_datasets[cid])    
    return X_y_train_set, X_y_val_set

class FlowerClient(fl.client.NumPyClient):
    
    def __init__(self, trainset, valset, cid):
        self.x_train, self.y_train = trainset
        self.x_val, self.y_val = valset
        self.cid = cid        
        self.model = get_model() 
            
    def adaptive_personalized_update(self, model, global_params, x_train, y_train, batch_size, learning_rate, s, epochs=None):
        # Sample a random subset of the data
        subset_size = int(len(x_train) * s)
        indices = tf.random.shuffle(tf.range(len(x_train)))
        x_train_subset = tf.gather(x_train, indices[:subset_size])
        y_train_subset = tf.gather(y_train, indices[:subset_size])

        # Convert to TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((x_train_subset, y_train_subset))
        dataset = dataset.batch(batch_size)

        # Set initial weights and compile model
        model.set_weights(global_params)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate), metrics=[
            'accuracy',
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
            metrics.AUC(curve='ROC', name='auc_roc')
        ])
        
        # Train model
        if epochs:  # If epochs is specified, perform one epoch for rounds greater than 2
            model.fit(dataset, epochs=1)
        else:  # For round 2, train until convergence
            # Here you might want to implement a convergence check and/or a maximum number of iterations
            for batch_x, batch_y in dataset:
                model.train_on_batch(batch_x, batch_y)

        return model.get_weights()
    
    def train_local_model(self, model, x_train, y_train, batch_size, epochs, learning_rate, momentum, loss_object):
        # Initialize the first moment (velocity)
        v = [tf.zeros_like(w) for w in model.weights]

        # Cache the dataset and prefetch batches for efficiency
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)

        # Training loop
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs}")
            for step, (x_batch, y_batch) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    predictions = model(x_batch, training=True)
                    loss = loss_object(y_batch, predictions)
                grads = tape.gradient(loss, model.trainable_variables)

                for i, grad in enumerate(grads):
                    v[i] = momentum * v[i] + grad
                    model.weights[i].assign_sub(learning_rate * v[i])
                
                if step % 100 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.numpy()}")

        return model.get_weights()

    def get_parameters(self, config=None):
        print("get_parameters")
        return self.model.get_weights()

    def set_parameters(self, params):
        print("set_parameters")
        self.model.set_weights(params)
        
    def fit(self, parameters, config):
        print("Start training")
        # Retrieve the relevant configurations
        batch, epochs, current_round, learning_rate, momentum = (
                    config["batch_size"],
                    config["local_epochs"],
                    config["server_round"],
                    config["learning_rate"],
                    config["momentum"]
                )
        
        print(f"We are in round {current_round}!")      
        
        # Prepare dataset splits
        train_set, val_set = prepare_dataset(self.cid)
        self.x_train, self.y_train = train_set
        self.x_val, self.y_val = val_set
        
        # Update local model with global parameters received from the server
        self.model.set_weights(parameters)

        # Adaptive Personalized Update (Algorithm 2) performed if t > 1
        if current_round > 1:
            print("Execute Algorithm 2")
            self.model.set_weights(
                self.adaptive_personalized_update(
                    self.model,
                    parameters,
                    self.x_train,
                    self.y_train,
                    batch_size=batch,
                    learning_rate=config["learning_rate"],
                    s=config["s"],
                    epochs=epochs if current_round > 2 else None
                )
            )

        # Local model training (Algorithm 3)
        print("Execute Algorithm 3")
        loss_object = tf.keras.losses.BinaryCrossentropy()
        self.model.set_weights(
            self.train_local_model(
                self.model,
                self.x_train,
                self.y_train,
                batch_size=batch,
                epochs=epochs,
                learning_rate=learning_rate,
                momentum=momentum,
                loss_object=loss_object
            )
        )
        
        updated_parameters = self.get_parameters()
        
        return updated_parameters, len(self.x_train), {}

    def evaluate(self, parameters, config):
        print("Start evaluation")
        self.set_parameters(parameters)        
        loss, acc, precision, recall, auroc = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        return loss, len(self.x_val), {"accuracy" : acc, "precision": precision, "recall" : recall, "AUROC_SCORE" : auroc}

def main():
    args = parser.parse_args()
    log(INFO, "Starting Flower Client using: %s", args)
    
    assert args.cid < NUM_CLIENTS
    
    log(INFO, "Retrieving dataset from Kaggle")
    download_dataset_from_kaggle(args.cid)
    log(INFO, "Dataset retrieved")

    log(INFO, "Starting dataset preprocessing")
    train_set, val_set = prepare_dataset(args.cid)
    log(INFO, "Finished dataset preprocessing")

    fl.client.start_numpy_client(
        server_address=args.server,
        client=FlowerClient(trainset=train_set, valset=val_set, cid=args.cid),
    )

if __name__ == "__main__":
    main()
    
    