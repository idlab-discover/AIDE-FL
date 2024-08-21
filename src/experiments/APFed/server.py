import argparse
import os
import glob
import gc
from .config import UNSEEN_DATA, list_datasets
import flwr as fl
from .tools.preprocess import preprocess_queensland_data
from .tools.misc import download_all_datasets_from_kaggle
from .custom_strategies.save_model_strategy import SaveModelStrategy
import tensorflow as tf
from tensorflow import keras as keras
from flwr.common.logger import log
from logging import INFO, DEBUG
from .model.model import get_model

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server",
    type=str,
    default="0.0.0.0:8080",
    help="gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=5,
    help="Number of rounds of federated learning (default: 5)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=3,
    help="Minimum number of available clients required for sampling (default: 4)",
)

def evaluate_global_model(model, dataset_list, file):
    try:
        unseen_filename = os.path.splitext(os.path.basename(UNSEEN_DATA))
        with open(file, 'a') as file:
            for dataset_path in dataset_list:
                try:
                    log(DEBUG, f"Evaluating {dataset_path}")
                    _, _, X_y_test_set = preprocess_queensland_data(dataset_path, True)          
                    
                    loss, acc, precision, recall, auroc = model.evaluate(X_y_test_set[0], X_y_test_set[1], verbose=0)                    
                    
                    filename, _ = os.path.splitext(os.path.basename(dataset_path))
                    file.write(f"{filename}: {precision} -> unseen {unseen_filename}\n")
                    file.write(f"{filename}: {recall} -> unseen {unseen_filename}\n")
                    file.write(f"{filename}: {auroc} -> unseen {unseen_filename}\n")
                    
                    log(DEBUG, f"Test Loss: {loss}")
                    log(DEBUG, f"Test Accuracy: {acc}")
                    log(DEBUG, f"Test Precision: {precision}")
                    log(DEBUG, f"Test Recall: {recall}")
                    log(DEBUG, f"Test AUROC: {auroc}")
                    
                except Exception as eval_exception:
                    log(DEBUG, f"Error during evaluation of dataset {dataset_path}: {eval_exception}")
                    file.write(f"Error evaluating {dataset_path}: {eval_exception}\n")

    except IOError as io_error:
        log(DEBUG, f"Error opening or writing to the AUROC scores file {file}: {io_error}")
    
def average_metrics(metrics):
    accuracies = [metric["accuracy"] for _, metric in metrics]
    recalls = [metric["recall"] for _, metric in metrics]
    precisions = [metric["precision"] for _, metric in metrics]
    aucs = [metric["AUROC_SCORE"] for _, metric in metrics]
    
    accuracies = sum(accuracies) / len(accuracies)
    recalls = sum(recalls) / len(recalls)
    precisions = sum(precisions) / len(precisions)
    aucs = sum(aucs) / len(aucs)

    avg_scores = {"acc": accuracies, "rec": recalls, "prec": precisions, "auc": aucs}
    
    with open("avg_scores.txt", "a") as file:
        file.write(f"{avg_scores}\n\n")

    return avg_scores
    
def get_initial_parameters():
    model = get_model()
    return model.get_weights()

def main() -> None:
    args = parser.parse_args()
    
    log(INFO, "Starting Flower Server using: %s", args)
        
    strategy = SaveModelStrategy(
        learning_rate=0.0001,
        batch_size=128,
        local_epochs=1,
        momentum=0.9,
        s=0.2,
        min_fit_clients=args.min_num_clients,
        evaluate_metrics_aggregation_fn=average_metrics,
    )
    
    strategy.on_fit_config_fn = strategy.fit_config
    
    fl.server.start_server(
        server_address=args.server,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    log(INFO, "Downloading all datasets from Kaggle")
    download_all_datasets_from_kaggle()
    log(INFO, "Downloaded all datasets from Kaggle")
    
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoint_dir = os.path.join(current_file_dir, "custom_strategies", "model_checkpoints")
  
    list_of_files = glob.glob(os.path.join(checkpoint_dir, '*'))
    if list_of_files:
        latest_round_file = max(list_of_files, key=os.path.getctime)
        log(DEBUG, f"Loading pre-trained model from: {latest_round_file}")

        model = tf.keras.models.load_model(latest_round_file, safe_mode=False)
            
        auroc_score_file = "scores.txt"    
        evaluate_global_model(model, list_datasets, auroc_score_file)
        
        files = glob.glob(os.path.join(checkpoint_dir, '*'))

        for f in files:
            try:
                os.remove(f)
                print(f"Removed: {f}")
            except Exception as e:
                print(f"Failed to remove {f}: {e}")
        else:
            log(DEBUG, "No files in Modelcheckpoints found!")
            
if __name__ == "__main__":
    main()
