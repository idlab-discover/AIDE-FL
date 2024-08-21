from pathlib import Path
# Setup situation where each client gets diff att from same dataset

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent

TON_PATH = str(Path("/app/NF-ToN-IoT-V2.parquet"))
# NIDS_PATH = str(Path("/app/NF-UQ-NIDS-V2.parquet"))
CIC_PATH = str(Path("/app/NF-CSE-CIC-IDS2018-V2.parquet"))
BOT_PATH = str(Path("/app/NF-BoT-IoT-V2.parquet"))
UNSW_PATH = str(Path("/app/NF-UNSW-NB15-V2.parquet"))

NUM_CLIENTS = 3

list_datasets = [TON_PATH, CIC_PATH, UNSW_PATH, BOT_PATH]

UNSEEN_DATA = BOT_PATH

TARGET=1000000 # equates to 14 000 samples per epoch

MAJORITY_FRAC = 1
SAMPLING_FRAC = 0.5