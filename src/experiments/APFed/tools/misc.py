import kaggle
import zipfile
from pathlib import Path
from ..config import list_datasets

def download_dataset_from_kaggle(cid: int):
    intended_dataset_path = Path(list_datasets[cid])    
    kaggle.api.dataset_download_file(dataset=f"dhoogla/{intended_dataset_path.stem.replace('-', '').lower()}", file_name=f"{intended_dataset_path.name}", path=f"{intended_dataset_path.parent}")
    with zipfile.ZipFile(f"{intended_dataset_path.name}.zip", mode="r") as archive:
        archive.extract(f"{intended_dataset_path.name}", f"{intended_dataset_path.parent}")
    Path(f"{intended_dataset_path.name}.zip").unlink(missing_ok=True)


def download_all_datasets_from_kaggle():
    download_dataset_from_kaggle(cid=0)
    download_dataset_from_kaggle(cid=1)
    download_dataset_from_kaggle(cid=2)
    download_dataset_from_kaggle(cid=3)
    # download_dataset_from_kaggle(cid=4)
