from fastapi import FastAPI
import pandas as pd
from minio import Minio
from io import BytesIO
import kaggle
from pathlib import Path
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        if not minio_client.bucket_exists("kaggle-datasets"):
            minio_client.make_bucket("kaggle-datasets")
    except Exception as e:
        print(f"Error initializing MinIO bucket: {str(e)}")
    yield
    # Shutdown (if needed)

app = FastAPI(lifespan=lifespan)

# Initialize MinIO client
minio_client = Minio(
    "minio-server:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)



@app.post("/cache/kaggle/{dataset_name}")
async def cache_kaggle_dataset(dataset_name: str):
    """
    Downloads dataset from Kaggle and caches it in MinIO
    dataset_name should be in format 'kaggle_username/datasetname'
    """
    
    # Create temp directory for download
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Download from Kaggle
        kaggle.api.dataset_download_files(
            dataset=dataset_name,
            path=temp_dir,
            unzip=True
        )
        
        # Process all CSV and Parquet files in the temp directory
        for file_path in temp_dir.glob('*'):
            if file_path.suffix.lower() in ['.csv', '.parquet']:
                # Read file into pandas based on extension
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                else:  # .parquet
                    df = pd.read_parquet(file_path)
                
                # Convert to parquet in memory
                parquet_buffer = BytesIO()
                df.to_parquet(parquet_buffer)
                parquet_buffer.seek(0)
                
                # Store in MinIO
                minio_client.put_object(
                    "kaggle-datasets",
                    f"{file_path.stem}.parquet",
                    parquet_buffer,
                    length=parquet_buffer.getbuffer().nbytes
                )
            
        return {"message": f"Successfully cached dataset {dataset_name}"}
        
    finally:
        # Cleanup temp files
        for file in temp_dir.glob('*'):
            file.unlink()
        temp_dir.rmdir()

@app.get("/dataset/{dataset_name}")
async def get_dataset(dataset_name: str):
    """
    Retrieves a cached dataset from MinIO
    dataset_name should be the name without extension
    """
    try:
        data = minio_client.get_object("kaggle-datasets", f"{dataset_name}.parquet")
        df = pd.read_parquet(BytesIO(data.read()))
        return df.to_dict()
    except Exception as e:
        return {"error": f"Dataset not found: {str(e)}"}