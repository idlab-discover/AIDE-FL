from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
import onnxruntime as ort
from safetensors.torch import load_file, save_file
from fastapi.responses import FileResponse
from typing import Union
import os
import shutil
from datetime import datetime
from contextlib import asynccontextmanager

class ModelMetadata(BaseModel):
    experiment_id: str
    model_type: str  # "onnx" or "safetensor"
    version: int
    created_at: str
    description: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    os.makedirs("model_storage", exist_ok=True)
    yield
    # Shutdown (if needed)

app = FastAPI(lifespan=lifespan, title="Model Server API")

# Register routes
@app.post("/models/upload/{experiment_id}")
async def upload_model(
    experiment_id: str,
    model_file: UploadFile,
    model_type: str,
    description: Optional[str] = None
):
    if model_type not in ["onnx", "safetensor"]:
        raise HTTPException(status_code=400, message="Invalid model type")

    # Create experiment directory if it doesn't exist
    experiment_path = os.path.join("model_storage", experiment_id)
    os.makedirs(experiment_path, exist_ok=True)

    # Determine version number
    version = _get_next_version(experiment_id)
    
    # Save the model file
    file_extension = ".onnx" if model_type == "onnx" else ".safetensors"
    filename = f"model_v{version}{file_extension}"
    file_path = os.path.join(experiment_path, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(model_file.file, buffer)

    # Validate model based on type
    try:
        if model_type == "onnx":
            ort.InferenceSession(file_path)
        else:
            load_file(file_path)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Invalid model file: {str(e)}")

    # Store metadata
    metadata = ModelMetadata(
        experiment_id=experiment_id,
        model_type=model_type,
        version=version,
        created_at=datetime.utcnow().isoformat(),
        description=description
    )
    models_metadata[f"{experiment_id}_v{version}"] = metadata

    return metadata

@app.get("/models/{experiment_id}/latest")
async def get_latest_model(experiment_id: str):
    version = _get_latest_version(experiment_id)
    if version is None:
        raise HTTPException(status_code=404, detail="No models found for this experiment")
    
    metadata = models_metadata.get(f"{experiment_id}_v{version}")
    if not metadata:
        raise HTTPException(status_code=404, detail="Model metadata not found")

    return metadata

@app.get("/models/{experiment_id}/versions")
async def get_model_versions(experiment_id: str) -> List[ModelMetadata]:
    versions = [
        metadata for metadata in models_metadata.values()
        if metadata.experiment_id == experiment_id
    ]
    if not versions:
        raise HTTPException(status_code=404, detail="No models found for this experiment")
    return versions

@app.get("/models/{experiment_id}/{version}")
async def get_model_metadata(experiment_id: str, version: Union[int, str]) -> ModelMetadata:
    # Handle "latest" version request
    if version == "latest":
        version = _get_latest_version(experiment_id)
        if version is None:
            raise HTTPException(status_code=404, detail="No models found for this experiment")
    else:
        version = int(version)

    metadata = models_metadata.get(f"{experiment_id}_v{version}")
    if not metadata:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return metadata

@app.get("/models/{experiment_id}/{version}/download")
async def download_model(experiment_id: str, version: Union[int, str]):
    # Handle "latest" version request
    if version == "latest":
        version = _get_latest_version(experiment_id)
        if version is None:
            raise HTTPException(status_code=404, detail="No models found for this experiment")
    else:
        version = int(version)

    metadata = models_metadata.get(f"{experiment_id}_v{version}")
    if not metadata:
        raise HTTPException(status_code=404, detail="Model not found")

    file_extension = ".onnx" if metadata.model_type == "onnx" else ".safetensors"
    filename = f"model_v{version}{file_extension}"
    file_path = os.path.join("model_storage", experiment_id, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=f"{experiment_id}_model_v{version}{file_extension}"
    )

@app.delete("/models/{experiment_id}/{version}")
async def delete_model(experiment_id: str, version: Union[int, str]):
    # Handle "latest" version request
    if version == "latest":
        version = _get_latest_version(experiment_id)
        if version is None:
            raise HTTPException(status_code=404, detail="No models found for this experiment")
    else:
        version = int(version)

    metadata = models_metadata.get(f"{experiment_id}_v{version}")
    if not metadata:
        raise HTTPException(status_code=404, detail="Model not found")

    # Delete the model file
    file_extension = ".onnx" if metadata.model_type == "onnx" else ".safetensors"
    filename = f"model_v{version}{file_extension}"
    file_path = os.path.join("model_storage", experiment_id, filename)

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        # Remove metadata
        del models_metadata[f"{experiment_id}_v{version}"]
        
        # Remove experiment directory if it's empty
        experiment_dir = os.path.join("model_storage", experiment_id)
        if os.path.exists(experiment_dir) and not os.listdir(experiment_dir):
            os.rmdir(experiment_dir)
        
        return {"message": f"Model version {version} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")

def _get_next_version(experiment_id: str) -> int:
    latest = _get_latest_version(experiment_id)
    return 1 if latest is None else latest + 1

def _get_latest_version(experiment_id: str) -> Optional[int]:
    versions = [
        metadata.version
        for metadata in models_metadata.values()
        if metadata.experiment_id == experiment_id
    ]
    return max(versions) if versions else None

# Initialize global metadata dictionary
models_metadata: Dict[str, ModelMetadata] = {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)