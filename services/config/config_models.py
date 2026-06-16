import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import dotenv_values

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
dotenv_path = os.path.join(src_dir, "config", ".env")

env_variables = dotenv_values(dotenv_path)

for k, v in env_variables.items():
    if v is not None:
        os.environ[k] = v

def get_development_settings(key, default_value):
    if "APP_ENVIRONMENT" not in os.environ:
        print("APP_ENVIRONMENT must be set.")
        exit(1)

    if os.environ["APP_ENVIRONMENT"] != "production":
        value = os.environ.get(key)
        match value:
            case 'true' | 'True' | 'TRUE':
                return True
            case 'false' | 'False' | 'FALSE':
                return False
            case _:
                return value
    
    else:
        return default_value
    
class AppConfig(BaseModel):
    name: str 
    version: str
    environment: str = get_development_settings("APP_ENVIRONMENT", "production")
    debug: bool = get_development_settings("DEBUG", False)
    
class DataConfig(BaseModel):
    data_folder: str = get_development_settings("PATHS_DATASET_FOLDER", "datasets")

class ModelConfig(BaseModel):
    name: str = get_development_settings("MODEL_NAME", "dmis-lab/biobert-base-cased-v1.1")
    max_length: int = get_development_settings("MODEL_MAX_LENGTH", 512)

class TrainingConfig(BaseModel):
    use_cuda: bool = get_development_settings("USE_CUDA", False)
    seed: int = get_development_settings("TRAINING_SEED", 42)

class TotalConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    app: AppConfig = Field(alias="app")
    data: DataConfig = Field(alias="data")
    model: ModelConfig = Field(alias="model")
    training: TrainingConfig = Field(alias="training")
