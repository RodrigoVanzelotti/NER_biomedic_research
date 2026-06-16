import os
from pydantic import BaseModel
import yaml
from typing import Any, Dict, Type, TypeVar, Generic

from services.config.config_models import TotalConfig   

T = TypeVar('T', bound=BaseModel)

class ConfigLoader:
    @staticmethod
    def _replace_setting_with_env_vars(value: str) -> str:
        if not isinstance(value, str):
            return value
        
        trimmed_value = value.strip()
        if not trimmed_value.startswith("${") or not trimmed_value.endswith("}"):
            return value
        
        inner_content = trimmed_value[2:-1]
        if ":" not in inner_content:
            return value
        
        env_var_name, default_value = inner_content.split(":", 1)

        return os.environ.get(env_var_name, default_value)
    
    @staticmethod
    def _process_config(config: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in config.items():
            if isinstance(value, dict):
                config[key] = ConfigLoader._process_config(value)
            elif isinstance(value, str):
                config[key] = ConfigLoader._replace_setting_with_env_vars(value)
        return config
    
    @staticmethod
    def load_settings(config_name: str, config_class: Type[T]) -> T:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(os.path.dirname(current_dir))

        config_file = "settings.config.yaml"
        config_path = os.path.join(src_dir, "config", config_file)

        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
            processed_config = ConfigLoader._process_config(raw_config)
        
        return config_class.model_validate(processed_config)