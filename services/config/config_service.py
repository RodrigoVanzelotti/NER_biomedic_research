from typing import Any, Dict
from services.config.config_loader import ConfigLoader
from services.config.config_models import TotalConfig

# Singleton instance of the configuration
class ConfigService:
    _config: TotalConfig

    def __init__(self):
        self.__config = ConfigLoader.load_settings("settings", TotalConfig)

    def get(self) -> TotalConfig:
        return self.__config