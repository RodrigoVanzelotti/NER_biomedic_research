from transformers import BertTokenizerFast
from services.config.config_service import ConfigService

config = ConfigService().get()

def get_tokenizer():
    return BertTokenizerFast.from_pretrained(
        config.model.name
    )