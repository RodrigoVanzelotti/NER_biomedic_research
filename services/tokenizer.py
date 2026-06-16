from transformers import BertTokenizerFast


def get_tokenizer():
    return BertTokenizerFast.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1"
)