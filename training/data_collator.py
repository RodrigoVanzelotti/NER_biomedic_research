from transformers import DataCollatorForTokenClassification


def build_data_collator(tokenizer):

    return DataCollatorForTokenClassification(tokenizer=tokenizer)