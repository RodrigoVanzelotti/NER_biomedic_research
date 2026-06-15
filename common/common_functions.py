from datetime import datetime
from pathlib import Path
from transformers import BertTokenizerFast
from collections import Counter

from common.dataset_inspection import _inspect_ner_dataset


tokenizer = BertTokenizerFast.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1"
)

base_tokenizer_config = {
    "truncation": True,
    "max_length": 512
}

def reconstruct_document(sample):
    passages = sample["passages"]

    ordered_passages = sorted(
        passages,
        key=lambda x: x["offsets"][0][0]
    )

    full_text = ""

    for passage in ordered_passages:
        full_text += passage["text"][0] + " "

    return full_text.strip()

def tokenize_with_offsets(text):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        **base_tokenizer_config
    )

    return encoding

def create_bio_tags(text, entities, predefined_labels):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        **base_tokenizer_config
    )
    offsets = encoding["offset_mapping"]
    labels = ["O"] * len(offsets)

    for entity in entities:

        entity_start = entity["offsets"][0][0]
        entity_end = entity["offsets"][0][1]

        entity_type = entity["semantic_type_id"]

        first_token = True

        for idx, (start, end) in enumerate(offsets):
            if start == end:
                continue

            overlap = (
                start < entity_end and
                end > entity_start
            )

            if overlap:

                if first_token:
                    labels[idx] = f"B-{entity_type}"
                    first_token = False

                else:
                    labels[idx] = f"I-{entity_type}"

    numeric_labels = [
        predefined_labels[label]
        for label in labels
    ]

    encoding["labels"] = numeric_labels

    return encoding

def process_sample(sample, predefined_labels):

    text = reconstruct_document(sample)

    processed = create_bio_tags(
        text,
        sample["entities"],
        predefined_labels
    )

    return processed

def get_semantic_label_pairs(train_dataset):
    semantic_types = set()

    for sample in train_dataset:

        for entity in sample["entities"]:

            semantic_types.add(
                entity["semantic_type_id"]
            )

    LABELS = {
        "O": 0
    }

    current_id = 1

    for semantic_type in sorted(semantic_types):

        LABELS[f"B-{semantic_type}"] = current_id
        current_id += 1

        LABELS[f"I-{semantic_type}"] = current_id
        current_id += 1

    ID2LABEL = {
        value: key
        for key, value in LABELS.items()
    }

    return LABELS, ID2LABEL

def inspect_ner_dataset(
    dataset,
    raw_dataset,
    id2label: dict[int, str], 
):
    return _inspect_ner_dataset(
        dataset,
        tokenizer,
        raw_dataset,
        id2label
    )