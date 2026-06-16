from datetime import datetime
from pathlib import Path
from collections import Counter

from common.dataset_inspection import _inspect_ner_dataset


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

def create_bio_tags(text, entities, predefined_labels, tokenizer, max_length=512):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length
    )
    offsets = encoding["offset_mapping"]
    
    word_ids = encoding.word_ids()

    o_id = predefined_labels["O"]

    labels = []
    previous_word_id = None
    
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != previous_word_id:
            labels.append(o_id)
        else:
            labels.append(-100)
        previous_word_id = word_id


    for entity in entities:

        entity_start = entity["offsets"][0][0]
        entity_end = entity["offsets"][0][1]

        entity_type = entity["semantic_type_id"]

        first_token = True

        for idx, (start, end) in enumerate(offsets):
            if labels[idx] == -100 or start == end:
                continue

            overlap = (
                start < entity_end and
                end > entity_start
            )

            if not overlap: continue

            tag = f'B-{entity_type}' if first_token else f'I-{entity_type}'
            first_token = False
            labels[idx] = predefined_labels[tag]
            
    encoding.pop("offset_mapping", None)
    encoding["labels"] = labels

    return encoding

def process_sample(sample, predefined_labels, tokenizer, max_length=512):

    text = reconstruct_document(sample)

    processed = create_bio_tags(
        text,
        sample["entities"],
        predefined_labels,
        tokenizer,
        max_length
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
    tokenizer
):
    return _inspect_ner_dataset(
        dataset,
        tokenizer,
        raw_dataset,
        id2label
    )