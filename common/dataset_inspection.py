from pathlib import Path
import statistics
import json
from datetime import datetime
from collections import Counter


def _raw_sample_to_text(sample) -> str:
    passages = sample["passages"]

    if isinstance(passages, list):
        if passages and isinstance(passages[0], dict):
            ordered_passages = sorted(
                passages,
                key=lambda passage: passage["offsets"][0][0],
            )

            return " ".join(
                passage["text"][0]
                for passage in ordered_passages
            ).strip()

        return " ".join(str(passage) for passage in passages).strip()

    return str(passages)


def _inspect_ner_dataset(
    dataset,
    tokenizer,
    raw_dataset,
    id2label: dict[int, str],
) -> bool:

    # =========================================================================
    # HEALTH THRESHOLDS
    # =========================================================================

    MIN_ENTITY_COVERAGE = 0.05
    MAX_ENTITY_IMBALANCE_RATIO = 50.0
    MAX_TRUNCATED_SAMPLE_RATIO = 0.50

    # =========================================================================
    # VALIDATION
    # =========================================================================

    required_columns = {
        "input_ids",
        "attention_mask",
        "labels",
    }

    missing_columns = (
        required_columns
        - set(dataset.column_names)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    valid_label_ids = set(id2label.keys())

    sequence_lengths = []

    label_counter = Counter()
    entity_counter = Counter()

    invalid_samples = []
    invalid_labels = []
    bio_violations = []
    special_token_errors = []

    # =========================================================================
    # DATASET VALIDATION
    # =========================================================================

    for sample_idx, sample in enumerate(dataset):

        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        labels = sample["labels"]

        if not (
            len(input_ids)
            == len(attention_mask)
            == len(labels)
        ):
            invalid_samples.append(sample_idx)

        sequence_lengths.append(
            len(input_ids)
        )

        tokens = tokenizer.convert_ids_to_tokens(
            input_ids
        )

        previous_label = "O"

        for token_idx, (token, label_id) in enumerate(
            zip(tokens, labels)
        ):

            if label_id == -100:

                if token not in {
                    "[CLS]",
                    "[SEP]",
                    "[PAD]",
                }:
                    special_token_errors.append(
                        {
                            "sample": sample_idx,
                            "token": token_idx,
                            "value": token,
                        }
                    )

                continue

            if label_id not in valid_label_ids:

                invalid_labels.append(
                    {
                        "sample": sample_idx,
                        "token": token_idx,
                        "label_id": label_id,
                    }
                )

                continue

            label_name = id2label[label_id]

            label_counter[label_name] += 1

            if label_name.startswith("B-"):
                entity_counter[
                    label_name[2:]
                ] += 1

            if label_name.startswith("I-"):

                current_type = label_name[2:]

                if previous_label == "O":

                    bio_violations.append(
                        {
                            "sample": sample_idx,
                            "token": token_idx,
                            "reason": "I tag after O",
                        }
                    )

                elif previous_label.startswith(("B-", "I-")):

                    previous_type = previous_label[2:]

                    if previous_type != current_type:

                        bio_violations.append(
                            {
                                "sample": sample_idx,
                                "token": token_idx,
                                "reason": "Entity type mismatch",
                            }
                        )

            previous_label = label_name

    # =========================================================================
    # BASIC STATS
    # =========================================================================

    total_tokens = sum(
        label_counter.values()
    )

    entity_tokens = sum(
        count
        for label, count in label_counter.items()
        if label != "O"
    )

    coverage = (
        entity_tokens / total_tokens
        if total_tokens
        else 0
    )

    max_length = max(sequence_lengths)

    max_length_samples = sum(
        1
        for x in sequence_lengths
        if x == max_length
    )

    max_length_sample_ratio = (
        max_length_samples
        / len(sequence_lengths)
    )

    largest_entity = (
        max(entity_counter.values())
        if entity_counter
        else 0
    )

    smallest_entity = (
        min(entity_counter.values())
        if entity_counter
        else 1
    )

    entity_imbalance_ratio = (
        largest_entity
        / max(1, smallest_entity)
    )

    o_ratio = (
        label_counter.get("O", 0)
        / total_tokens
        if total_tokens
        else 0
    )

    avg_entity_length = (
        entity_tokens
        / max(1, sum(entity_counter.values()))
    )

    rare_entities = {
        entity: count
        for entity, count in entity_counter.items()
        if count < 100
    }

    # =========================================================================
    # TRUNCATION ANALYSIS
    # =========================================================================

    original_lengths = []
    final_lengths = []
    lost_tokens = []

    for sample in raw_dataset:

        text = _raw_sample_to_text(sample)

        original = tokenizer(
            text,
            truncation=False,
            add_special_tokens=True,
        )

        truncated = tokenizer(
            text,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )

        original_length = len(
            original["input_ids"]
        )

        final_length = len(
            truncated["input_ids"]
        )

        original_lengths.append(
            original_length
        )

        final_lengths.append(
            final_length
        )

        lost_tokens.append(
            max(
                0,
                original_length - final_length
            )
        )

    truncated_samples = sum(
        1
        for x in lost_tokens
        if x > 0
    )

    truncation_stats = {
        "samples": len(raw_dataset),
        "truncated_samples": truncated_samples,
        "truncated_sample_ratio": (
            truncated_samples
            / len(raw_dataset)
        ),
        "average_original_tokens":
            statistics.mean(original_lengths),
        "average_tokens_after_truncation":
            statistics.mean(final_lengths),
        "average_truncated_tokens":
            statistics.mean(
                [
                    x
                    for x in lost_tokens
                    if x > 0
                ]
            )
            if truncated_samples
            else 0,
        "max_truncated_tokens":
            max(lost_tokens),
        "total_lost_tokens":
            sum(lost_tokens),
    }

    # =========================================================================
    # HEALTH CHECKS
    # =========================================================================

    health_checks = {
        "valid_sample_lengths":
            len(invalid_samples) == 0,

        "valid_labels":
            len(invalid_labels) == 0,

        "valid_bio":
            len(bio_violations) == 0,

        "entity_coverage":
            coverage >= MIN_ENTITY_COVERAGE,

        "entity_balance":
            entity_imbalance_ratio
            <= MAX_ENTITY_IMBALANCE_RATIO,

        "truncation":
            truncation_stats[
                "truncated_sample_ratio"
            ]
            <= MAX_TRUNCATED_SAMPLE_RATIO,
    }

    healthy = all(
        health_checks.values()
    )

    # =========================================================================
    # REPORT
    # =========================================================================

    json_report = {
        "healthy": healthy,

        "health_checks": health_checks,

        "health_signals": {
            "coverage": coverage,
            "o_ratio": o_ratio,
            "entity_imbalance_ratio":
                entity_imbalance_ratio,
            "avg_entity_length":
                avg_entity_length,
            "max_length_sample_ratio":
                max_length_sample_ratio,
        },

        "dataset_size": len(dataset),

        "length_statistics": {
            "min_length":
                min(sequence_lengths),
            "max_length":
                max(sequence_lengths),
            "mean_length":
                statistics.mean(sequence_lengths),
            "std_length":
                statistics.stdev(sequence_lengths),
        },

        "integrity": {
            "invalid_samples":
                invalid_samples,
            "invalid_labels":
                invalid_labels,
            "bio_violations":
                bio_violations,
            "special_token_errors":
                special_token_errors,
        },

        "entity_distribution":
            dict(entity_counter),

        "label_distribution":
            dict(label_counter),

        "rare_entities":
            rare_entities,

        "truncation_stats":
            truncation_stats,
    }

    file_ts = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    if not Path("reports").exists():
        Path("reports").mkdir()

    json_path = (
        Path("reports")
        / f"ner_dataset_report_{file_ts}.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            json_report,
            fp,
            indent=4,
            ensure_ascii=False,
        )

    return healthy