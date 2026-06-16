from transformers import (
    BertForTokenClassification,
    Trainer
)

from services.tokenizer import get_tokenizer
from training.metrics import build_compute_metrics


def evaluate_model(
    model_path,
    test_dataset,
    id2label
):

    tokenizer = get_tokenizer()

    model = BertForTokenClassification.from_pretrained(
        model_path
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(
            id2label
        )
    )

    predictions = trainer.predict(
        test_dataset
    )

    return predictions.metrics