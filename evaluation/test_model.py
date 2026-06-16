from transformers import (
    BertForTokenClassification,
    Trainer,
    TrainingArguments
)

from services.config.config_service import ConfigService
from services.tokenizer import get_tokenizer
from training.metrics import build_compute_metrics

config = ConfigService().get()

def evaluate_model(
    model_path,
    test_dataset,
    id2label
):

    tokenizer = get_tokenizer()

    model = BertForTokenClassification.from_pretrained(
        model_path
    )

    eval_args = TrainingArguments(
        output_dir="./results",
        report_to="none",
        use_cpu=not config.training.use_cuda
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(
            id2label
        )
    )

    predictions = trainer.predict(
        test_dataset
    )

    return predictions.metrics