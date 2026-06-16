from transformers import (
    BertForTokenClassification,
    Trainer
)

from common.common_functions import *
from services.data_loader import load_data
from services.tokenizer import get_tokenizer
from training.training_args import build_training_args
from training.metrics import build_compute_metrics
from training.data_collator import build_data_collator


dataset = load_data()

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
test_dataset = dataset["test"]

LABELS, ID2LABEL = get_semantic_label_pairs(train_dataset)

train_processed = train_dataset.map(lambda sample: process_sample(sample, LABELS))
validation_processed = validation_dataset.map(lambda sample: process_sample(sample, LABELS))
test_processed = test_dataset.map(lambda sample: process_sample(sample, LABELS))

report_healthy = inspect_ner_dataset(
    train_processed,
    train_dataset,
    ID2LABEL,
)

if not report_healthy:
    print("Dataset inspection failed. Please check the report for details.")
    exit(1)
else:
    print("Dataset inspection passed. The dataset is healthy and ready for training.")


model = BertForTokenClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABELS
)

# model.save_pretrained("./my_biobert_model")
    
training_args = build_training_args()

tokenizer = get_tokenizer()

compute_metrics = build_compute_metrics(
    ID2LABEL
)

data_collator = build_data_collator(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_processed,
    eval_dataset=validation_processed,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

trainer.train()

validation_metrics = trainer.evaluate()

print(validation_metrics)

test_results = trainer.predict(
    test_processed
)

print(test_results.metrics)

trainer.save_model(
    "./artifacts/biobert_ner"
)