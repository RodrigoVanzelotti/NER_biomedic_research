from transformers import TrainingArguments
from services.config.config_service import ConfigService

config = ConfigService().get()

def build_training_args(experiment=None):
    experiment = experiment or {}

    use_cuda = config.training.use_cuda

    return TrainingArguments(
        output_dir="./results",

        eval_strategy="epoch",
        save_strategy="epoch",

        learning_rate=experiment.get("learning_rate", 2e-5),

        per_device_train_batch_size=experiment.get("batch_size", 8),
        per_device_eval_batch_size=experiment.get("batch_size", 8),

        num_train_epochs=experiment.get("epochs", 3),
        weight_decay=experiment.get("weight_decay", 0.01),

        warmup_ratio=0.1,

        learning_rate=2e-5,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        num_train_epochs=3,

        weight_decay=0.01,

        logging_steps=10,

        load_best_model_at_end=True,

        metric_for_best_model="eval_f1",
        greater_is_better=True,

        save_total_limit=2,

        report_to="none",

        seed=config.training.seed,

        use_cpu=not use_cuda,
        fp16=use_cuda,
        dataloader_pin_memory=use_cuda,
        dataloading_num_workers=2
    )