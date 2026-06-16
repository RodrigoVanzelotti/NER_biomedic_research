from transformers import TrainingArguments


def build_training_args():

    return TrainingArguments(
        output_dir="./results",

        eval_strategy="epoch",
        save_strategy="epoch",

        learning_rate=2e-5,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        num_train_epochs=3,

        weight_decay=0.01,

        logging_steps=10,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        save_total_limit=2,

        report_to="none"
    )