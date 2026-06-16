import csv
import os
import shutil
import statistics
import time
from datetime import datetime
from pathlib import Path

from transformers import BertForTokenClassification, Trainer, set_seed

from common.common_functions import (
    get_semantic_label_pairs,
    process_sample,
)
from experiments.baseline import EXPERIMENT as EXP_BASELINE
from experiments.experiment_5_epochs import EXPERIMENT as EXP_5_EPOCHS
from experiments.experiment_lr_3e5 import EXPERIMENT as EXP_LR_3E5
from experiments.experiment_batch_16 import EXPERIMENT as EXP_BATCH_16

from experiments.experiment_batch_16_fixed import EXPERIMENT as EXP_BATCH_16_FIXED
from experiments.experiment_8_epochs import EXPERIMENT as EXP_8_EPOCHS
from experiments.experiment_combined import EXPERIMENT as EXP_COMBINED

from services.config.config_service import ConfigService
from services.data_loader import load_data
from services.tokenizer import get_tokenizer
from training.data_collator import build_data_collator
from training.metrics import build_compute_metrics
from training.training_args import build_training_args


EXPERIMENTS = [EXP_BATCH_16_FIXED, EXP_8_EPOCHS, EXP_COMBINED]
SEEDS = [42, 43, 44]


def run_one(
    experiment,
    seed,
    train_processed,
    validation_processed,
    test_processed,
    labels,
    id2label,
    tokenizer,
    compute_metrics,
    data_collator,
    model_name,
):
    # Override seed via env var so ConfigService picks it up when
    # build_training_args() reloads the YAML.
    os.environ["TRAINING_SEED"] = str(seed)
    set_seed(seed)

    run_id = f"{experiment['name']}_seed{seed}"

    model = BertForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=labels,
    )

    training_args = build_training_args(experiment, run_id=run_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_processed,
        eval_dataset=validation_processed,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    started = time.time()
    train_output = trainer.train()
    train_runtime = time.time() - started

    val_metrics = trainer.evaluate()
    test_metrics = trainer.predict(test_processed).metrics

    del trainer, model
    shutil.rmtree(f"./results/{run_id}", ignore_errors=True)

    return {
        "experiment": experiment["name"],
        "seed": seed,
        "epochs": experiment["epochs"],
        "batch_size": experiment["batch_size"],
        "learning_rate": experiment["learning_rate"],
        "warmup_steps": experiment.get("warmup_steps", 0),
        "train_runtime_s": round(train_runtime, 2),
        "train_loss": round(train_output.training_loss, 4),
        "val_loss": round(val_metrics["eval_loss"], 4),
        "val_precision": round(val_metrics["eval_precision"], 4),
        "val_recall": round(val_metrics["eval_recall"], 4),
        "val_f1": round(val_metrics["eval_f1"], 4),
        "test_loss": round(test_metrics["test_loss"], 4),
        "test_precision": round(test_metrics["test_precision"], 4),
        "test_recall": round(test_metrics["test_recall"], 4),
        "test_f1": round(test_metrics["test_f1"], 4),
    }


def main():
    config = ConfigService().get()
    tokenizer = get_tokenizer()
    max_length = config.model.max_length
    model_name = config.model.name

    dataset = load_data()
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    labels, id2label = get_semantic_label_pairs(train_dataset)

    train_processed = train_dataset.map(
        lambda sample: process_sample(sample, labels, tokenizer, max_length)
    )
    validation_processed = validation_dataset.map(
        lambda sample: process_sample(sample, labels, tokenizer, max_length)
    )
    test_processed = test_dataset.map(
        lambda sample: process_sample(sample, labels, tokenizer, max_length)
    )

    compute_metrics = build_compute_metrics(id2label)
    data_collator = build_data_collator(tokenizer)

    results = []
    total = len(EXPERIMENTS) * len(SEEDS)
    idx = 0

    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            idx += 1
            print(
                f"\n=== [{idx}/{total}] {experiment['name']} (seed={seed}) ===\n"
            )
            row = run_one(
                experiment,
                seed,
                train_processed,
                validation_processed,
                test_processed,
                labels,
                id2label,
                tokenizer,
                compute_metrics,
                data_collator,
                model_name,
            )
            results.append(row)
            print(
                f"  -> val_f1={row['val_f1']}  test_f1={row['test_f1']}  "
                f"runtime={row['train_runtime_s']}s"
            )

    # Persist CSV
    Path("reports").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path("reports") / f"sweep_{ts}.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Aggregate per experiment
    print("\n\n=== SUMMARY (mean +/- stdev across seeds) ===\n")
    print(f"{'experiment':<22} {'val_f1':<18} {'test_f1':<18} {'runtime_s':<10}")
    print("-" * 70)

    by_exp = {}
    for row in results:
        by_exp.setdefault(row["experiment"], []).append(row)

    summary_rows = []
    for name, rows in by_exp.items():
        val_f1s = [r["val_f1"] for r in rows]
        test_f1s = [r["test_f1"] for r in rows]
        runtimes = [r["train_runtime_s"] for r in rows]

        val_mean, val_std = statistics.mean(val_f1s), statistics.stdev(val_f1s) if len(val_f1s) > 1 else 0.0
        test_mean, test_std = statistics.mean(test_f1s), statistics.stdev(test_f1s) if len(test_f1s) > 1 else 0.0
        rt_mean = statistics.mean(runtimes)

        print(
            f"{name:<22} "
            f"{val_mean:.4f} +/- {val_std:.4f}   "
            f"{test_mean:.4f} +/- {test_std:.4f}   "
            f"{rt_mean:.1f}"
        )
        summary_rows.append({
            "experiment": name,
            "val_f1_mean": round(val_mean, 4),
            "val_f1_std": round(val_std, 4),
            "test_f1_mean": round(test_mean, 4),
            "test_f1_std": round(test_std, 4),
            "runtime_s_mean": round(rt_mean, 1),
            "n_seeds": len(rows),
        })

    summary_path = Path("reports") / f"sweep_{ts}_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nFull results: {csv_path}")
    print(f"Summary:      {summary_path}")


if __name__ == "__main__":
    main()
