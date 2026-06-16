import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from seqeval.metrics import classification_report
from transformers import BertForTokenClassification, Trainer, set_seed

from common.common_functions import (
    get_semantic_label_pairs,
    process_sample,
)
from services.config.config_service import ConfigService
from services.data_loader import load_data
from services.tokenizer import get_tokenizer
from training.data_collator import build_data_collator
from training.metrics import build_compute_metrics
from training.training_args import build_training_args


FINAL_EXPERIMENT = {
    "name": "final",
    "learning_rate": 2e-5,
    "epochs": 8,
    "batch_size": 8,
    "weight_decay": 0.01,
    "warmup_steps": 40,
}
FINAL_SEED = 42
OUTPUT_DIR = Path("./artifacts/biobert_ner_final")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def main():
    config = ConfigService().get()
    set_seed(FINAL_SEED)

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

    model = BertForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=labels,
    )

    training_args = build_training_args(FINAL_EXPERIMENT, run_id="final")
    compute_metrics = build_compute_metrics(id2label)
    data_collator = build_data_collator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_processed,
        eval_dataset=validation_processed,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save final model + tokenizer
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nSaved model + tokenizer -> {OUTPUT_DIR}")

    # 2. Training curves
    save_training_curves(trainer.state.log_history)

    # 3. Test predictions
    test_predictions = trainer.predict(test_processed)
    pred_ids = np.argmax(test_predictions.predictions, axis=2)
    label_ids = test_predictions.label_ids

    true_labels, pred_labels = decode_predictions(pred_ids, label_ids, id2label)

    # 4. Per-class metrics
    save_per_class_metrics(true_labels, pred_labels)

    # 5. Error analysis
    save_error_analysis(
        test_processed, pred_ids, label_ids, id2label, tokenizer
    )

    # 6. Headline metrics
    val_metrics = trainer.evaluate()
    save_final_metrics(val_metrics, test_predictions.metrics)

    print(f"\nAll artifacts written under {OUTPUT_DIR}/")


def decode_predictions(pred_ids, label_ids, id2label):
    true_labels = []
    pred_labels = []
    for pred_seq, label_seq in zip(pred_ids, label_ids):
        cur_true = []
        cur_pred = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            cur_true.append(id2label[int(l)])
            cur_pred.append(id2label[int(p)])
        true_labels.append(cur_true)
        pred_labels.append(cur_pred)
    return true_labels, pred_labels


def save_training_curves(log_history):
    rows = []
    for entry in log_history:
        if "eval_loss" in entry:
            rows.append({
                "epoch": entry.get("epoch"),
                "step": entry.get("step"),
                "kind": "eval",
                "loss": entry.get("eval_loss"),
                "precision": entry.get("eval_precision"),
                "recall": entry.get("eval_recall"),
                "f1": entry.get("eval_f1"),
                "lr": None,
                "grad_norm": None,
            })
        elif "loss" in entry:
            rows.append({
                "epoch": entry.get("epoch"),
                "step": entry.get("step"),
                "kind": "train",
                "loss": entry.get("loss"),
                "precision": None,
                "recall": None,
                "f1": None,
                "lr": entry.get("learning_rate"),
                "grad_norm": entry.get("grad_norm"),
            })

    path = OUTPUT_DIR / "training_curves.csv"
    fieldnames = ["epoch", "step", "kind", "loss", "precision", "recall", "f1", "lr", "grad_norm"]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved training curves data -> {path}")

    try_plot_curves(rows)


def try_plot_curves(rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping curve plot. "
              "Install with: pip install matplotlib")
        return

    train = [r for r in rows if r["kind"] == "train" and r["loss"] is not None]
    eval_ = [r for r in rows if r["kind"] == "eval" and r["loss"] is not None]

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(13, 5))

    ax_loss.plot(
        [r["epoch"] for r in train],
        [r["loss"] for r in train],
        label="Training Loss", marker=".", alpha=0.7,
    )
    ax_loss.plot(
        [r["epoch"] for r in eval_],
        [r["loss"] for r in eval_],
        label="Validation Loss", marker="o", linewidth=2,
    )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training & Validation Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_f1.plot(
        [r["epoch"] for r in eval_],
        [r["f1"] for r in eval_],
        label="Validation F1", marker="o", color="green", linewidth=2,
    )
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("F1")
    ax_f1.set_title("Validation F1 per Epoch")
    ax_f1.legend()
    ax_f1.grid(True, alpha=0.3)

    fig.suptitle("Final Model Training Curves (epochs=8, lr=2e-5, bs=8, seed=42)")
    fig.tight_layout()
    out = OUTPUT_DIR / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved training curves plot -> {out}")


def save_per_class_metrics(true_labels, pred_labels):
    report = classification_report(
        true_labels, pred_labels, output_dict=True, zero_division=0
    )

    with open(OUTPUT_DIR / "per_class_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, default=_json_default)

    rows = []
    for entity_type, metrics in report.items():
        if not isinstance(metrics, dict):
            continue
        rows.append({
            "entity": entity_type,
            "precision": round(float(metrics.get("precision", 0.0)), 4),
            "recall": round(float(metrics.get("recall", 0.0)), 4),
            "f1": round(float(metrics.get("f1-score", 0.0)), 4),
            "support": int(metrics.get("support", 0)),
        })

    rows.sort(key=lambda r: r["support"], reverse=True)

    csv_path = OUTPUT_DIR / "per_class_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["entity", "precision", "recall", "f1", "support"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-class metrics -> {csv_path}")

    print("\nPer-class test metrics:")
    print(f"{'entity':<32} {'P':>7} {'R':>7} {'F1':>7} {'support':>8}")
    print("-" * 65)
    for r in rows:
        print(
            f"{r['entity']:<32} {r['precision']:>7.4f} "
            f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['support']:>8}"
        )


def categorize_error(true_label, pred_label):
    if true_label == "O" and pred_label != "O":
        return "false_positive"
    if true_label != "O" and pred_label == "O":
        return "false_negative"

    true_prefix, true_type = true_label.split("-", 1)
    pred_prefix, pred_type = pred_label.split("-", 1)

    if true_type != pred_type:
        return "wrong_type"
    if true_prefix != pred_prefix:
        return "boundary"
    return "other"


def save_error_analysis(test_processed, pred_ids, label_ids, id2label, tokenizer):
    errors = []
    for sample_idx in range(len(test_processed)):
        sample = test_processed[sample_idx]
        input_ids = sample["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        seq_preds = pred_ids[sample_idx]
        seq_labels = label_ids[sample_idx]

        for token_idx, (token, p, l) in enumerate(
            zip(tokens, seq_preds, seq_labels)
        ):
            if l == -100:
                continue
            true_label = id2label[int(l)]
            pred_label = id2label[int(p)]
            if true_label == pred_label:
                continue

            # Capture +/- 3 tokens of context for readability
            ctx_start = max(0, token_idx - 3)
            ctx_end = min(len(tokens), token_idx + 4)
            context = " ".join(tokens[ctx_start:ctx_end])

            errors.append({
                "sample": sample_idx,
                "token_idx": token_idx,
                "token": token,
                "context": context,
                "true": true_label,
                "pred": pred_label,
                "category": categorize_error(true_label, pred_label),
            })

    fields = ["sample", "token_idx", "token", "context", "true", "pred", "category"]

    full_path = OUTPUT_DIR / "errors_full.csv"
    with open(full_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(errors)

    pattern_counter = Counter()
    pattern_examples = {}
    for err in errors:
        key = (err["token"], err["true"], err["pred"])
        pattern_counter[key] += 1
        pattern_examples.setdefault(key, err)

    top_patterns = []
    for (token, true_label, pred_label), count in pattern_counter.most_common(20):
        example = pattern_examples[(token, true_label, pred_label)]
        top_patterns.append({
            "rank": len(top_patterns) + 1,
            "count": count,
            "token": token,
            "true": true_label,
            "pred": pred_label,
            "category": example["category"],
            "example_context": example["context"],
        })

    top_path = OUTPUT_DIR / "errors_top_patterns.csv"
    with open(top_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["rank", "count", "token", "true", "pred", "category", "example_context"],
        )
        writer.writeheader()
        writer.writerows(top_patterns)

    category_counts = Counter(err["category"] for err in errors)
    summary = {
        "total_errors": len(errors),
        "by_category": dict(category_counts),
        "by_category_pct": {
            k: round(100 * v / max(1, len(errors)), 2)
            for k, v in category_counts.items()
        },
    }
    with open(OUTPUT_DIR / "errors_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, default=_json_default)

    print(f"Saved error analysis -> {full_path}")
    print(f"Saved top error patterns -> {top_path}")
    print("\nError category breakdown:")
    for cat, count in category_counts.most_common():
        pct = summary["by_category_pct"][cat]
        print(f"  {cat:<18} {count:>6} ({pct:>5.1f}%)")


def save_final_metrics(val_metrics, test_metrics):
    def clean(d):
        return {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in d.items()
        }

    final = {
        "experiment": FINAL_EXPERIMENT,
        "seed": FINAL_SEED,
        "val": clean(val_metrics),
        "test": clean(test_metrics),
    }
    path = OUTPUT_DIR / "final_metrics.json"
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(final, fp, indent=2, default=_json_default)
    print(f"Saved headline metrics -> {path}")


if __name__ == "__main__":
    main()
