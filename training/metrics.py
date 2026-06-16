import numpy as np

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score
)


def build_compute_metrics(id2label):

    def compute_metrics(eval_pred):

        predictions, labels = eval_pred

        predictions = np.argmax(
            predictions,
            axis=2
        )

        true_predictions = []
        true_labels = []

        for prediction, label in zip(
            predictions,
            labels
        ):

            current_predictions = []
            current_labels = []

            for pred_id, label_id in zip(
                prediction,
                label
            ):

                if label_id == -100:
                    continue

                current_predictions.append(
                    id2label[pred_id]
                )

                current_labels.append(
                    id2label[label_id]
                )

            true_predictions.append(
                current_predictions
            )

            true_labels.append(
                current_labels
            )

        return {
            "precision": precision_score(
                true_labels,
                true_predictions
            ),
            "recall": recall_score(
                true_labels,
                true_predictions
            ),
            "f1": f1_score(
                true_labels,
                true_predictions
            )
        }

    return compute_metrics