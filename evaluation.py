"""Метрики на данных, отложенных при обучении."""

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from data_processing import CLASSES

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_precision": "Macro precision",
    "macro_recall": "Macro recall",
    "macro_f1": "Macro F1",
}


def evaluate(actual, predicted):
    measured_classes = [label for label in CLASSES if label in set(actual)]
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=measured_classes, average="macro", zero_division=0,
    )
    scores = {
        "accuracy": accuracy_score(actual, predicted),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }
    return scores, confusion_matrix(actual, predicted, labels=list(CLASSES))
