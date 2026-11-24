"""Сводный CSV активного эксперимента."""

import csv
import json
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from data_processing import CLASSES, FEATURES
from model import Experiment

REPORT_PATH = Path(__file__).resolve().parent / "reports" / "experiment_summary.csv"


def summary_csv(run: Experiment) -> bytes:
    row_count, train_count, test_count = run.sizes
    counts = {
        label: run.class_counts["train"][label] + run.class_counts["test"][label]
        for label in CLASSES
    }
    row = {
        "trained_at": run.trained_at.isoformat(timespec="seconds"),
        **run.params,
        "row_count": row_count,
        "train_count": train_count,
        "test_count": test_count,
        "features": json.dumps(FEATURES),
        "classes": json.dumps(CLASSES, ensure_ascii=False),
        "class_counts": json.dumps(counts),
        "train_class_counts": json.dumps(run.class_counts["train"]),
        "test_class_counts": json.dumps(run.class_counts["test"]),
        **run.scores,
    }
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    return output.getvalue().encode("utf-8-sig")


def save_summary(content: bytes, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
