"""Обучение конвейера и прогноз для нового клиента."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from data_processing import CLASSES, make_preprocessor, split_data, validate
from evaluation import evaluate

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "model.joblib"


@dataclass
class Experiment:
    pipeline: Pipeline
    trained_at: datetime
    params: dict
    sizes: tuple[int, int, int]
    class_counts: dict[str, dict[str, int]]
    scores: dict[str, float]
    matrix: np.ndarray


def train(data: pd.DataFrame, test_size=0.2, criterion="gini", max_depth=6,
          min_samples_split=2, min_samples_leaf=1, random_state=42) -> Experiment:
    train_x, test_x, train_y, test_y = split_data(data, test_size, random_state)
    tree_params = dict(
        criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf, random_state=random_state,
    )
    pipeline = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])
    pipeline.fit(train_x, train_y)
    scores, matrix = evaluate(test_y, pipeline.predict(test_x))

    counts = {}
    for name, values in (("train", train_y), ("test", test_y)):
        frequencies = values.value_counts()
        counts[name] = {label: int(frequencies.get(label, 0)) for label in CLASSES}
    return Experiment(
        pipeline, datetime.now().astimezone(), {"test_size": test_size, **tree_params},
        (len(data), len(train_y), len(test_y)), counts, scores, matrix,
    )


def save_model(pipeline: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        joblib.dump(pipeline, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def predict_client(pipeline: Pipeline, values: dict) -> tuple[str, pd.DataFrame]:
    client = validate(pd.DataFrame([values]), training=False)
    return str(pipeline.predict(client)[0]), client
