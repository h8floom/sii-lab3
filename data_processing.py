"""Загрузка, проверка и подготовка входных данных."""

import csv
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

NUMERIC = (
    "monthly_income", "monthly_expenses", "existing_monthly_payments",
    "employment_months", "down_payment", "target_car_price",
)
EMPLOYMENT = "employment_months"
METHOD = "purchase_method"
TARGET = "ability_class"
FEATURES = (*NUMERIC, METHOD)
CLASSES = {"low": "Низкая", "medium": "Средняя", "high": "Высокая"}
METHODS = {"cash": "Полная оплата", "credit": "Кредит", "lease": "Лизинг"}
FIELDS = {
    "monthly_income": ("Доход за месяц", "₽/мес."),
    "monthly_expenses": ("Обязательные расходы", "₽/мес."),
    "existing_monthly_payments": ("Текущие платежи", "₽/мес."),
    EMPLOYMENT: ("Стаж работы", "мес."),
    "down_payment": ("Первоначальный взнос", "₽"),
    "target_car_price": ("Цена автомобиля", "₽"),
}
MAX_TREE_NUMBER = float(np.finfo(np.float32).max)


class DataError(ValueError):
    """Ошибка входных данных, которую можно показать пользователю."""


def read_csv(content: bytes | None) -> pd.DataFrame:
    if content is None:
        raise DataError("Выберите CSV-файл или загрузите демо-набор")
    try:
        source = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise DataError("Сохраните CSV-файл в кодировке UTF-8") from error
    if "\x00" in source:
        raise DataError("Файл содержит двоичные данные. Выберите текстовый CSV")
    try:
        reader = csv.reader(StringIO(source, newline=""), strict=True)
        header = next((row for row in reader if row), [])
        if not header:
            raise DataError("CSV-файл пуст")
        if len(set(header)) != len(header):
            raise DataError("Удалите повторяющиеся названия столбцов из CSV")
        for row in reader:
            if row and len(row) != len(header):
                raise DataError(
                    f"Строка файла {reader.line_num}: число полей не совпадает с заголовком"
                )
        frame = pd.read_csv(StringIO(source), dtype=str, keep_default_na=False)
    except (csv.Error, pd.errors.ParserError) as error:
        raise DataError("Проверьте кавычки и разделители в CSV-файле. Разделитель — запятая") from error
    return validate(frame, training=True)


def validate(frame: pd.DataFrame, *, training: bool) -> pd.DataFrame:
    required = (*FEATURES, TARGET) if training else FEATURES
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataError(f"Добавьте обязательные столбцы: {', '.join(missing)}")
    if frame.empty:
        raise DataError("В CSV есть заголовок, но нет записей. Добавьте строки с данными")
    columns = ["client_id", *required] if training and "client_id" in frame else list(required)
    result = frame.loc[:, columns].reset_index(drop=True).copy()

    def check(invalid: pd.Series, message: str) -> None:
        index = invalid.to_numpy().nonzero()[0]
        if index.size:
            prefix = f"Запись {index[0] + 1}: " if training else ""
            raise DataError(f"{prefix}{message}")

    for column in NUMERIC:
        label, _ = FIELDS[column]
        raw = result[column].astype("string").str.strip()
        blank = raw.isna() | raw.eq("")
        numbers = pd.to_numeric(raw.mask(blank), errors="coerce").astype(float)
        if not (training and column == EMPLOYMENT):
            check(blank, f"заполните поле «{label}»")
        check(~blank & ~np.isfinite(numbers), f"поле «{label}» должно содержать число")
        check(numbers < 0, f"поле «{label}» должно быть неотрицательным")
        check(numbers > MAX_TREE_NUMBER, f"значение поля «{label}» слишком велико")
        if column == EMPLOYMENT:
            check(numbers.notna() & (numbers % 1 != 0), "укажите стаж целым числом месяцев")
        result[column] = numbers

    check(result["target_car_price"] <= 0, "цена автомобиля должна быть больше нуля")
    check(result["down_payment"] > result["target_car_price"], "первоначальный взнос превышает цену автомобиля")

    categories = {METHOD: METHODS, TARGET: CLASSES} if training else {METHOD: METHODS}
    for column, choices in categories.items():
        values = result[column].astype("string").str.strip()
        if column == METHOD:
            check(values.isna() | values.eq(""), "выберите способ оплаты")
        check(~values.isin(choices), f"для {column} допустимы значения {', '.join(choices)}")
        result[column] = values.astype(str)
    return result


def split_data(data: pd.DataFrame, test_size: float, random_state: int):
    y = data[TARGET]
    counts = y.value_counts()
    if len(counts) < 2:
        raise DataError("Для обучения нужны записи как минимум двух классов")
    if (counts < 2).any():
        raise DataError("Для стратификации добавьте хотя бы две записи каждого представленного класса")
    try:
        split = train_test_split(
            data.loc[:, list(FEATURES)], y, test_size=test_size,
            random_state=random_state, stratify=y,
        )
    except ValueError as error:
        raise DataError(
            "Данных недостаточно для указанной доли теста. Измените долю или добавьте записи"
        ) from error
    x_train, _, y_train, y_test = split
    if set(y_train) != set(counts.index) or set(y_test) != set(counts.index):
        raise DataError(
            "Некоторые классы не попали в обе выборки. Измените долю теста или добавьте записи редких классов"
        )
    if x_train[EMPLOYMENT].isna().all():
        raise DataError("В обучающей выборке нет значений стажа. Добавьте записи с указанным стажем")
    return split


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("employment", SimpleImputer(strategy="median"), [EMPLOYMENT]),
        ("money", "passthrough", [column for column in NUMERIC if column != EMPLOYMENT]),
        ("method", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [METHOD]),
    ], verbose_feature_names_out=False)
