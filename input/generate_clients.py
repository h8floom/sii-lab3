"""Воспроизводимый синтетический набор без персональных данных."""

import csv
from pathlib import Path
from random import Random


def generate(path: Path) -> None:
    rng = Random(42)
    columns = (
        "client_id", "monthly_income", "monthly_expenses", "existing_monthly_payments",
        "employment_months", "down_payment", "target_car_price", "purchase_method", "ability_class",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(columns)
        for index in range(1, 1201):
            income = rng.randrange(55, 251) * 1000
            expenses = int(income * rng.uniform(0.40, 0.74) / 1000) * 1000
            payments = rng.randrange(0, 41, 2) * 1000
            employment = "" if rng.random() < 0.025 else rng.randrange(3, 181)
            price = rng.randrange(22, 111) * 50_000
            down = int(price * rng.uniform(0.08, 0.49) / 1000) * 1000
            method = rng.choices(("cash", "credit", "lease"), weights=(3, 6, 1))[0]
            free = max(0, income - expenses - payments)
            ratio = (down + 36 * free) / price
            label = "low" if ratio < 0.8 else "medium" if ratio < 1.2 else "high"
            writer.writerow((f"D{index:04}", income, expenses, payments, employment,
                             down, price, method, label))


if __name__ == "__main__":
    output = Path(__file__).with_name("clients.csv")
    if output.exists():
        raise SystemExit("clients.csv уже существует. Переместите файл перед повторной генерацией")
    generate(output)
