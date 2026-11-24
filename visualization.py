"""Графики эксперимента и фактические ветви прогноза."""

from io import BytesIO
from threading import Lock

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from data_processing import CLASSES, EMPLOYMENT, FIELDS, METHODS

INK = "#20201E"
MUTED = "#55554E"
COLORMAP = LinearSegmentedColormap.from_list("counts", ["#F5F5F2", "#BCBCB3", "#373733"])
PLOT_LOCK = Lock()

SHORT_FEATURES = {
    "monthly_income": "Доход",
    "monthly_expenses": "Расходы",
    "existing_monthly_payments": "Платежи",
    EMPLOYMENT: "Стаж",
    "down_payment": "Взнос",
    "target_car_price": "Цена",
}


def number(value, digits=None) -> str:
    """Русское представление числа с точностью, достаточной для порогов дерева."""
    if abs(value) >= 1e15 or (value and abs(value) < 1e-6):
        return np.format_float_scientific(float(value), precision=digits, trim="-").replace(".", ",")
    integer, _, fraction = np.format_float_positional(
        float(value), precision=digits, trim="-",
    ).partition(".")
    grouped = f"{int(integer):,}".replace(",", "\u202f")
    return f"{grouped},{fraction}" if fraction else grouped


def confusion_figure(matrix) -> Figure:
    figure = Figure(figsize=(5.1, 3.0), dpi=145, facecolor="white")
    axes = figure.subplots()
    image = axes.imshow(matrix, cmap=COLORMAP, vmin=0, aspect="auto")
    labels = list(CLASSES.values())
    axes.set_xticks(range(len(labels)), labels=labels)
    axes.set_yticks(range(len(labels)), labels=labels)
    axes.tick_params(length=0, pad=8, labelsize=10, colors=MUTED)
    axes.set_xlabel("Прогноз", labelpad=8, color=MUTED, fontsize=10)
    axes.set_ylabel("Факт", labelpad=8, color=MUTED, fontsize=10)
    axes.spines[:].set_visible(False)
    for (row, column), count in np.ndenumerate(matrix):
        shade = image.cmap(image.norm(count))[0]
        axes.text(column, row, str(count), ha="center", va="center",
                  color="white" if shade < 0.46 else "black",
                  fontsize=18, fontweight="medium")
    axes.set_xticks(np.arange(-0.5, len(labels)), minor=True)
    axes.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    axes.tick_params(which="minor", length=0)
    axes.grid(which="minor", color="white", linewidth=2)
    figure.subplots_adjust(left=0.22, right=0.97, top=0.96, bottom=0.22)
    return figure


def explain_path(pipeline, client) -> list[tuple[str, str]]:
    tree = pipeline["tree"]
    structure = tree.tree_
    preprocessor = pipeline["preprocessor"]
    names = preprocessor.get_feature_names_out()
    transformed = preprocessor.transform(client)
    path = tree.decision_path(transformed).indices
    conditions = []
    for node, child in zip(path, path[1:]):
        index = structure.feature[node]
        feature = names[index]
        left = child == structure.children_left[node]
        threshold = structure.threshold[node]
        if feature in FIELDS:
            label, unit = FIELDS[feature]
            actual = float(np.float32(transformed[0, index]))
            relation = "≤" if left else ">"
            actual_text = number(actual)
            if feature != EMPLOYMENT:
                raw = float(client.iloc[0][feature])
                if float(np.float32(raw)) == actual and (raw <= threshold) == left:
                    actual_text = number(raw)
            conditions.append((label, f"{actual_text} {relation} {number(threshold)} {unit}"))
        else:
            tested = METHODS[feature.removeprefix("purchase_method_")].lower()
            actual_method = METHODS[client.iloc[0]["purchase_method"]]
            rule = "не «" if left else "«"
            conditions.append(("Способ оплаты", f"{actual_method} · условие {rule}{tested}»"))
    return conditions


def tree_figure(pipeline, depth: int) -> Figure:
    tree = pipeline["tree"]
    structure = tree.tree_
    names = pipeline["preprocessor"].get_feature_names_out()
    depth = min(depth, tree.get_depth())
    figure = Figure(figsize=(11.6, max(1.5, 1.12 * (depth + 1))), dpi=145, facecolor="white")
    axes = figure.subplots()
    axes.set_axis_off()
    axes.set_xlim(0, 2 ** depth)
    axes.set_ylim(-depth - 0.55, 0.55)
    figure.subplots_adjust(left=0.018, right=0.982, top=0.98, bottom=0.02)

    def draw(node, level, start, end):
        x = (start + end) / 2
        left, right = structure.children_left[node], structure.children_right[node]
        leaf = left == right
        count = structure.n_node_samples[node]
        if leaf:
            label = CLASSES[tree.classes_[np.argmax(structure.value[node, 0])]]
            title = f"{label}\nЗаписей: {count}"
        elif level == depth:
            title = f"Ветви скрыты\nЗаписей: {count}"
        else:
            feature = names[structure.feature[node]]
            if feature in FIELDS:
                _, unit = FIELDS[feature]
                value = number(structure.threshold[node], digits=3)
                title = f"{SHORT_FEATURES[feature]}\n≤ {value} {unit}"
            else:
                method = METHODS[feature.removeprefix("purchase_method_")]
                title = f"Способ оплаты\n{method}?"
            for child, low, high in ((left, start, x), (right, x, end)):
                child_x = (low + high) / 2
                axes.plot([x, child_x], [-level, -level - 1], color="#9D9D94", linewidth=1)
                is_left = child == left
                branch = ("Да" if is_left else "Нет") if feature in FIELDS else ("Нет" if is_left else "Да")
                axes.text((x + child_x) / 2, -level - 0.5, branch,
                          ha="center", va="center", color=MUTED, fontsize=9,
                          bbox=dict(facecolor="white", edgecolor="none", pad=1))
                draw(child, level + 1, low, high)
        axes.text(x, -level, title, ha="center", va="center", color=INK, fontsize=10,
                  linespacing=1.38, zorder=3,
                  bbox=dict(boxstyle="round,pad=0.47", facecolor="#E8E8E1" if leaf else "#F5F5F2",
                            edgecolor="#A9A99E" if leaf else "#D6D6CE", linewidth=0.9))

    draw(0, 0, 0, 2 ** depth)
    return figure


def png(figure: Figure) -> bytes:
    with PLOT_LOCK:
        buffer = BytesIO()
        figure.savefig(buffer, format="png")
    return buffer.getvalue()
