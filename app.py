"""Локальный интерфейс демонстрации дерева решений."""

from base64 import b64encode
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

import export
import model
from data_processing import CLASSES, EMPLOYMENT, FIELDS, METHODS, TARGET, DataError, read_csv
from evaluation import METRIC_LABELS
from visualization import confusion_figure, explain_path, number, png, tree_figure

ROOT = Path(__file__).resolve().parent
DEMO_PATH = ROOT / "input" / "clients.csv"
SECTIONS = ("Данные", "Обучение и проверка", "Оценка клиента")
PARAM_DEFAULTS = {
    "test_percent": 20, "criterion": "gini", "max_depth": 6,
    "min_samples_split": 2, "min_samples_leaf": 1, "random_state": 42,
}
CLIENT_KEYS = (*[f"client_{key}" for key in FIELDS], "client_method")


def navigate(section):
    st.session_state.section = section


def load_source(name, content):
    state = st.session_state
    for key in ("dataset", "source", "run", "assessment", "saved_report", "source_error"):
        state.pop(key, None)
    try:
        data = read_csv(content)
    except DataError as error:
        state.source_error = str(error)
    else:
        state.dataset = data
        state.source = name


def load_demo():
    state = st.session_state
    state.upload_generation = state.get("upload_generation", 0) + 1
    try:
        content = DEMO_PATH.read_bytes()
    except OSError:
        load_source(None, None)
        state.source_error = "Демо-файл input/clients.csv недоступен. Проверьте папку проекта"
    else:
        load_source("Демонстрационный набор", content)


def load_upload(key):
    uploaded = st.session_state.get(key)
    if uploaded is None:
        load_source(None, None)
    else:
        load_source(uploaded.name, uploaded.getvalue())


def page_intro(title, description):
    with st.container(key="page_intro"):
        st.html('<a class="skip-link" href="#main-workspace">К основному содержимому</a>'
                f'<h1 id="main-workspace" tabindex="-1">{escape(title)}</h1>')
        st.write(description)


def sidebar():
    state = st.session_state
    with st.sidebar:
        st.html(
            '<div class="brand"><strong>Оценка в автосалоне</strong>'
            '<small>Системы искусственного интеллекта · ЛР №3</small></div>'
        )
        section = st.radio("Разделы приложения", SECTIONS, key="section",
                           label_visibility="collapsed", width="stretch")
        data_status = f"{number(len(state.dataset))} записей" if "dataset" in state else "ожидают загрузки"
        run_status = "обучена" if "run" in state else "ожидает обучения"
        st.html(
            f'<div class="session-status"><p><strong>Данные</strong> · {data_status}</p>'
            f'<p><strong>Модель</strong> · {run_status}</p></div>'
        )
    return section


def data_page():
    state = st.session_state
    page_intro(
        "Покупательская способность",
        "Оценка для клиента автосалона по доходам, обязательствам и условиям покупки",
    )
    source, overview = st.columns([1, 1.12])
    with source, st.container(border=True, key="source_panel"):
        st.subheader("Источник данных", anchor=False)
        key = f"upload_{state.get('upload_generation', 0)}"
        st.file_uploader("CSV с метками классов", type="csv", key=key, on_change=load_upload, args=(key,))
        st.caption("UTF-8, запятая между полями, точка в дробных числах")
        st.button("Загрузить демо-набор", on_click=load_demo, type="primary")
        if state.get("source_error"):
            st.error(state.source_error)
        with st.expander("Какие столбцы нужны"):
            st.markdown(
                "Доход, расходы, текущие платежи, стаж, взнос, цена, способ оплаты и метка класса"
            )
            st.code(",".join((*FIELDS, "purchase_method", "ability_class")), language=None, wrap_lines=True)
            st.caption(
                "`purchase_method` принимает `cash`, `credit`, `lease`. "
                "`ability_class` принимает `low`, `medium`, `high`. "
                "`client_id` необязателен, стаж можно оставить пустым."
            )

    with overview, st.container(border=True, key="data_overview"):
        st.subheader("Состав выборки", anchor=False)
        data = state.get("dataset")
        if data is None:
            st.write("Выберите CSV или загрузите демонстрационный набор с тремя классами")
            st.caption(
                "Демо-набор состоит из синтетических записей. Метка зависит от взноса "
                "и суммы, которую клиент может накопить за 36 месяцев."
            )
        else:
            st.text(state.source)
            st.metric("Записей в наборе", number(len(data)))
            counts = data[TARGET].value_counts()
            distribution = pd.DataFrame({
                "Класс": list(CLASSES.values()),
                "Метка CSV": list(CLASSES),
                "Записей": [number(counts.get(label, 0)) for label in CLASSES],
            })
            st.html(distribution.to_html(index=False, border=0, classes="class-table"))
            missing = data[EMPLOYMENT].isna().sum()
            if missing:
                st.caption(
                    f"Записей без стажа: {number(missing)}. Для заполнения используем медиану по обучающей выборке"
                )
            st.button("Перейти к обучению", on_click=navigate, args=(SECTIONS[1],))


def training_controls(data):
    with st.container(border=True, key="train_controls"):
        st.subheader("Параметры дерева", anchor=False)
        test_percent = st.slider("Доля тестовой выборки, %", 1, 99, key="test_percent")
        criterion = st.selectbox("Критерий разбиения", ["gini", "entropy", "log_loss"], key="criterion")
        depth = st.number_input("Макс. глубина", min_value=1, step=1, key="max_depth")
        first, second = st.columns(2, gap="small")
        with first:
            min_split = st.number_input(
                "Мин. для деления", min_value=2, step=1, key="min_samples_split",
                help="Минимум записей в узле, чтобы попробовать его разделить",
            )
        with second:
            min_leaf = st.number_input(
                "Мин. в листе", min_value=1, step=1, key="min_samples_leaf",
                help="Минимальное число записей в каждом листе",
            )
        seed = st.number_input(
            "Случайное зерно (random_state)", min_value=0, max_value=2**32 - 1,
            step=1, key="random_state",
        )
        params = dict(test_size=test_percent / 100, criterion=criterion, max_depth=depth,
                      min_samples_split=min_split, min_samples_leaf=min_leaf, random_state=seed)
        if st.button(
            "Обучить дерево", type="primary", disabled=data is None, width="stretch", key="train",
        ):
            with st.spinner("Обучаем дерево…"):
                try:
                    run = model.train(data, **params)
                    model.save_model(run.pipeline, model.MODEL_PATH)
                except DataError as error:
                    st.error(str(error))
                except OSError:
                    st.error("Не удалось сохранить модель. Проверьте доступ к папке artifacts")
                else:
                    st.session_state.run = run
                    st.session_state.pop("assessment", None)
                    st.session_state.pop("saved_report", None)
                    st.rerun()
        return params


def chart(figure, description, *, tree=False):
    encoded = b64encode(png(figure)).decode("ascii")
    width, height = (round(size * figure.dpi) for size in figure.get_size_inches())
    kind = "chart tree-chart" if tree else "chart"
    scroll = ' tabindex="0" aria-label="Прокручиваемая схема дерева"' if tree else ""
    st.html(
        f'<figure class="{kind}" style="--chart-width: {width}px;"{scroll}>'
        f'<img alt="{escape(description, quote=True)}" width="{width}" height="{height}" '
        f'src="data:image/png;base64,{encoded}"></figure>'
    )


def show_quality(run, params):
    with st.container(border=True, key="quality_panel"):
        st.subheader("Качество на тестовой выборке", anchor=False)
        if run is None:
            st.write("После обучения здесь появятся метрики и матрица ошибок")
            st.caption("Стратифицированное разбиение сохраняет доли классов, насколько позволяет размер выборки")
            return
        if run.params != params:
            st.warning("Настройки изменились. Показан предыдущий успешный запуск. Обучите дерево, чтобы применить их")
        _, train_size, test_size = run.sizes
        st.caption(f"Обучение: {number(train_size)} записей · Тест: {number(test_size)} записей")
        for column, (metric, label) in zip(st.columns(4, gap="small"), METRIC_LABELS.items()):
            column.metric(label, f"{run.scores[metric]:.3f}".replace(".", ","))
        st.caption("Macro — среднее с одинаковым весом по каждому представленному классу")
        matrix, actions = st.columns([2, 1], gap="medium")
        with matrix:
            st.markdown("#### Матрица ошибок", unsafe_allow_html=False)
            rows = [
                f"«{label}»: {', '.join(str(value) for value in row)}"
                for label, row in zip(CLASSES.values(), run.matrix)
            ]
            chart(confusion_figure(run.matrix), "Матрица ошибок. Столбцы: низкая, средняя, высокая. "
                  f"Фактические классы по строкам: {'. '.join(rows)}.")
            st.caption("Совпадения факта и прогноза лежат на диагонали")
        with actions:
            st.markdown("#### Сводка запуска")
            st.caption(
                f"{run.params['criterion']} · глубина {run.pipeline['tree'].get_depth()} · "
                f"random_state {run.params['random_state']}"
            )
            content = export.summary_csv(run)
            if st.button("Сохранить сводку", key="save_report", width="stretch"):
                try:
                    export.save_summary(content, export.REPORT_PATH)
                except OSError:
                    st.error("Не удалось сохранить сводку. Проверьте доступ к папке reports")
                else:
                    st.session_state.saved_report = True
            if st.session_state.get("saved_report"):
                st.caption("Сохранено в reports/experiment_summary.csv")
            st.download_button(
                "Скачать CSV", content, file_name=export.REPORT_PATH.name, mime="text/csv",
                on_click="ignore", width="stretch",
            )


def training_page():
    state = st.session_state
    data = state.get("dataset")
    page_intro("Обучение и проверка", "Настройте дерево и проверьте прогнозы на отложенных данных")
    if data is None:
        st.info("Сначала загрузите CSV в разделе «Данные»")
        st.button("Открыть данные", on_click=navigate, args=(SECTIONS[0],))
    else:
        st.text(state.source)
    controls, result = st.columns([1, 2.12])
    with controls:
        params = training_controls(data)
    with result:
        run = state.get("run")
        show_quality(run, params)
    if run is not None:
        with st.expander("Структура дерева"):
            actual_depth = run.pipeline["tree"].get_depth()
            limit = min(3, actual_depth)
            depth = 0 if not limit else st.slider("Глубина отображения", 0, limit, min(2, limit))
            st.caption(
                f"Глубина модели: {actual_depth}. На схеме видно до {depth} уровней от корня. "
                "Точные пороги для конкретного клиента доступны в пояснении к прогнозу."
            )
            chart(tree_figure(run.pipeline, depth), f"Схема дерева до глубины {depth}.", tree=True)


def assessment_page():
    state = st.session_state
    page_intro("Оценка клиента", "Введите финансовые параметры клиента и условия покупки автомобиля")
    run = state.get("run")
    if run is None:
        st.info("Для оценки клиента сначала обучите дерево на загруженных данных")
        st.button("Открыть обучение", on_click=navigate, args=(SECTIONS[1],))
        return

    form, result = st.columns([1.24, 1])
    with form, st.container(border=True, key="client_form"):
        st.subheader("Параметры клиента", anchor=False)
        with st.form("client_inputs", border=False):
            values = {}
            for title, columns in (
                ("Доход и обязательства", tuple(FIELDS)[:4]),
                ("Условия покупки", tuple(FIELDS)[4:]),
            ):
                st.markdown(f"#### {title}")
                for offset in range(0, len(columns), 2):
                    for box, key in zip(st.columns(2, gap="small"), columns[offset:offset + 2]):
                        label, unit = FIELDS[key]
                        values[key] = box.number_input(
                            f"{label}, {unit.rstrip('.')}", value=None, key=f"client_{key}",
                            step=1.0 if key == EMPLOYMENT else 1_000.0,
                            format="%g" if key == EMPLOYMENT else None,
                        )
            values["purchase_method"] = st.selectbox(
                "Способ оплаты", list(METHODS), index=None, format_func=METHODS.__getitem__,
                placeholder="Выберите способ…", key="client_method",
            )
            submitted = st.form_submit_button("Оценить клиента", type="primary")
        if submitted:
            state.pop("assessment", None)
            try:
                label, client = model.predict_client(run.pipeline, values)
            except DataError as error:
                st.error(str(error))
            else:
                state.assessment = label, explain_path(run.pipeline, client)

    with result, st.container(border=True, key="client_result"):
        st.subheader("Результат оценки", anchor=False)
        assessment = state.get("assessment")
        if assessment is None:
            st.write("После оценки здесь появятся класс и условия, которым соответствуют введённые данные")
        else:
            label, conditions = assessment
            st.html(f'<p class="result-class" role="status">{CLASSES[label]} покупательская способность</p>')
            st.markdown("#### Путь по дереву")
            if conditions:
                steps = "".join(
                    f"<li><strong>{escape(field)}</strong><span>{escape(detail)}</span></li>"
                    for field, detail in conditions
                )
                st.html(f'<ol class="branch-path">{steps}</ol>')
            else:
                st.write("Дерево состоит из одного листа и присваивает один класс всем входным данным")
        st.caption("Результат предназначен для учебной демонстрации")


def main():
    st.set_page_config(page_title="Оценка в автосалоне", layout="wide", initial_sidebar_state="expanded")
    # Keep widget values when Streamlit removes the controls on another page.
    for key, default in PARAM_DEFAULTS.items():
        st.session_state[key] = st.session_state.get(key, default)
    for key in CLIENT_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    st.html(ROOT / "styles.css")
    section = sidebar()
    if section == SECTIONS[0]:
        data_page()
    elif section == SECTIONS[1]:
        training_page()
    else:
        assessment_page()


if __name__ == "__main__":
    main()
