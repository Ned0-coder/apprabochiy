import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import re

# Настройка страницы
st.set_page_config(page_title="Визуализация функции", layout="wide")


# Функция для обработки модулей
def parse_absolute_values(expr):
    """Заменяет |x| на abs(x)"""
    if '|' not in expr:
        return expr

    # Простые замены
    expr = expr.replace('|x|', 'abs(x)')
    expr = expr.replace('|a|', 'abs(a)')

    # Обрабатываем сложные случаи
    # Находим все выражения внутри модулей
    pattern = r'\|([^|]+)\|'

    def replace_match(match):
        inner = match.group(1)
        # Убираем лишние пробелы
        inner = inner.strip()
        return f'abs({inner})'

    # Применяем замену
    while '|' in expr:
        new_expr = re.sub(pattern, replace_match, expr)
        if new_expr == expr:  # Если ничего не изменилось
            break
        expr = new_expr

    return expr


# Функция вычисления для одного значения x
def calculate_point(func_str, x_val, a_val):
    """Вычисляет значение функции в одной точке"""
    try:
        # Парсим модули
        expr = parse_absolute_values(func_str)

        # Создаем безопасное окружение
        safe_dict = {
            'x': x_val,
            'a': a_val,
            'abs': abs,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
            'pi': np.pi,
            'e': np.e
        }

        # Вычисляем
        result = eval(expr, {"__builtins__": {}}, safe_dict)

        # Проверяем на особые значения
        if isinstance(result, (int, float)):
            return float(result)
        else:
            return 0.0

    except Exception as e:
        return 0.0


# Функция вычисления для массива x
def calculate_function(func_str, x_values, a_val):
    """Вычисляет функцию для массива значений x"""
    y_values = []
    for x in x_values:
        y = calculate_point(func_str, x, a_val)
        y_values.append(y)
    return np.array(y_values)


# Функция для создания анимации
def create_animation(func_str, a_min, a_max, x_min, x_max, steps=50):
    """Создает анимацию изменения функции"""
    try:
        # Генерируем значения
        a_values = np.linspace(a_min, a_max, steps)
        x_values = np.linspace(x_min, x_max, 300)

        # Создаем фигуру
        fig = go.Figure()

        # Добавляем первый кадр
        y_initial = calculate_function(func_str, x_values, a_values[0])
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_initial,
            mode='lines',
            line=dict(color='blue', width=3),
            name=f'a = {a_values[0]:.2f}'
        ))

        # Создаем кадры для анимации
        frames = []
        for i, a in enumerate(a_values):
            y = calculate_function(func_str, x_values, a)

            frame = go.Frame(
                data=[go.Scatter(
                    x=x_values,
                    y=y,
                    mode='lines',
                    line=dict(color='blue', width=3),
                    name=f'a = {a:.2f}'
                )],
                name=f'frame_{i}'
            )
            frames.append(frame)

        fig.frames = frames

        # Настройки анимации
        animation_settings = {
            'frame': {'duration': 100, 'redraw': True},
            'fromcurrent': True,
            'mode': 'immediate'
        }

        # Кнопки управления
        updatemenus = [{
            'type': 'buttons',
            'buttons': [
                {
                    'label': '▶️ Воспроизвести',
                    'method': 'animate',
                    'args': [None, animation_settings]
                },
                {
                    'label': '⏸️ Пауза',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': False,
            'x': 0.1,
            'y': 0
        }]

        # Ползунок
        sliders = [{
            'steps': [
                {
                    'method': 'animate',
                    'args': [
                        [f'frame_{k}'],
                        {'frame': {'duration': 0}, 'mode': 'immediate'}
                    ],
                    'label': f'{a_values[k]:.2f}'
                } for k in range(len(a_values))
            ],
            'active': 0,
            'currentvalue': {
                'font': {'size': 16},
                'prefix': 'a = ',
                'visible': True
            },
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0
        }]

        # Настройка макета
        fig.update_layout(
            title={
                'text': f'Анимация: {func_str}',
                'font': {'size': 20}
            },
            xaxis_title='x',
            yaxis_title='f(a, x)',
            showlegend=True,
            updatemenus=updatemenus,
            sliders=sliders,
            height=500,
            template='plotly_white'
        )

        fig.update_xaxes(
            gridcolor='lightgray',
            zerolinecolor='lightgray',
            range=[x_min, x_max]
        )

        fig.update_yaxes(
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        )

        return fig, None

    except Exception as e:
        return None, str(e)


# Функция для статичного графика
def create_static_plot(func_str, a_val, x_min, x_max):
    """Создает статичный график функции"""
    try:
        x_values = np.linspace(x_min, x_max, 500)
        y_values = calculate_function(func_str, x_values, a_val)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            line=dict(color='red', width=3),
            name=f'f({a_val:.2f}, x)'
        ))

        fig.update_layout(
            title={
                'text': f'График: {func_str} при a = {a_val:.2f}',
                'font': {'size': 18}
            },
            xaxis_title='x',
            yaxis_title='f(a, x)',
            height=450,
            template='plotly_white',
            showlegend=True
        )

        fig.update_xaxes(
            gridcolor='lightgray',
            zerolinecolor='lightgray',
            range=[x_min, x_max]
        )

        fig.update_yaxes(
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        )

        return fig, None

    except Exception as e:
        return None, str(e)


# Основной интерфейс
st.title("📈 Визуализация функции f(a, x)")
st.markdown("---")

# Создаем колонки
col_left, col_right = st.columns([1, 1.5])

with col_left:
    st.header("⚙️ Параметры")

    # Выбор функции
    st.subheader("Функция")

    example_funcs = {
        "a * |x|": "a * |x|",
        "|a * x|": "|a * x|",
        "sin(|x|)": "sin(|x|)",
        "|sin(x)|": "|sin(x)|",
        "a * x²": "a * x**2",
        "sin(a * x)": "sin(a * x)",
        "a * sin(x)": "a * sin(x)",
        "exp(-|x|)": "exp(-|x|)",
        "|x| - a": "|x| - a",
        "cos(a * |x|)": "cos(a * |x|)"
    }

    selected = st.selectbox(
        "Выберите пример:",
        list(example_funcs.keys())
    )

    func_input = st.text_input(
        "Ваша функция:",
        value=example_funcs[selected],
        help="Используйте a как параметр, x как переменную. |x| - модуль."
    )

    # Диапазоны
    st.subheader("Диапазоны")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        a_min = st.number_input("a мин", value=-3.0, step=0.5)
    with col_a2:
        a_max = st.number_input("a макс", value=3.0, step=0.5)

    col_x1, col_x2 = st.columns(2)
    with col_x1:
        x_min = st.number_input("x мин", value=-10.0, step=1.0)
    with col_x2:
        x_max = st.number_input("x макс", value=10.0, step=1.0)

    # Текущее значение a
    st.subheader("Текущее значение параметра")
    a_current = st.slider(
        "a =",
        min_value=float(a_min),
        max_value=float(a_max),
        value=1.0,
        step=0.1
    )

    # Кнопки
    st.markdown("---")
    animate_clicked = st.button(
        "🎬 Запустить анимацию",
        type="primary",
        use_container_width=True
    )

    st.button(
        "🔄 Обновить график",
        type="secondary",
        use_container_width=True
    )

with col_right:
    st.header("📊 График")

    if func_input:
        # Всегда показываем статичный график
        with st.spinner("Рисуем график..."):
            static_fig, error = create_static_plot(
                func_input,
                a_current,
                x_min,
                x_max
            )

            if error:
                st.error(f"Ошибка: {error}")
                st.info("Проверьте правильность функции")
            else:
                st.plotly_chart(static_fig, use_container_width=True)

                # Показываем преобразованную функцию
                parsed = parse_absolute_values(func_input)
                with st.expander("ℹ️ Информация"):
                    st.markdown(f"""
                    **Функция:** `{func_input}`

                    **Для вычислений:** `{parsed}`

                    **Параметры:**
                    - a = {a_current:.2f}
                    - x ∈ [{x_min}, {x_max}]
                    """)

        # Если нажата кнопка анимации
        if animate_clicked:
            with st.spinner("Создаем анимацию..."):
                animate_fig, error = create_animation(
                    func_input,
                    a_min,
                    a_max,
                    x_min,
                    x_max
                )

                if error:
                    st.error(f"Ошибка анимации: {error}")
                else:
                    st.plotly_chart(animate_fig, use_container_width=True)
                    st.success("Анимация готова! Используйте кнопки управления.")

    else:
        st.info("👈 Введите функцию в левой панели")

        # Демонстрационный график
        x_demo = np.linspace(-10, 10, 500)
        y_demo = np.abs(x_demo)

        demo_fig = go.Figure()
        demo_fig.add_trace(go.Scatter(
            x=x_demo,
            y=y_demo,
            mode='lines',
            line=dict(color='green', width=3),
            name='|x|'
        ))

        demo_fig.update_layout(
            title="Пример: f(x) = |x|",
            height=400,
            template='plotly_white'
        )

        st.plotly_chart(demo_fig, use_container_width=True)

# Футер
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Визуализация математических функций • Поддержка модуля |x|"
    "</div>",
    unsafe_allow_html=True
)

# CSS стили для улучшения вида
st.markdown("""
<style>
    .stButton > button {
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
    }
    .stNumberInput input {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)