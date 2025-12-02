import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sympy as sp
from plotly.subplots import make_subplots

# Настройка страницы
st.set_page_config(page_title="Визуализация системы уравнений", layout="wide")

# Функция для анимации конкретной системы из задачи
def animate_task_system(a_min, a_max, x_min, x_max, steps=30):
    """Анимация системы: (x+a)^4 - y^4 - 0.5a^2(x+a)^2 + 0.5a^2y^2 = 0, y = ax + a/2"""
    
    a_values = np.linspace(a_min, a_max, steps)
    x_values = np.linspace(x_min, x_max, 400)
    
    # Создаем фигуру с двумя графиками
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('График системы', 'Зависимость решений от параметра a'),
        horizontal_spacing=0.15
    )
    
    frames = []
    
    for i, a in enumerate(a_values):
        # Уравнение 1: (x+a)^4 - y^4 - 0.5a^2(x+a)^2 + 0.5a^2y^2 = 0
        # Уравнение 2: y = ax + a/2
        
        # Подставляем y из второго уравнения в первое
        y_from_eq2 = a * x_values + a/2
        
        # Вычисляем левую часть первого уравнения
        eq1_values = (x_values + a)**4 - y_from_eq2**4 - 0.5*a**2*(x_values + a)**2 + 0.5*a**2*y_from_eq2**2
        
        # Находим корни (нули)
        roots = []
        for j in range(len(x_values)-1):
            if eq1_values[j] * eq1_values[j+1] <= 0:  # Знак меняется
                # Линейная интерполяция для нахождения корня
                x1, x2 = x_values[j], x_values[j+1]
                y1, y2 = eq1_values[j], eq1_values[j+1]
                if y2 != y1:
                    root = x1 - y1 * (x2 - x1) / (y2 - y1)
                    y_root = a * root + a/2
                    roots.append((root, y_root))
        
        # Создаем данные для кадра
        scatter_roots = go.Scatter(
            x=[r[0] for r in roots],
            y=[r[1] for r in roots],
            mode='markers',
            marker=dict(size=10, color='red'),
            name=f'Решения (a={a:.2f})',
            showlegend=True
        )
        
        # График уравнения
        scatter_eq = go.Scatter(
            x=x_values,
            y=eq1_values,
            mode='lines',
            line=dict(color='blue', width=2),
            name=f'F(x) при a={a:.2f}',
            showlegend=True
        )
        
        # Второй график: количество решений в зависимости от a
        scatter_count = go.Scatter(
            x=a_values[:i+1],
            y=[len(roots)] * (i+1) if roots else [0] * (i+1),
            mode='lines+markers',
            line=dict(color='green', width=3),
            name='Количество решений',
            showlegend=True
        )
        
        frame = go.Frame(
            data=[scatter_eq, scatter_roots, scatter_count],
            name=f'frame_{i}',
            layout=go.Layout(
                title=f'Система при a = {a:.2f}'
            )
        )
        frames.append(frame)
    
    # Первый кадр
    a_first = a_values[0]
    y_first = a_first * x_values + a_first/2
    eq1_first = (x_values + a_first)**4 - y_first**4 - 0.5*a_first**2*(x_values + a_first)**2 + 0.5*a_first**2*y_first**2
    
    # Находим корни для первого кадра
    roots_first = []
    for j in range(len(x_values)-1):
        if eq1_first[j] * eq1_first[j+1] <= 0:
            x1, x2 = x_values[j], x_values[j+1]
            y1, y2 = eq1_first[j], eq1_first[j+1]
            if y2 != y1:
                root = x1 - y1 * (x2 - x1) / (y2 - y1)
                y_root = a_first * root + a_first/2
                roots_first.append((root, y_root))
    
    # Добавляем первый график
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=eq1_first,
            mode='lines',
            line=dict(color='blue', width=2),
            name=f'F(x) при a={a_first:.2f}'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[r[0] for r in roots_first],
            y=[r[1] for r in roots_first],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Решения'
        ),
        row=1, col=1
    )
    
    # Добавляем второй график
    fig.add_trace(
        go.Scatter(
            x=[a_first],
            y=[len(roots_first)],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Количество решений'
        ),
        row=1, col=2
    )
    
    fig.frames = frames
    
    # Настройка анимации
    animation_settings = dict(
        frame=dict(duration=150, redraw=True),
        fromcurrent=True,
        mode='immediate'
    )
    
    # Кнопки управления
    updatemenus = [dict(
        type="buttons",
        buttons=[
            dict(
                label="▶️ Воспроизвести",
                method="animate",
                args=[None, animation_settings]
            ),
            dict(
                label="⏸️ Пауза",
                method="animate",
                args=[[None], dict(mode="immediate", frame=dict(duration=0))]
            ),
            dict(
                label="⏪ Назад",
                method="animate",
                args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))]
            )
        ],
        direction="left",
        pad=dict(r=10, t=10),
        showactive=True,
        x=0.1,
        y=1.15,
        xanchor="right",
        yanchor="top"
    )]
    
    # Ползунок
    sliders = [dict(
        steps=[dict(
            method='animate',
            args=[
                [f'frame_{k}'],
                dict(mode='immediate', frame=dict(duration=0))
            ],
            label=f'{a_values[k]:.2f}'
        ) for k in range(len(a_values))],
        active=0,
        currentvalue=dict(
            font=dict(size=14),
            prefix="a = ",
            visible=True,
            xanchor="center"
        ),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.1,
        y=0,
        xanchor="left",
        yanchor="top",
        transition=dict(duration=0)
    )]
    
    # Обновляем макет
    fig.update_layout(
        title=dict(
            text="Анимация системы: (x+a)⁴ - y⁴ - 0.5a²(x+a)² + 0.5a²y² = 0, y = ax + a/2",
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        height=500,
        template='plotly_white',
        updatemenus=updatemenus,
        sliders=sliders,
        showlegend=True
    )
    
    # Настройка осей
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="F(x) = (x+a)⁴ - y⁴ - 0.5a²(x+a)² + 0.5a²y²", row=1, col=1)
    
    fig.update_xaxes(title_text="Параметр a", row=1, col=2)
    fig.update_yaxes(title_text="Количество решений", row=1, col=2)
    
    return fig

# Функция для анимации одиночной функции
def animate_single_function(func_str, a_min, a_max, x_min, x_max, steps=50):
    """Анимация одиночной функции с параметром a"""
    
    # Парсим функцию
    def eval_func(x, a):
        try:
            # Создаем безопасное окружение
            safe_dict = {
                'x': x, 'a': a,
                'abs': abs,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'pi': np.pi, 'e': np.e
            }
            # Заменяем |x| на abs(x) и x² на x**2
            expr = func_str.replace('|x|', 'abs(x)').replace('|a|', 'abs(a)')
            expr = expr.replace('x²', 'x**2').replace('x^2', 'x**2')
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            return float(result) if isinstance(result, (int, float)) else 0.0
        except:
            return 0.0
    
    a_values = np.linspace(a_min, a_max, steps)
    x_values = np.linspace(x_min, x_max, 400)
    
    fig = go.Figure()
    
    frames = []
    for i, a in enumerate(a_values):
        # Вычисляем y для всех x
        y_values = [eval_func(x, a) for x in x_values]
        
        frame = go.Frame(
            data=[go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines',
                line=dict(color='blue', width=3),
                name=f'f(x) при a={a:.2f}'
            )],
            name=f'frame_{i}',
            layout=go.Layout(
                title=f'f(x) = {func_str} при a = {a:.2f}'
            )
        )
        frames.append(frame)
    
    # Первый кадр
    a_first = a_values[0]
    y_first = [eval_func(x, a_first) for x in x_values]
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_first,
        mode='lines',
        line=dict(color='blue', width=3),
        name=f'f(x) при a={a_first:.2f}'
    ))
    
    fig.frames = frames
    
    # Настройки анимации
    animation_settings = dict(
        frame=dict(duration=100, redraw=True),
        fromcurrent=True,
        mode='immediate'
    )
    
    # Управление
    updatemenus = [dict(
        type="buttons",
        buttons=[
            dict(label="▶️", method="animate", args=[None, animation_settings]),
            dict(label="⏸️", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])
        ]
    )]
    
    # Ползунок
    sliders = [dict(
        steps=[dict(
            method='animate',
            args=[[f'frame_{k}'], dict(mode='immediate', frame=dict(duration=0))],
            label=f'{a_values[k]:.2f}'
        ) for k in range(len(a_values))],
        active=0,
        currentvalue=dict(prefix="a = ", visible=True)
    )]
    
    fig.update_layout(
        title=f"Анимация функции: f(x) = {func_str}",
        xaxis_title="x",
        yaxis_title="f(x)",
        height=500,
        updatemenus=updatemenus,
        sliders=sliders,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

# Основной интерфейс
st.title("📈 Анимация системы уравнений из задачи")

# Создаем вкладки
tab1, tab2, tab3 = st.tabs(["📋 Задача", "🎬 Анимация системы", "📊 Одиночные функции"])

with tab1:
    st.header("Условие задачи")
    st.markdown("""
    ### Система уравнений:
    
    \[
    \\begin{cases} 
    (x + a)^4 - y^4 - 0.5a^2(x + a)^2 + 0.5a^2y^2 = 0, \\\\ 
    y = ax + \\frac{a}{2} 
    \\end{cases}
    \]
    
    ### Что нужно найти:
    Все положительные значения параметра \( a \), при каждом из которых система имеет ровно **два различных решения**.
    
    ### Как работает визуализация:
    1. Подставляем \( y = ax + \\frac{a}{2} \) в первое уравнение
    2. Получаем уравнение относительно \( x \):
       \[ F(x) = (x+a)^4 - (ax + a/2)^4 - 0.5a^2(x+a)^2 + 0.5a^2(ax + a/2)^2 = 0 \]
    3. Ищем корни этого уравнения (точки пересечения графиков)
    4. Анимируем изменение количества корней при изменении \( a \)
    """)
    
    st.info("💡 **Подсказка:** Решения системы соответствуют точкам, где график F(x) пересекает ось OX.")

with tab2:
    st.header("Анимация системы из задачи")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Настройки параметров")
        
        st.markdown("**Диапазон параметра a:**")
        a_min = st.number_input("Минимум a", -5.0, 5.0, 0.1, 0.1)
        a_max = st.number_input("Максимум a", -5.0, 5.0, 3.0, 0.1)
        
        st.markdown("**Диапазон x:**")
        x_min = st.number_input("Минимум x", -10.0, 10.0, -5.0, 0.5)
        x_max = st.number_input("Максимум x", -10.0, 10.0, 5.0, 0.5)
        
        steps = st.slider("Количество кадров", 10, 100, 30)
        
        animate_system_btn = st.button("🎬 Запустить анимацию", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("**Интерпретация результатов:**")
        st.info("""
        - **2 решения**: система имеет 2 различных точки пересечения
        - **1 решение**: графики касаются в одной точке
        - **0 решений**: графики не пересекаются
        - **>2 решений**: возможно при некоторых значениях a
        """)
    
    with col2:
        if animate_system_btn:
            with st.spinner("Создаем анимацию..."):
                fig = animate_task_system(a_min, a_max, x_min, x_max, steps)
                st.plotly_chart(fig, use_container_width=True)
            
            # Анализ результатов
            st.subheader("🔍 Анализ при конкретных значениях a")
            
            # Тестируем несколько значений a
            test_a_values = np.linspace(a_min, a_max, 10)
            
            results = []
            for a_test in test_a_values:
                # Вычисляем F(x) для этого a
                x_test = np.linspace(x_min, x_max, 1000)
                y_test = a_test * x_test + a_test/2
                F_test = (x_test + a_test)**4 - y_test**4 - 0.5*a_test**2*(x_test + a_test)**2 + 0.5*a_test**2*y_test**2
                
                # Находим корни
                roots_count = 0
                for j in range(len(x_test)-1):
                    if F_test[j] * F_test[j+1] <= 0:
                        roots_count += 1
                
                results.append((a_test, roots_count))
            
            # Показываем таблицу
            st.write("Количество решений при различных a:")
            for a_val, count in results:
                st.write(f"a = {a_val:.2f}: {count} решений")

with tab3:
    st.header("Анимация одиночных функций")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Выбор функции")
        
        example_funcs = {
            "a * sin(x)": "a * sin(x)",
            "a * cos(x)": "a * cos(x)",
            "a * x²": "a * x**2",
            "sin(a * x)": "sin(a * x)",
            "a * |x|": "a * abs(x)",
            "exp(-a * x)": "exp(-a * x)",
            "a * log(|x| + 1)": "a * log(abs(x) + 1)",
            "x² - a": "x**2 - a"
        }
        
        selected = st.selectbox("Примеры функций:", list(example_funcs.keys()))
        func_input = st.text_input("f(x) =", value=example_funcs[selected])
        
        st.subheader("Параметры анимации")
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            a_min_func = st.number_input("a мин", -5.0, 5.0, -2.0, 0.1, key="a_min_func")
        with col_a2:
            a_max_func = st.number_input("a макс", -5.0, 5.0, 2.0, 0.1, key="a_max_func")
        
        col_x1, col_x2 = st.columns(2)
        with col_x1:
            x_min_func = st.number_input("x мин", -10.0, 10.0, -5.0, 0.5, key="x_min_func")
        with col_x2:
            x_max_func = st.number_input("x макс", -10.0, 10.0, 5.0, 0.5, key="x_max_func")
        
        steps_func = st.slider("Количество кадров", 10, 100, 40, key="steps_func")
        
        animate_func_btn = st.button("🎬 Анимировать функцию", type="primary", use_container_width=True)
    
    with col2:
        if animate_func_btn and func_input:
            with st.spinner("Создаем анимацию..."):
                fig_func = animate_single_function(func_input, a_min_func, a_max_func, 
                                                  x_min_func, x_max_func, steps_func)
                st.plotly_chart(fig_func, use_container_width=True)
            
            # Дополнительная информация о функции
            st.subheader("ℹ️ Информация о функции")
            
            # Вычисляем некоторые характеристики
            try:
                x_sample = np.linspace(x_min_func, x_max_func, 100)
                a_mid = (a_min_func + a_max_func) / 2
                
                # Создаем безопасное окружение
                safe_dict = {
                    'x': x_sample, 'a': a_mid,
                    'abs': np.abs,
                    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                    'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                    'pi': np.pi, 'e': np.e
                }
                
                # Вычисляем значения
                expr = func_input.replace('|x|', 'abs(x)').replace('|a|', 'abs(a)')
                expr = expr.replace('x²', 'x**2').replace('x^2', 'x**2')
                y_sample = eval(expr, {"__builtins__": {}}, safe_dict)
                
                if isinstance(y_sample, (int, float)):
                    y_sample = np.full_like(x_sample, y_sample)
                
                # Находим экстремумы
                if len(y_sample) > 1:
                    diff = np.diff(y_sample)
                    extremum_count = np.sum((diff[:-1] * diff[1:] <= 0) & (np.abs(diff[:-1]) > 1e-6))
                    
                    st.write(f"**Среднее значение функции:** {np.mean(y_sample):.3f}")
                    st.write(f"**Максимальное значение:** {np.max(y_sample):.3f}")
                    st.write(f"**Минимальное значение:** {np.min(y_sample):.3f}")
                    st.write(f"**Примерное число экстремумов:** {extremum_count}")
                
            except Exception as e:
                st.write(f"Невозможно проанализировать функцию: {e}")

# Информация о синтаксисе
with st.expander("📚 Справка по синтаксису функций"):
    st.markdown("""
    ### Доступные функции:
    
    **Математические функции:**
    - `abs(x)` или `|x|` - модуль
    - `sin(x)`, `cos(x)`, `tan(x)` - тригонометрия
    - `exp(x)`, `log(x)`, `sqrt(x)`
    - `pi` (≈3.14159), `e` (≈2.71828)
    
    **Операторы:**
    - `+`, `-`, `*`, `/` - арифметика
    - `**` - возведение в степень
    - `()` - скобки
    
    **Примеры:**
    - `a * sin(x)`
    - `x**2 - a`
    - `exp(-a * abs(x))`
    - `a * log(x**2 + 1)`
    """)

# Футер
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
Визуализация системы уравнений из задачи и анимация функций • 
<a href='https://github.com/вашлогин/function-visualizer' target='_blank'>GitHub</a> • 
<a href='https://function-visualizer.streamlit.app' target='_blank'>Онлайн версия</a>
</div>
""", unsafe_allow_html=True)

# CSS для улучшения отображения
st.markdown("""
<style>
    .stButton > button {
        transition: all 0.3s;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)
