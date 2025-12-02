import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

# Настройка страницы
st.set_page_config(page_title="Анимация системы уравнений", layout="wide")

# Функция для анимации СИСТЕМЫ из задачи
def animate_task_system(a_min, a_max, x_min, x_max, y_min, y_max, steps=30):
    """Анимация системы из задачи без решения, только визуализация"""
    
    a_values = np.linspace(a_min, a_max, steps)
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Создаем фигуру с двумя графиками
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Первое уравнение', 'Второе уравнение'),
        horizontal_spacing=0.15
    )
    
    frames = []
    
    for i, a in enumerate(a_values):
        # Уравнение 1: (x+a)^4 - y^4 - 0.5a^2(x+a)^2 + 0.5a^2y^2 = 0
        # Уравнение 2: y = ax + a/2
        
        # Вычисляем значения первого уравнения
        Z1 = (X + a)**4 - Y**4 - 0.5*a**2*(X + a)**2 + 0.5*a**2*Y**2
        
        # Для второго уравнения создаем линию y = ax + a/2
        x_line = np.linspace(x_min, x_max, 100)
        y_line = a * x_line + a/2
        
        # Создаем следы для кадра
        # 1. Контур первого уравнения (нулевая линия уровня)
        contour1 = go.Contour(
            z=Z1,
            x=x_grid,
            y=y_grid,
            contours=dict(
                coloring='lines',
                start=0,
                end=0,
                size=0,
                showlabels=True
            ),
            line_width=3,
            colorscale='Reds',
            name=f'Ур.1: (x+{a:.1f})⁴ - y⁴ - 0.5·{a:.1f}²(x+{a:.1f})² + 0.5·{a:.1f}²y² = 0',
            showscale=False,
            contours_coloring='lines',
            line_color='red'
        )
        
        # 2. Линия второго уравнения
        scatter2 = go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            line=dict(color='blue', width=3),
            name=f'Ур.2: y = {a:.1f}x + {a/2:.1f}'
        )
        
        # 3. Контур первого уравнения для второго графика (те же данные)
        contour1_copy = go.Contour(
            z=Z1,
            x=x_grid,
            y=y_grid,
            contours=dict(
                coloring='lines',
                start=0,
                end=0,
                size=0
            ),
            line_width=3,
            colorscale='Reds',
            showscale=False,
            line_color='red'
        )
        
        # 4. Пересечения (где оба уравнения выполняются)
        # Находим точки, близкие к выполнению обоих уравнений
        mask = np.abs(Z1) < 0.5 * np.max(np.abs(Z1))  # Упрощенный критерий
        
        # Собираем точки пересечения
        intersection_points = []
        if np.any(mask):
            # Берем подвыборку точек
            indices = np.where(mask)
            if len(indices[0]) > 0:
                for idx in range(0, min(20, len(indices[0])), 2):  # Берем каждую вторую точку до 20
                    xi = indices[1][idx]
                    yi = indices[0][idx]
                    # Проверяем, близка ли точка ко второму уравнению
                    y_expected = a * x_grid[xi] + a/2
                    if abs(y_grid[yi] - y_expected) < 0.5:
                        intersection_points.append((x_grid[xi], y_grid[yi]))
        
        # Добавляем точки пересечения
        scatter_intersect = go.Scatter(
            x=[p[0] for p in intersection_points],
            y=[p[1] for p in intersection_points],
            mode='markers',
            marker=dict(
                size=12,
                color='green',
                symbol='circle',
                line=dict(width=2, color='white')
            ),
            name='Возможные решения',
            showlegend=len(intersection_points) > 0
        )
        
        # Создаем кадр
        frame = go.Frame(
            data=[contour1, scatter2, contour1_copy, scatter_intersect],
            name=f'frame_{i}',
            traces=[0, 1, 2, 3]  # Указываем какие следы в каких subplots
        )
        frames.append(frame)
    
    # Первый кадр
    a_first = a_values[0]
    Z1_first = (X + a_first)**4 - Y**4 - 0.5*a_first**2*(X + a_first)**2 + 0.5*a_first**2*Y**2
    y_line_first = a_first * x_line + a_first/2
    
    # Добавляем первый график (уравнение 1)
    fig.add_trace(
        go.Contour(
            z=Z1_first,
            x=x_grid,
            y=y_grid,
            contours=dict(
                coloring='lines',
                start=0,
                end=0,
                size=0
            ),
            line_width=3,
            colorscale='Reds',
            name=f'Ур.1 при a={a_first:.1f}',
            showscale=False,
            line_color='red'
        ),
        row=1, col=1
    )
    
    # Добавляем второй график (уравнение 2)
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line_first,
            mode='lines',
            line=dict(color='blue', width=3),
            name=f'Ур.2 при a={a_first:.1f}'
        ),
        row=1, col=2
    )
    
    # Добавляем те же данные в оба subplot для анимации
    fig.add_trace(
        go.Contour(
            z=Z1_first,
            x=x_grid,
            y=y_grid,
            contours=dict(
                coloring='lines',
                start=0,
                end=0,
                size=0
            ),
            line_width=3,
            colorscale='Reds',
            showscale=False,
            line_color='red'
        ),
        row=1, col=2
    )
    
    fig.frames = frames
    
    # Настройка анимации
    animation_settings = dict(
        frame=dict(duration=200, redraw=True),
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
                label="🔄 Сброс",
                method="animate",
                args=[["frame_0"], dict(mode="immediate", frame=dict(duration=0))]
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
    
    # Ползунок для параметра a
    sliders = [dict(
        steps=[dict(
            method='animate',
            args=[
                [f'frame_{k}'],
                dict(mode='immediate', frame=dict(duration=0))
            ],
            label=f'a = {a_values[k]:.2f}'
        ) for k in range(len(a_values))],
        active=0,
        currentvalue=dict(
            font=dict(size=14),
            prefix="Параметр a = ",
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
            text="Анимация системы уравнений",
            font=dict(size=20, color='darkblue'),
            x=0.5,
            xanchor='center',
            y=0.95
        ),
        height=550,
        template='plotly_white',
        updatemenus=updatemenus,
        sliders=sliders,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )
    
    # Настройка осей
    fig.update_xaxes(title_text="x", range=[x_min, x_max], row=1, col=1)
    fig.update_yaxes(title_text="y", range=[y_min, y_max], row=1, col=1)
    
    fig.update_xaxes(title_text="x", range=[x_min, x_max], row=1, col=2)
    fig.update_yaxes(title_text="y", range=[y_min, y_max], row=1, col=2)
    
    return fig

# Функция для анимации одиночных функций
def animate_single_function(func_str, a_min, a_max, x_min, x_max, steps=40):
    """Анимация одиночной функции с параметром a"""
    
    # Простая функция для вычисления
    def calculate_y(x_vals, a_val, func):
        y_vals = []
        for x in x_vals:
            try:
                # Безопасное вычисление
                expr = func.replace('a', str(a_val)).replace('x', str(x))
                expr = expr.replace('|x|', f'abs({x})')
                expr = expr.replace('|a|', f'abs({a_val})')
                expr = expr.replace('x²', f'({x}**2)').replace('x^2', f'({x}**2)')
                expr = expr.replace('sin', 'np.sin').replace('cos', 'np.cos').replace('tan', 'np.tan')
                expr = expr.replace('exp', 'np.exp').replace('log', 'np.log').replace('sqrt', 'np.sqrt')
                
                # Вычисляем
                result = eval(expr, {"np": np, "__builtins__": {}}, {})
                y_vals.append(float(result) if isinstance(result, (int, float)) else 0.0)
            except:
                y_vals.append(0.0)
        return np.array(y_vals)
    
    a_values = np.linspace(a_min, a_max, steps)
    x_values = np.linspace(x_min, x_max, 300)
    
    fig = go.Figure()
    
    # Создаем кадры
    frames_data = []
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for i, a in enumerate(a_values):
        y_values = calculate_y(x_values, a, func_str)
        
        # Выбираем цвет на основе индекса
        color_idx = i % len(colors)
        
        frame = go.Frame(
            data=[go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines',
                line=dict(
                    color=colors[color_idx],
                    width=3,
                    dash='solid'
                ),
                name=f'a = {a:.2f}',
                fill='tozeroy',
                fillcolor=f'rgba{tuple(int(colors[color_idx].lstrip("#")[j:j+2], 16) for j in (0, 2, 4)) + (0.1,)}'
            )],
            name=f'frame_{i}'
        )
        frames_data.append(frame)
    
    # Первый кадр
    a_first = a_values[0]
    y_first = calculate_y(x_values, a_first, func_str)
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_first,
        mode='lines',
        line=dict(color='blue', width=3),
        name=f'a = {a_first:.2f}',
        fill='tozeroy',
        fillcolor='rgba(0, 0, 255, 0.1)'
    ))
    
    fig.frames = frames_data
    
    # Настройка анимации
    animation_settings = dict(
        frame=dict(duration=150, redraw=True),
        fromcurrent=True,
        mode='immediate',
        transition=dict(duration=100)
    )
    
    # Простые кнопки управления
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="▶️", method="animate", args=[None, animation_settings]),
                dict(label="⏸️", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])
            ],
            x=0.1,
            y=0,
            xanchor="right",
            yanchor="bottom"
        )],
        sliders=[dict(
            steps=[dict(
                method="animate",
                args=[[f'frame_{k}'], dict(mode="immediate", frame=dict(duration=0))],
                label=f'{a_values[k]:.2f}'
            ) for k in range(len(a_values))],
            active=0,
            currentvalue=dict(prefix="a = ", visible=True),
            pad=dict(t=30)
        )]
    )
    
    # Настройка макета
    fig.update_layout(
        title=dict(
            text=f"Анимация функции: f(x) = {func_str}",
            font=dict(size=18),
            x=0.5
        ),
        xaxis_title="x",
        yaxis_title="f(x)",
        height=500,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

# Основной интерфейс
st.title("🎬 Анимация системы уравнений")

# Создаем вкладки
tab1, tab2 = st.tabs(["🔢 Система из задачи", "📊 Одиночные функции"])

with tab1:
    st.header("Система уравнений из задачи")
    
    st.markdown("""
    ### Условие:
    \[
    \\begin{cases} 
    (x + a)^4 - y^4 - 0.5a^2(x + a)^2 + 0.5a^2y^2 = 0 \\\\ 
    y = ax + \\frac{a}{2}
    \\end{cases}
    \]
    
    ### Что анимируем:
    1. **Красная кривая** - график первого уравнения
    2. **Синяя прямая** - график второго уравнения
    3. **Зеленые точки** - возможные решения системы
    """)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Настройки анимации")
        
        st.markdown("**Диапазон параметра a:**")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            a_min_sys = st.number_input("Минимум a", -10.0, 10.0, 0.1, 0.5, key="a_min_sys")
        with col_a2:
            a_max_sys = st.number_input("Максимум a", -10.0, 10.0, 5.0, 0.5, key="a_max_sys")
        
        st.markdown("**Диапазон x:**")
        col_x1, col_x2 = st.columns(2)
        with col_x1:
            x_min_sys = st.number_input("x мин", -20.0, 20.0, -10.0, 1.0, key="x_min_sys")
        with col_x2:
            x_max_sys = st.number_input("x макс", -20.0, 20.0, 10.0, 1.0, key="x_max_sys")
        
        st.markdown("**Диапазон y:**")
        col_y1, col_y2 = st.columns(2)
        with col_y1:
            y_min_sys = st.number_input("y мин", -20.0, 20.0, -10.0, 1.0, key="y_min_sys")
        with col_y2:
            y_max_sys = st.number_input("y макс", -20.0, 20.0, 10.0, 1.0, key="y_max_sys")
        
        steps_sys = st.slider("Количество кадров", 10, 50, 25, key="steps_sys")
        
        st.markdown("---")
        btn_system = st.button("🎬 Запустить анимацию системы", 
                              type="primary", 
                              use_container_width=True)
        
        # Инструкция
        with st.expander("💡 Как интерпретировать графики"):
            st.markdown("""
            1. **Левый график**: показывает только первое уравнение
            2. **Правый график**: показывает оба уравнения вместе
            3. **Точки пересечения** (зеленые) - где система может иметь решения
            4. **Изменяйте параметр a** чтобы увидеть как меняются графики
            """)
    
    with col2:
        if btn_system:
            with st.spinner("Создаем анимацию системы..."):
                fig_system = animate_task_system(
                    a_min_sys, a_max_sys,
                    x_min_sys, x_max_sys,
                    y_min_sys, y_max_sys,
                    steps_sys
                )
                st.plotly_chart(fig_system, use_container_width=True, config={'displayModeBar': True})
            
            # Дополнительная информация
            st.markdown("---")
            st.subheader("📈 Анализ при текущем a")
            
            # Пример вычисления для среднего a
            a_mid = (a_min_sys + a_max_sys) / 2
            
            st.info(f"""
            **При a = {a_mid:.2f}:**
            
            Второе уравнение: **y = {a_mid:.2f}x + {a_mid/2:.2f}**
            
            Первое уравнение представляет собой сложную кривую 4-й степени.
            
            **Наблюдайте:**
            - Как меняется форма красной кривой при изменении a
            - Где пересекаются красная кривая и синяя прямая
            - При каких a появляются/исчезают точки пересечения
            """)

with tab2:
    st.header("Анимация одиночных функций")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Выберите функцию")
        
        # Простые примеры функций
        example_funcs_simple = {
            "a * sin(x)": "a * sin(x)",
            "a * cos(x)": "a * cos(x)", 
            "a * x²": "a * x**2",
            "sin(a * x)": "sin(a * x)",
            "a * |x|": "a * abs(x)",
            "exp(-a * x)": "exp(-a * x)",
            "x² - a": "x**2 - a",
            "a * log(|x| + 1)": "a * log(abs(x) + 1)",
            "1/(x² + a)": "1/(x**2 + a)",
            "sqrt(|x| + a)": "sqrt(abs(x) + a)"
        }
        
        selected_func = st.selectbox(
            "Пример функции:",
            list(example_funcs_simple.keys()),
            key="func_select"
        )
        
        func_input = st.text_input(
            "Или введите свою функцию f(x):",
            value=example_funcs_simple[selected_func],
            key="func_input_single",
            help="Используйте x как переменную, a как параметр"
        )
        
        st.subheader("Параметры анимации")
        
        col_a1f, col_a2f = st.columns(2)
        with col_a1f:
            a_min_func = st.number_input("a мин", -10.0, 10.0, -3.0, 0.5, key="a_min_f")
        with col_a2f:
            a_max_func = st.number_input("a макс", -10.0, 10.0, 3.0, 0.5, key="a_max_f")
        
        col_x1f, col_x2f = st.columns(2)
        with col_x1f:
            x_min_func = st.number_input("x мин", -20.0, 20.0, -5.0, 0.5, key="x_min_f")
        with col_x2f:
            x_max_func = st.number_input("x макс", -20.0, 20.0, 5.0, 0.5, key="x_max_f")
        
        steps_func = st.slider("Количество кадров", 10, 100, 30, key="steps_f")
        
        st.markdown("---")
        btn_function = st.button("🎬 Анимировать функцию", 
                                type="primary", 
                                use_container_width=True,
                                key="btn_func")
        
        # Справка по синтаксису
        with st.expander("📝 Синтаксис функций"):
            st.markdown("""
            **Доступные функции:**
            - `sin(x)`, `cos(x)`, `tan(x)`
            - `exp(x)`, `log(x)`, `sqrt(x)`
            - `abs(x)` или `|x|` - модуль
            
            **Примеры:**
            - `a * sin(2*x)`
            - `x**3 - a*x`
            - `exp(-a*x**2)`
            - `sin(a*x) * cos(x)`
            """)
    
    with col2:
        if btn_function and func_input:
            with st.spinner("Создаем анимацию функции..."):
                fig_function = animate_single_function(
                    func_input,
                    a_min_func, a_max_func,
                    x_min_func, x_max_func,
                    steps_func
                )
                st.plotly_chart(fig_function, use_container_width=True, config={'displayModeBar': True})
            
            # Информация о функции
            st.markdown("---")
            st.subheader("ℹ️ О функции")
            
            # Показываем пример значений
            try:
                x_sample = np.linspace(-3, 3, 5)
                a_sample = (a_min_func + a_max_func) / 2
                
                st.write(f"**Пример вычислений при a = {a_sample:.2f}:**")
                for x in x_sample:
                    try:
                        expr = func_input.replace('a', str(a_sample)).replace('x', str(x))
                        expr = expr.replace('|x|', f'abs({x})').replace('|a|', f'abs({a_sample})')
                        expr = expr.replace('x²', f'({x}**2)').replace('x^2', f'({x}**2)')
                        expr = expr.replace('sin', 'np.sin').replace('cos', 'np.cos').replace('tan', 'np.tan')
                        expr = expr.replace('exp', 'np.exp').replace('log', 'np.log').replace('sqrt', 'np.sqrt')
                        
                        result = eval(expr, {"np": np, "__builtins__": {}}, {})
                        st.write(f"f({x:.1f}) = {float(result):.4f}")
                    except:
                        st.write(f"f({x:.1f}) = не вычисляется")
            except:
                st.write("Не удалось показать примеры вычислений")

# Футер
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
Анимация математических графиков • Используйте ползунок или кнопки для управления анимацией
</div>
""", unsafe_allow_html=True)

# CSS стили
st.markdown("""
<style>
    .stButton > button {
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-size: 16px;
        font-weight: 600;
    }
    
    .stNumberInput input {
        text-align: center;
    }
    
    /* Анимация для кнопок */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .stButton > button[data-testid="baseButton-primary"] {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)
