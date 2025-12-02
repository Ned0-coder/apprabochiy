import streamlit as st
import plotly.graph_objects as go
import numpy as np
import re
from plotly.subplots import make_subplots

# Настройка страницы
st.set_page_config(page_title="Визуализация функций и систем", layout="wide")

# Функция для обработки модулей
def parse_absolute_values(expr):
    """Заменяет |x| на abs(x)"""
    if '|' not in expr:
        return expr
    
    expr = expr.replace('|x|', 'abs(x)')
    expr = expr.replace('|a|', 'abs(a)')
    expr = expr.replace('|y|', 'abs(y)')
    
    pattern = r'\|([^|]+)\|'
    def replace_match(match):
        inner = match.group(1).strip()
        return f'abs({inner})'
    
    while '|' in expr:
        new_expr = re.sub(pattern, replace_match, expr)
        if new_expr == expr:
            break
        expr = new_expr
    
    return expr

def calculate_point(expr, x_val, y_val=None, a_val=None, b_val=None):
    """Вычисляет значение выражения в точке"""
    try:
        expr_parsed = parse_absolute_values(expr)
        
        safe_dict = {
            'x': x_val,
            'y': y_val if y_val is not None else 0,
            'a': a_val if a_val is not None else 1,
            'b': b_val if b_val is not None else 1,
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
        
        result = eval(expr_parsed, {"__builtins__": {}}, safe_dict)
        return float(result) if isinstance(result, (int, float)) else 0.0
    except:
        return 0.0

# Функция для создания графика системы уравнений
def plot_system_2d(eq1, eq2, x_range, y_range, a_val, b_val):
    """Строит график системы двух уравнений в 2D"""
    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)
    
    # Вычисляем значения уравнений
    Z1 = np.zeros_like(X)
    Z2 = np.zeros_like(X)
    
    for i in range(len(x)):
        for j in range(len(y)):
            Z1[j, i] = calculate_point(eq1, X[j, i], Y[j, i], a_val, b_val)
            Z2[j, i] = calculate_point(eq2, X[j, i], Y[j, i], a_val, b_val)
    
    # Создаем график
    fig = make_subplots(rows=1, cols=1)
    
    # Линии уровня (нули уравнений)
    fig.add_trace(go.Contour(
        z=Z1,
        x=x,
        y=y,
        contours=dict(
            coloring='lines',
            start=0,
            end=0,
            size=0,
            showlabels=True
        ),
        line_width=2,
        colorscale='Reds',
        name=f'{eq1} = 0',
        showscale=False
    ))
    
    fig.add_trace(go.Contour(
        z=Z2,
        x=x,
        y=y,
        contours=dict(
            coloring='lines',
            start=0,
            end=0,
            size=0,
            showlabels=True
        ),
        line_width=2,
        colorscale='Blues',
        name=f'{eq2} = 0',
        showscale=False
    ))
    
    # Находим приближенные решения
    solutions = []
    threshold = 0.1
    
    for i in range(1, len(x)-1):
        for j in range(1, len(y)-1):
            if abs(Z1[j, i]) < threshold and abs(Z2[j, i]) < threshold:
                solutions.append((x[i], y[j]))
    
    if solutions:
        sol_x, sol_y = zip(*solutions)
        fig.add_trace(go.Scatter(
            x=sol_x,
            y=sol_y,
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Решение системы'
        ))
    
    fig.update_layout(
        title=f'Система уравнений (a={a_val}, b={b_val})',
        xaxis_title='x',
        yaxis_title='y',
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig, solutions[:5]  # Возвращаем первые 5 решений

# Функция для анимации системы
def animate_system(eq1, eq2, x_range, y_range, a_range, b_range, steps=30):
    """Создает анимацию изменения системы"""
    a_values = np.linspace(a_range[0], a_range[1], steps)
    b_values = np.linspace(b_range[0], b_range[1], steps)
    
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    
    frames = []
    for i, (a, b) in enumerate(zip(a_values, b_values)):
        Z1 = np.zeros_like(X)
        Z2 = np.zeros_like(X)
        
        for xi in range(len(x)):
            for yi in range(len(y)):
                Z1[yi, xi] = calculate_point(eq1, X[yi, xi], Y[yi, xi], a, b)
                Z2[yi, xi] = calculate_point(eq2, X[yi, xi], Y[yi, xi], a, b)
        
        frame = go.Frame(
            data=[
                go.Contour(
                    z=Z1, x=x, y=y,
                    contours=dict(coloring='lines', start=0, end=0, size=0),
                    line_width=2, colorscale='Reds',
                    showscale=False
                ),
                go.Contour(
                    z=Z2, x=x, y=y,
                    contours=dict(coloring='lines', start=0, end=0, size=0),
                    line_width=2, colorscale='Blues',
                    showscale=False
                )
            ],
            name=f'frame_{i}'
        )
        frames.append(frame)
    
    # Первый кадр
    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )
    
    # Настройка анимации
    fig.update_layout(
        title=f'Анимация системы: {eq1} = 0 и {eq2} = 0',
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {'label': '▶️', 'method': 'animate', 'args': [None]},
                {'label': '⏸️', 'method': 'animate', 'args': [[None]]}
            ]
        }],
        sliders=[{
            'steps': [
                {'args': [[f'frame_{k}'], {'frame': {'duration': 0}}],
                 'label': f'a={a_values[k]:.1f}, b={b_values[k]:.1f}',
                 'method': 'animate'} for k in range(len(a_values))
            ]
        }]
    )
    
    return fig

# Основной интерфейс
st.title("📈 Визуализация функций и систем уравнений")

# Создаем вкладки
tab1, tab2, tab3 = st.tabs(["📊 Одна функция", "⚖️ Система уравнений", "🎬 Анимация системы"])

with tab1:
    st.header("Функция одной переменной")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        example_funcs = {
            "a * |x|": "a * |x|",
            "sin(a * x)": "sin(a * x)",
            "a * x²": "a * x**2",
            "exp(-a * |x|)": "exp(-a * |x|)",
            "|x - a|": "|x - a|"
        }
        
        selected = st.selectbox("Пример:", list(example_funcs.keys()), key="single")
        func_input = st.text_input("f(x) =", value=example_funcs[selected], key="func_input")
        
        a_val = st.slider("Параметр a", -5.0, 5.0, 1.0, 0.1, key="a_single")
        x_min, x_max = st.slider("Диапазон x", -20.0, 20.0, (-10.0, 10.0), key="x_range_single")
    
    with col2:
        if func_input:
            x_values = np.linspace(x_min, x_max, 500)
            y_values = [calculate_point(func_input, x, a_val=a_val) for x in x_values]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_values, y=y_values,
                mode='lines',
                line=dict(color='blue', width=3),
                name=f'a = {a_val}'
            ))
            
            fig.update_layout(
                title=f'f(x) = {func_input}',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Система двух уравнений")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Уравнение 1")
        eq1 = st.text_input("f₁(x, y) = 0", value="x**2 + y**2 - a", key="eq1")
        
        st.subheader("Уравнение 2")
        eq2 = st.text_input("f₂(x, y) = 0", value="x - y - b", key="eq2")
        
        col_a, col_b = st.columns(2)
        with col_a:
            a_val_sys = st.slider("Параметр a", -5.0, 5.0, 4.0, 0.1, key="a_sys")
        with col_b:
            b_val_sys = st.slider("Параметр b", -5.0, 5.0, 0.0, 0.1, key="b_sys")
        
        col_x, col_y = st.columns(2)
        with col_x:
            x_min_sys, x_max_sys = st.slider("Диапазон x", -10.0, 10.0, (-5.0, 5.0), key="x_sys")
        with col_y:
            y_min_sys, y_max_sys = st.slider("Диапазон y", -10.0, 10.0, (-5.0, 5.0), key="y_sys")
    
    with col2:
        if eq1 and eq2:
            fig, solutions = plot_system_2d(
                eq1, eq2,
                (x_min_sys, x_max_sys),
                (y_min_sys, y_max_sys),
                a_val_sys, b_val_sys
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if solutions:
                st.success("Найдены решения:")
                for i, (x_sol, y_sol) in enumerate(solutions, 1):
                    st.write(f"Решение {i}: x ≈ {x_sol:.3f}, y ≈ {y_sol:.3f}")
            else:
                st.info("Решения не найдены в указанном диапазоне")

with tab3:
    st.header("Анимация системы уравнений")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Система для анимации")
        eq1_anim = st.text_input("f₁(x, y) = 0", value="x**2 + y**2 - a", key="eq1_anim")
        eq2_anim = st.text_input("f₂(x, y) = 0", value="y - sin(a*x) - b", key="eq2_anim")
        
        st.subheader("Параметры анимации")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            a_min_anim = st.number_input("a мин", -5.0, 5.0, 1.0, 0.5, key="a_min_anim")
        with col_a2:
            a_max_anim = st.number_input("a макс", -5.0, 5.0, 5.0, 0.5, key="a_max_anim")
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            b_min_anim = st.number_input("b мин", -5.0, 5.0, -1.0, 0.5, key="b_min_anim")
        with col_b2:
            b_max_anim = st.number_input("b макс", -5.0, 5.0, 1.0, 0.5, key="b_max_anim")
        
        col_x1, col_x2 = st.columns(2)
        with col_x1:
            x_min_anim = st.number_input("x мин", -10.0, 10.0, -5.0, 0.5, key="x_min_anim")
        with col_x2:
            x_max_anim = st.number_input("x макс", -10.0, 10.0, 5.0, 0.5, key="x_max_anim")
        
        col_y1, col_y2 = st.columns(2)
        with col_y1:
            y_min_anim = st.number_input("y мин", -10.0, 10.0, -5.0, 0.5, key="y_min_anim")
        with col_y2:
            y_max_anim = st.number_input("y макс", -10.0, 10.0, 5.0, 0.5, key="y_max_anim")
        
        animate_btn = st.button("🎬 Создать анимацию", type="primary", use_container_width=True)
    
    with col2:
        if animate_btn and eq1_anim and eq2_anim:
            with st.spinner("Создаем анимацию..."):
                fig_anim = animate_system(
                    eq1_anim, eq2_anim,
                    (x_min_anim, x_max_anim),
                    (y_min_anim, y_max_anim),
                    (a_min_anim, a_max_anim),
                    (b_min_anim, b_max_anim),
                    steps=20
                )
                
                st.plotly_chart(fig_anim, use_container_width=True)
                st.success("Анимация создана! Используйте кнопки управления.")

# Справка
with st.expander("📚 Справка по синтаксису"):
    st.markdown("""
    ### Доступные функции и операторы:
    
    **Математические функции:**
    - `abs(x)` или `|x|` - модуль
    - `sin(x)`, `cos(x)`, `tan(x)` - тригонометрия
    - `exp(x)`, `log(x)`, `sqrt(x)`
    - `pi` (≈3.14159), `e` (≈2.71828)
    
    **Операторы:**
    - `+`, `-`, `*`, `/` - арифметика
    - `**` - возведение в степень (x² = x**2)
    - `()` - скобки для приоритета
    
    **Переменные и параметры:**
    - `x`, `y` - переменные
    - `a`, `b` - параметры (можно менять)
    
    ### Примеры систем:
    1. Окружность и прямая:
       - `x**2 + y**2 - a = 0`
       - `y - b*x = 0`
    
    2. Параболы:
       - `y - a*x**2 = 0`
       - `x - b*y**2 = 0`
    
    3. Тригонометрическая:
       - `sin(a*x) - y = 0`
       - `cos(b*y) - x = 0`
    """)

# Футер
st.markdown("---")
st.markdown("*Приложение для визуализации функций и систем уравнений*")
