import streamlit as st
import pandas as pd


def integer_sqrt(n):
    """Вычисление целого квадратного корня методом Ньютона."""
    if n < 0: return 0
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

def gcd(a, b):
    """Нахождение НОД (Алгоритм Евклида)."""
    while b:
        a, b = b, a % b
    return a


def run_quadratic_sieve_with_overlay(m, mods):
    """Исправленная логика: возвращает 3 значения, как и ожидает интерфейс."""
    residues_all = {mod: set((x * x) % mod for x in range(mod)) for mod in mods}
    
    # 1. Подготовка индивидуальных решет (для маленьких таблиц)
    individual_logs = []
    for mod in mods:
        x_vals = list(range(mod))
        x_sq_mod = [(x * x) % mod for x in x_vals]
        z_mod = [(v - m) % mod for v in x_sq_mod]
        s_marks = ["Подходит" if val in residues_all[mod] else "Отказ" for val in z_mod]
        individual_logs.append({
            "mod": mod,
            "df": pd.DataFrame({"x": x_vals, "Z mod": z_mod, "Статус": s_marks})
        })

    # 2. Подготовка таблицы наложения (для большой таблицы)
    start_x = integer_sqrt(m) + 1
    overlay_data = []
    for x in range(start_x, start_x + 30):
        z = x * x - m
        row = {"x": x, "Z = x² - m": z}
        is_good_overall = True
        for mod in mods:
            z_mod = z % mod
            passed = z_mod in residues_all[mod]
            row[f"S{mod}"] = "Подходит" if passed else "Отказ"
            if not passed:
                is_good_overall = False
        row["Результат"] = "КАНДИДАТ" if is_good_overall else "ОТСЕЯНО"
        overlay_data.append(row)

    overlay_df = pd.DataFrame(overlay_data)

    # 3. Поиск результата
    result = None
    for x in range(start_x, start_x + 20000):
        z = x * x - m
        if z < 0: continue
        if all((z % mod) in residues_all[mod] for mod in mods):
            y = integer_sqrt(z)
            if y * y == z:
                result = (x, y, x + y, x - y)
                break

    return individual_logs, overlay_df, result

def run_pollard_rho(m, x0_1, x0_2):
    """Логика ро-метода Полларда."""
    x1, x2 = x0_1, x0_2
    history = []
    step = 0
    result = None

    while step < 1000: # Защита от бесконечного цикла
        step += 1
        x1 = (x1 * x1 + 1) % m
        # x2 идет в два раза быстрее (f(f(x)))
        x2 = ((x2 * x2 + 1) * (x2 * x2 + 1) + 1) % m
        diff = abs(x1 - x2)
        d = gcd(diff, m)
        
        history.append({
            "Шаг": step, "x1": x1, "x2": x2, "diff": diff, "НОД": d
        })

        if 1 < d < m:
            result = (d, m // d)
            break
        if d == m:
            break
            
    return pd.DataFrame(history), result


def lab6():
    st.title("Лабораторная работа №6 — Методы факторизации")
    st.write("**Цель:** Изучение алгоритмов разложения числа на множители (Квадратичное решето и метод Полларда).")

    tab1, tab2 = st.tabs(["Квадратичное решето", "ρ-метод Полларда"])

    with tab1:
        st.subheader("Метод квадратичного решета")
        col1, col2 = st.columns([1, 3])
        with col1:
            m_qs = st.number_input("Число m:", value=512327, key="m_qs")
            a = st.number_input("Модуль a:", value=3)
            b = st.number_input("Модуль b:", value=5)
            c = st.number_input("Модуль c:", value=7)
        
        if st.button("Выполнить QS"):
            # Теперь здесь ровно 3 переменные: logs, overlay_df, res
            logs, overlay_df, res = run_quadratic_sieve_with_overlay(m_qs, [a, b, c])
            
            st.write("### 1. Индивидуальные решета модулей")
            cols = st.columns(len(logs))
            for i, log in enumerate(logs):
                with cols[i]:
                    st.write(f"Модуль {log['mod']}")
                    st.dataframe(log['df'], hide_index=True)

            st.write("### 2. Визуализация наложения решет")
            
            # Функция для раскраски строк
            def style_rows(row):
                color = '#d9ead3' if row['Результат'] == 'КАНДИДАТ' else '#f4cccc'
                return [f'background-color: {color}'] * len(row)

            st.dataframe(overlay_df.style.apply(style_rows, axis=1), use_container_width=True)
            
            if res:
                st.success(f"Успех! Найдено при x = {res[0]}, y = {res[1]}")
                st.write(f"Делители: **p = {res[2]}, q = {res[3]}**")
            else:
                st.error("Решение не найдено.")

    with tab2:
        st.subheader("ρ-метод Полларда")
        col1, col2 = st.columns([1, 2])
        with col1:
            m_pr = st.number_input("Число m для факторизации (Pollard):", value=512327, key="m_pr")
            x0_1 = st.number_input("x0 для 1-й послед.:", value=2)
            x0_2 = st.number_input("x0 для 2-й послед.:", value=2)
            
        if st.button("Выполнить Pollard"):
            df_history, res = run_pollard_rho(m_pr, x0_1, x0_2)
            st.write("**Ход решения:**")
            st.dataframe(df_history, use_container_width=True, hide_index=True)
            
            if res:
                st.success(f"Успех! Найден делитель d={res[0]}. Факторизация: **{m_pr} = {res[0]} × {res[1]}**")
            else:
                st.error("Метод зациклился или не нашел решение. Попробуйте другие x0.")

# Для запуска отдельно (тестирование)
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    lab6()