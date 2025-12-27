import streamlit as st
import pandas as pd


def integer_sqrt(n):
    """Вычисление целого квадратного корня."""
    if n < 0: return 0
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

def fast_pow(base, exp, mod):
    """
    Алгоритм быстрого возведения в степень по модулю (бинарное возведение).
    """
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:  # Если текущий бит степени равен 1 (число нечетное)
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1    # Сдвиг вправо (деление степени на 2)
    return result


def baby_giant_dlog(a, b, p):
    """
    Алгоритм 'Шаг младенца – шаг великана' для решения a^x ≡ b (mod p).
    """
    k = integer_sqrt(p) + 1
    
    # 1. Шаг младенца: вычисляем y_i = a^(i*k) mod p
    baby_steps = {}
    baby_log = []
    ak = fast_pow(a, k, p)
    current_baby = 1
    
    for i in range(k):
        baby_steps[current_baby] = i
        if i < 100:  # Логируем только начало для таблицы
            baby_log.append({"i": i, "y_i = a^(i*k) mod p": current_baby})
        current_baby = (current_baby * ak) % p
        
    # 2. Шаг великана: ищем z_j = b * a^j mod p
    giant_log = []
    current_giant = b
    result = None
    
    for j in range(k):
        if j < 100:
            giant_log.append({"j": j, "z_j = b * a^j mod p": current_giant})
            
        if current_giant in baby_steps:
            i = baby_steps[current_giant]
            x = i * k - j
            result = {
                "i": i, 
                "j": j, 
                "k": k, 
                "x": x, 
                "z": current_giant
            }
            break
        current_giant = (current_giant * a) % p
        
    return k, pd.DataFrame(baby_log), pd.DataFrame(giant_log), result

def lab7():
    st.title("Лабораторная работа №7")
    st.subheader("Дискретное логарифмирование (Алгоритм Шэнкса)")
    
    st.write("Решение уравнения: $a^x \\equiv b \\pmod{p}$")
    
    # Ввод данных
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        a = st.number_input("Основание a:", value=2)
    with col_in2:
        b = st.number_input("Результат b:", value=28620)
    with col_in3:
        p = st.number_input("Модуль p:", value=30539)
        
    if st.button("Найти x"):
        k, df_baby, df_giant, res = baby_giant_dlog(a, b, p)
        
        st.write(f"**Шаг сетки k:** {k}")
        
        # Отображение таблиц
        col_tab1, col_tab2 = st.columns(2)
        with col_tab1:
            st.write("**Шаги младенца (первые 100):**")
            st.dataframe(df_baby, hide_index=True, use_container_width=True)
        with col_tab2:
            st.write("**Шаги великана (первые 100):**")
            st.dataframe(df_giant, hide_index=True, use_container_width=True)
            
        if res:
            st.success("### Решение найдено!")
            st.markdown(f"""
            * Совпадение на шаге **j = {res['j']}**
            * Значение в таблице младенца: **i = {res['i']}**
            * Формула: $x = i \\cdot k - j = {res['i']} \\cdot {res['k']} - {res['j']}$
            * **Результат: x = {res['x']}**
            """)
            
            # Проверка
            check_val = fast_pow(a, res['x'], p)
            st.info(f"**Проверка:** {a}^{{{res['x']}}} mod {p} = {check_val}")
            if check_val == b:
                st.write("Значение верно!")
        else:
            st.error("Логарифм не найден. Возможно, решения не существует.")

if __name__ == "__main__":
    lab7()