# lab3_open_text_model.py
import streamlit as st
import re
import math
from collections import Counter
import matplotlib.pyplot as plt

# ======== основная функция для страницы ========
def main_page():
    st.title("Лабораторная работа №3 — Модель открытого текста")
    st.write("Цель: проведение частотного анализа и определение зависимости энтропии k-грамм.")

    text = st.text_area("Введите или вставьте текст (только русский алфавит):", height=300)

    if st.button("Рассчитать энтропию"):
        if not text.strip():
            st.warning("Введите текст для анализа.")
            return

        # Проверка языка — разрешены только русские буквы, пробелы и знаки препинания
        if re.search(r"[a-zA-Z]", text):
            st.error("Обнаружены латинские символы! Введите текст только на русском языке.")
            return

        # Предобработка текста
        clean_text = re.sub(r"[^а-яё]", "", text.lower())

        if len(clean_text) < 2:
            st.warning("Текст должен содержать хотя бы несколько русских букв.")
            return

        results = []
        for k in range(1, 6):
            hk = entropy_kgram(clean_text, k)
            results.append(hk / k)

        # Отображение таблицы
        st.subheader("Результаты вычислений:")
        for i, val in enumerate(results, start=1):
            st.write(f"H{i}(T)/{i} = {val:.5f}")

        # Построение графика
        fig, ax = plt.subplots()
        ax.plot(range(1, 6), results, marker='o')
        ax.set_xlabel("k")
        ax.set_ylabel("Hk(T)/k")
        ax.set_title("Зависимость энтропии Hk(T)/k от k")
        st.pyplot(fig)

# ======== функция вычисления энтропии k-грамм ========
def entropy_kgram(text: str, k: int) -> float:
    """
    Вычисляет энтропию k-грамм для данного текста.
    text — строка (только русские буквы),
    k — длина k-граммы
    """
    if len(text) < k:
        return 0.0

    kgrams = [text[i:i + k] for i in range(len(text) - k + 1)]
    total = len(kgrams)
    freq = Counter(kgrams)

    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy
