import streamlit as st
import re
import math
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import io

st.set_page_config(page_title="Лабораторная работа №3", layout="wide")

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

def main_page():
    st.title("Лабораторная работа №3 — Модель открытого текста")
    st.write("**Цель:** проведение частотного анализа и определение зависимости энтропии k-грамм.")
    
    st.markdown("---")
    
    # Выбор способа ввода
    input_method = st.radio(
        "Выберите способ ввода текста:",
        ["Вставить текст", "Загрузить файл"]
    )
    
    text = ""
    
    if input_method == "Вставить текст":
        text = st.text_area(
            "Введите текст (только русский алфавит):",
            height=200,
            placeholder="Вставьте сюда текст для анализа..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Загрузите текстовый файл (.txt)", 
            type=['txt']
        )
        if uploaded_file is not None:
            try:
                text = uploaded_file.read().decode('utf-8')
                st.success(f"Файл успешно загружен! Размер: {len(text)} символов")
            except:
                st.error("Ошибка чтения файла. Убедитесь, что файл в кодировке UTF-8.")
    
    # Кнопка для запуска расчётов
    if st.button("Рассчитать энтропию", type="primary"):
        if not text.strip():
            st.warning("Введите текст для анализа.")
            return
        
        # Проверка на латинские символы
        if re.search(r"[a-zA-Z]", text):
            st.error("Обнаружены латинские символы! Используйте только русский алфавит.")
            return
        
        # Предобработка текста
        clean_text = re.sub(r"[^а-яё]", "", text.lower())
        clean_text = clean_text.replace("ё", "е")
        
        if len(clean_text) < 10:
            st.warning("Текст слишком короткий для анализа (минимум 10 символов).")
            return
        
        # Вычисление энтропии для k от 1 до 5
        results = []
        entropy_values = []
        
        st.subheader("Результаты расчёта энтропии:")
        
        # Вывод результатов в формате, аналогичном сданной работе
        output_text = ""
        for k in range(1, 6):
            hk = entropy_kgram(clean_text, k)
            hk_k = hk / k
            results.append({
                "k": k,
                "H_k": round(hk, 6),
                "H_k/k": round(hk_k, 6)
            })
            entropy_values.append(hk_k)
            
            output_text += f"H_{k} = {hk:.6f}\n"
            output_text += f"H_{k} / {k} = {hk_k:.6f}\n\n"
        
        # Выводим отформатированный текст
        st.code(output_text, language="text")
        
        # Создание DataFrame для таблицы
        df = pd.DataFrame(results)
        
        # Отображение таблицы
        st.subheader("Таблица результатов:")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(df, use_container_width=True, hide_index=True)
            
        with col2:
            st.markdown("##### Скачать результаты:")
            
            # CSV файл
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Скачать CSV",
                data=csv_buffer.getvalue(),
                file_name="entropy_results.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        
        # Построение графика
        st.subheader("График зависимости H_k/k от k")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, 6), entropy_values, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel("k (длина k-граммы)", fontsize=12)
        ax.set_ylabel("H_k/k", fontsize=12)
        ax.set_title("Зависимость энтропии H_k/k от k", fontsize=14, pad=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Настройка осей
        ax.set_xticks(range(1, 6))
        ax.set_xlim(0.5, 5.5)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.pyplot(fig)
        
        with col2:
            # Сохранение графика
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                fig.savefig(tmpfile.name, dpi=300, bbox_inches='tight')
                with open(tmpfile.name, 'rb') as img_file:
                    img_bytes = img_file.read()
            
            st.download_button(
                label="Скачать график",
                data=img_bytes,
                file_name="entropy_plot.png",
                mime="image/png"
            )
        
        # Статистика текста
        st.markdown("---")
        st.subheader("Статистика текста")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Оригинальный текст", f"{len(text)} символов")
        with col2:
            st.metric("После обработки", f"{len(clean_text)} букв")
        with col3:
            unique_chars = len(set(clean_text))
            st.metric("Уникальных букв", unique_chars)