from laboratory.lab_1.lab_1_the_ceaser_cipher import main_page as caesar_page
from laboratory.lab_2.lab_2_the_affine_cipher import main_page as affine_page
from laboratory.lab_3.lab_3_open_text_model import main_page as open_text_model_page
import streamlit as st

# Настройка страницы
st.set_page_config(page_title="Криптографические методы", layout="wide")

# Боковая панель для навигации
st.sidebar.title("🔐 Лабораторные работы")
st.sidebar.markdown("Выберите лабораторную работу:")

# Выбор лабораторной работы
lab_work = st.sidebar.radio(
    "Лабораторные работы:",
    ["Шифр Цезаря", "Аффинный шифр", "Модель открытого текста"]
)

# Отображение выбранной страницы
if lab_work == "Шифр Цезаря":
    caesar_page()
elif lab_work == "Аффинный шифр":
    affine_page()
elif lab_work == "Модель открытого текста":
    open_text_model_page()