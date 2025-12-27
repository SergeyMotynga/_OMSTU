from laboratory.lab_1.lab_1_the_ceaser_cipher import main_page as caesar_page
from laboratory.lab_2.lab_2_the_affine_cipher import main_page as affine_page
from laboratory.lab_3.lab_3_open_text_model import main_page as open_text_model_page
from laboratory.lab_4.lab4_rsa import main_page as RSA_4
from laboratory.lab_5.lab_5 import lab5
from laboratory.lab_6.lab_6 import lab6
from laboratory.lab_7.lab_7 import lab7
import streamlit as st

# Настройка страницы
st.set_page_config(page_title="Криптографические методы", layout="wide")

# Боковая панель для навигации
st.sidebar.title("🔐 Лабораторные работы")
st.sidebar.markdown("Выберите лабораторную работу:")

# Выбор лабораторной работы
lab_work = st.sidebar.radio(
    "Лабораторные работы:",
    ["Шифр Цезаря", "Аффинный шифр", "Модель открытого текста", "RSA_4", "Криптоанализ RSA", "Методы факторизации", "Дискретное логарифмирование"]
)

# Отображение выбранной страницы
if lab_work == "Шифр Цезаря":
    caesar_page()
elif lab_work == "Аффинный шифр":
    affine_page()
elif lab_work == "Модель открытого текста":
    open_text_model_page()
elif lab_work == "RSA_4":
    RSA_4()
elif lab_work == "Криптоанализ RSA":
    lab5()
elif lab_work == "Методы факторизации":
    lab6()
elif lab_work == "Дискретное логарифмирование":
    lab7()