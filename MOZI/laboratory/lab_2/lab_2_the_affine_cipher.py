import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import plotly.express as px

# Русский алфавит и кодировка
# Алфавит: 33 символа (32 буквы + пробел в конце)
RUSSIAN_ALPHABET = 'абвгдежзийклмнопрстуфхцчшщъыьэюя '
ALPHABET_SIZE = 32  # Модуль для вычислений (без пробела)

# Эталонные частоты русского языка (строка по убыванию частоты)
# Пробел представлен как '-', ё объединен с е, ъ объединен с ь
RUSSIAN_TOP_LETTERS = '-оеаитнсрвлкмдпуяызьбгчйхжюшцщэф'

# Таблица эталонных частот русского языка
# Частоты в процентах
RUSSIAN_FREQUENCIES_DICT = {
    ' ': 0.175,  # пробел (представлен как '-')
    'о': 0.090,
    'е': 0.072,  # включая ё
    'а': 0.062,
    'и': 0.062,
    'н': 0.053,
    'т': 0.053,
    'с': 0.045,
    'р': 0.040,
    'в': 0.038,
    'л': 0.035,
    'к': 0.028,
    'м': 0.026,
    'д': 0.025,
    'п': 0.023,
    'у': 0.021,
    'я': 0.018,
    'ы': 0.016,
    'з': 0.016,
    'ь': 0.014,  # включая ъ
    'б': 0.014,
    'г': 0.013,
    'ч': 0.012,
    'й': 0.010,
    'х': 0.009,
    'ж': 0.007,
    'ю': 0.006,
    'ш': 0.006,
    'ц': 0.004,
    'щ': 0.003,
    'э': 0.003,
    'ф': 0.002
}

def char_to_code(char):
    """Преобразование символа в код"""
    # Обрабатываем специальные случаи
    if char == 'ё':
        char = 'е'
    elif char == 'ъ':
        char = 'ь'
    
    if char in RUSSIAN_ALPHABET:
        return RUSSIAN_ALPHABET.index(char)
    return None

def code_to_char(code):
    """Преобразование кода в символ"""
    if 0 <= code < len(RUSSIAN_ALPHABET):
        return RUSSIAN_ALPHABET[code]
    return None

def extended_gcd(a, b):
    """Расширенный алгоритм Евклида"""
    r_i, r_i_plus_1 = a, b
    xi, yi, x_i_plus_1, y_i_plus_1 = 1, 0, 0, 1
    while r_i_plus_1 != 0:
        q = r_i // r_i_plus_1
        r_i, r_i_plus_1 = r_i_plus_1, r_i - q * r_i_plus_1
        xi, x_i_plus_1 = x_i_plus_1, xi - q * x_i_plus_1
        yi, y_i_plus_1 = y_i_plus_1, yi - q * y_i_plus_1
    return r_i, xi, yi

def mod_inverse(a, m):
    """Нахождение обратного элемента по модулю"""
    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
        return None  # Обратный элемент не существует
    if x < 0:
        return x + m
    return x

def affine_encrypt(text, a, b):
    """Шифрование аффинным шифром (по методичке)"""
    encrypted = ""
    for char in text.lower():
        # Обрабатываем только символы из алфавита (без последнего символа - пробела)
        if char in RUSSIAN_ALPHABET[:-1]:
            x = char_to_code(char)
            if x is not None:
                y = (a * x + b) % ALPHABET_SIZE
                encrypted += code_to_char(y)
        else:
            encrypted += char
    return encrypted

def affine_decrypt(text, a, b):
    """Расшифрование аффинным шифром (по методичке)"""
    text = text.lower()
    clean_text = ''
    # Оставляем только символы из алфавита (без последнего символа - пробела)
    for let in text:
        if let in RUSSIAN_ALPHABET[:-1]:
            clean_text += let
    
    decoded_text = ''
    a_inv = mod_inverse(a, ALPHABET_SIZE)
    if a_inv is None:
        return None
    
    for yi in clean_text:
        # Формула расшифрования
        xi = ((RUSSIAN_ALPHABET.index(yi) - b) * a_inv) % ALPHABET_SIZE
        decoded_text += RUSSIAN_ALPHABET[xi]
    
    return decoded_text

def frequency_analysis(text, without_space=True):
    """Частотный анализ текста (по методичке)"""
    text = text.lower()
    clean_text = ''
    # Оставляем только символы из алфавита
    for let in text:
        if let in RUSSIAN_ALPHABET:
            clean_text += let
    
    # Замены по методичке: ё→е, ъ→ь, пробелы→-
    clean_text = clean_text.replace('ё', 'е')
    clean_text = clean_text.replace('ъ', 'ь')
    clean_text = clean_text.replace(' ', '-')
    
    if not clean_text:
        return {}
    
    # Вычисляем частоты
    frec_dict = {}
    letters = list(set(clean_text))
    for l in letters:
        frec_dict[l] = clean_text.count(l) / len(clean_text)
    
    # Удаляем пробел из результатов, если нужно
    if without_space and '-' in frec_dict:
        del frec_dict['-']
    
    return dict(sorted(frec_dict.items(), key=lambda x: -x[1]))

def sort_by_freqs(d):
    """Сортировка словаря частот по убыванию"""
    return ''.join(sorted(d, key=d.get, reverse=True))

def solve_comparison(a, b, m):
    """Решение сравнения вида ax ≡ b (mod m)"""
    # Правильно обрабатываем отрицательные значения
    a = a % m
    b = b % m
    
    gcd_val, x0, _ = extended_gcd(a, m)
    if b % gcd_val != 0:
        return None  # Нет решений
    
    a_reduced = a // gcd_val
    b_reduced = b // gcd_val
    small_mod = m // gcd_val
    
    a_inv = mod_inverse(a_reduced, small_mod)
    if a_inv is None:
        return None
    
    start_x = (a_inv * b_reduced) % small_mod
    solutions = []
    for i in range(gcd_val):
        x = (start_x + i * small_mod) % m
        solutions.append(x)
    
    return solutions

def solve_comparison_system(a1, a2, b1, b2, m):
    """Решение системы сравнений для нахождения ключа (по методичке)
    Система:
    a * a1 + b ≡ b1 (mod m)  где a - ключ, a1 - код открытого текста, b1 - код зашифрованного
    a * a2 + b ≡ b2 (mod m)
    """
    # Вычитаем: a * (a1 - a2) ≡ (b1 - b2) (mod m)
    # Не используем % m здесь, так как solve_comparison сама обработает отрицательные значения
    x_solutions = solve_comparison(a1 - a2, b1 - b2, m)
    if x_solutions is None:
        return None
    
    # Перебираем все решения и проверяем, что a взаимно прост с m
    for a_key in x_solutions:
        # Проверяем, что a взаимно прост с m (необходимо для существования обратного элемента)
        gcd_val, _, _ = extended_gcd(a_key, m)
        if gcd_val == 1:
            # Находим b (ключ)
            b_key = (b1 - a_key * a1) % m
            return a_key, b_key
    
    return None

def main_page():
    """Главная страница лабораторной работы №2"""
    
    st.title("Лабораторная работа №2: Криптоанализ аффинного шифра")
    
    # Краткое описание
    st.markdown("""
    **Цель работы:** проведение криптоанализа сообщения, зашифрованного аффинным шифром, 
    определение ключа шифрования и открытого текста.
    """)
    
    # Создаем вкладки
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Частотный анализ", 
        "Автоматический криптоанализ", 
        "Математические операции",
        "Шифрование/Расшифрование",
        "Теория"
    ])
    
    # Вкладка 1: Частотный анализ
    with tab1:
        st.header("Частотный анализ текста")
        
        col1, col2 = st.columns(2)
        
        with col1:
            input_text = st.text_area("Введите текст для анализа:", height=200,
                                    placeholder="Введите зашифрованный текст здесь...",
                                    key="freq_text")
            
            if st.button("Выполнить частотный анализ", key="analyze_freq"):
                if input_text:
                    # Анализ введенного текста
                    text_freq = frequency_analysis(input_text)
                    
                    if text_freq:
                        # Создаем DataFrame для отображения
                        freq_df = pd.DataFrame(list(text_freq.items()), 
                                             columns=['Символ', 'Частота'])
                        
                        # Добавляем позицию в эталонном списке (чем меньше позиция, тем выше частота)
                        freq_df['Позиция в эталоне'] = freq_df['Символ'].map(
                            lambda x: RUSSIAN_TOP_LETTERS.find(x) if x in RUSSIAN_TOP_LETTERS else -1
                        )
                        
                        st.subheader("Результаты частотного анализа")
                        st.dataframe(freq_df, use_container_width=True)
                        
                        # Визуализация
                        fig = px.bar(freq_df.head(10), x='Символ', y='Частота',
                                    title="Топ-10 символов по частоте в тексте")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Показываем самые частые символы
                        top_chars = list(text_freq.keys())[:5]
                        st.info(f"**Самые частые символы:** {', '.join(top_chars)}")
                        
                        # Сохраняем для использования в других вкладках
                        st.session_state.freq_analysis_result = top_chars
                    else:
                        st.warning("В тексте не найдено символов русского алфавита")
                else:
                    st.warning("Пожалуйста, введите текст для анализа")
        
        with col2:
            st.subheader("Эталонные частоты русского языка")
            st.write("**Порядок символов по убыванию частоты:**")
            st.code(RUSSIAN_TOP_LETTERS)
            
            # Создаем таблицу для отображения с реальными частотами
            ref_data = []
            for i, char in enumerate(RUSSIAN_TOP_LETTERS):
                display_char = ' ' if char == '-' else char
                freq_value = RUSSIAN_FREQUENCIES_DICT.get(char if char != '-' else ' ', 0)
                ref_data.append({
                    'Символ': display_char, 
                    'Позиция': i+1, 
                    'Частота, %': f"{freq_value * 100:.3f}" if freq_value > 0 else "—"
                })
            ref_df = pd.DataFrame(ref_data)
            st.dataframe(ref_df, use_container_width=True)
            
            st.markdown("""
            **Примечание:** 
            - Буквы 'е' и 'ё' объединены
            - Буквы 'ъ' и 'ь' объединены
            - Пробелы представлены как '-' и стоят на первом месте
            - Частоты приведены в процентах
            """)
    
    # Вкладка 2: Автоматический криптоанализ
    with tab2:
        st.header("Автоматический поиск ключа шифрования")
        
        cipher_text = st.text_area("Зашифрованный текст:", height=150,
                                 placeholder="Введите зашифрованный текст для криптоанализа...",
                                 key="crypto_text")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_custom_hypothesis = st.checkbox("Использовать пользовательские гипотезы")
            
            if use_custom_hypothesis:
                st.subheader("Задание гипотез")
                col_a, col_b = st.columns(2)
                with col_a:
                    # Используем эталонные частоты без пробела
                    plain_options = [c if c != '-' else ' ' for c in RUSSIAN_TOP_LETTERS[1:11]]
                    plain_char1 = st.selectbox("Буква открытого текста 1", 
                                             options=plain_options,
                                             index=0)
                    plain_char2 = st.selectbox("Буква открытого текста 2", 
                                             options=plain_options,
                                             index=1)
                with col_b:
                    cipher_char1 = st.text_input("Буква шифртекста 1", max_chars=1, value="")
                    cipher_char2 = st.text_input("Буква шифртекста 2", max_chars=1, value="")
        
        if st.button("Найти возможные ключи", key="find_keys"):
            if cipher_text:
                with st.spinner("Выполняется криптоанализ..."):
                    # Частотный анализ
                    freq = frequency_analysis(cipher_text)
                    if not freq:
                        st.error("В тексте не найдено символов русского алфавита")
                        return
                    
                    # Получаем отсортированные символы по частоте
                    encoded_top_letters = sort_by_freqs(freq)
                    
                    # Используем эталонные частоты (без пробела для криптоанализа)
                    russian_top_letters = RUSSIAN_TOP_LETTERS[1:]
                    
                    # Дополняем список зашифрованных символов недостающими
                    diff = ''.join(set(russian_top_letters).difference(set(encoded_top_letters)))
                    encoded_top_letters += diff
                    
                    st.subheader("Гипотезы о соответствии символов")
                    st.write(f"**Самые частые символы в шифртексте:** {', '.join(encoded_top_letters[:10])}")
                    st.write(f"**Самые частые символы в русском языке:** {', '.join(russian_top_letters[:10])}")
                    
                    # Перебираем гипотезы (по методичке)
                    solutions_found = []
                    
                    if use_custom_hypothesis and cipher_char1 and cipher_char2:
                        # Используем пользовательские гипотезы
                        # Используем прямой индекс из алфавита
                        rlet1_code = RUSSIAN_ALPHABET.index(plain_char1) if plain_char1 in RUSSIAN_ALPHABET else None
                        rlet2_code = RUSSIAN_ALPHABET.index(plain_char2) if plain_char2 in RUSSIAN_ALPHABET else None
                        elet1_code = RUSSIAN_ALPHABET.index(cipher_char1.lower()) if cipher_char1.lower() in RUSSIAN_ALPHABET else None
                        elet2_code = RUSSIAN_ALPHABET.index(cipher_char2.lower()) if cipher_char2.lower() in RUSSIAN_ALPHABET else None
                        
                        if all(x is not None for x in [rlet1_code, rlet2_code, elet1_code, elet2_code]):
                            # Пробуем оба варианта соответствия
                            res1 = solve_comparison_system(rlet1_code, rlet2_code, elet1_code, elet2_code, ALPHABET_SIZE)
                            res2 = solve_comparison_system(rlet1_code, rlet2_code, elet2_code, elet1_code, ALPHABET_SIZE)
                            
                            for res, (e1, e2) in [(res1, (elet1_code, elet2_code)), (res2, (elet2_code, elet1_code))]:
                                if res:
                                    a, b = res
                                    decrypted = affine_decrypt(cipher_text, a, b)
                                    if decrypted:
                                        solutions_found.append({
                                            'a': a, 
                                            'b': b, 
                                            'text': decrypted,
                                            'hypothesis': f"{plain_char1}→{code_to_char(e1)}, {plain_char2}→{code_to_char(e2)}"
                                        })
                    else:
                        # Автоматические гипотезы
                        # Берем первые два символа из эталона и перебираем пары из зашифрованного текста
                        rlet1, rlet2 = russian_top_letters[0], russian_top_letters[1]
                        # Используем прямой индекс из алфавита
                        rlet1_ind, rlet2_ind = RUSSIAN_ALPHABET.index(rlet1), RUSSIAN_ALPHABET.index(rlet2)
                        
                        # Перебираем все пары из зашифрованного текста
                        for i in range(len(encoded_top_letters)):
                            for j in range(i + 1, len(encoded_top_letters)):
                                elet1, elet2 = encoded_top_letters[i], encoded_top_letters[j]
                                # Используем прямой индекс из алфавита
                                elet1_ind, elet2_ind = RUSSIAN_ALPHABET.index(elet1), RUSSIAN_ALPHABET.index(elet2)
                                
                                if all(x is not None for x in [rlet1_ind, rlet2_ind, elet1_ind, elet2_ind]):
                                    # Пробуем оба варианта соответствия
                                    res1 = solve_comparison_system(rlet1_ind, rlet2_ind, elet1_ind, elet2_ind, ALPHABET_SIZE)
                                    res2 = solve_comparison_system(rlet1_ind, rlet2_ind, elet2_ind, elet1_ind, ALPHABET_SIZE)
                                    
                                    # Проверяем первый вариант
                                    if res1:
                                        a, b = res1
                                        decrypted = affine_decrypt(cipher_text, a, b)
                                        if decrypted:
                                            # Проверяем, что этот ключ еще не добавлен
                                            key_exists = any(sol['a'] == a and sol['b'] == b for sol in solutions_found)
                                            if not key_exists:
                                                solutions_found.append({
                                                    'a': a, 
                                                    'b': b, 
                                                    'text': decrypted,
                                                    'hypothesis': f"{code_to_char(rlet1_ind)}→{code_to_char(elet1_ind)}, {code_to_char(rlet2_ind)}→{code_to_char(elet2_ind)}"
                                                })
                                    
                                    # Проверяем второй вариант
                                    if res2:
                                        a, b = res2
                                        decrypted = affine_decrypt(cipher_text, a, b)
                                        if decrypted:
                                            # Проверяем, что этот ключ еще не добавлен
                                            key_exists = any(sol['a'] == a and sol['b'] == b for sol in solutions_found)
                                            if not key_exists:
                                                solutions_found.append({
                                                    'a': a, 
                                                    'b': b, 
                                                    'text': decrypted,
                                                    'hypothesis': f"{code_to_char(rlet1_ind)}→{code_to_char(elet2_ind)}, {code_to_char(rlet2_ind)}→{code_to_char(elet1_ind)}"
                                                })
                    
                    # Показываем результаты
                    if solutions_found:
                        st.success(f"Найдено {len(solutions_found)} возможных ключей")
                        
                        for i, sol in enumerate(solutions_found):
                            with st.expander(f"Ключ {i+1}: a={sol['a']}, b={sol['b']} ({sol['hypothesis']})"):
                                st.write(f"**Расшифрованный текст:**")
                                st.text(sol['text'][:500] + ("..." if len(sol['text']) > 500 else ""))
                                
                                if st.button(f"Выбрать этот ключ", key=f"select_{i}"):
                                    st.session_state.selected_key = (sol['a'], sol['b'])
                                    st.success(f"Ключ сохранен: a={sol['a']}, b={sol['b']}")
                    else:
                        st.warning("Не удалось найти подходящие ключи. Попробуйте другой текст или метод.")
            else:
                st.warning("Пожалуйста, введите зашифрованный текст")
    
    # Вкладка 3: Математические операции
    with tab3:
        st.header("Математические операции")
        
        # Выбор модуля
        mod_value = st.number_input("Модуль (mod):", min_value=2, value=32, key="mod_value")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Нахождение обратного элемента")
            a_inv = st.number_input("Число a:", min_value=1, max_value=mod_value-1, value=3, key="inv_a")
            
            if st.button("Найти обратный элемент", key="find_inv"):
                inv = mod_inverse(a_inv, mod_value)
                if inv is not None:
                    st.success(f"Обратный элемент к {a_inv} по модулю {mod_value}: {inv}")
                    st.write(f"Проверка: {a_inv} × {inv} = {a_inv * inv} ≡ {(a_inv * inv) % mod_value} (mod {mod_value})")
                else:
                    st.error(f"Обратный элемент к {a_inv} по модулю {mod_value} не существует")
        
        with col2:
            st.subheader("Решение сравнения")
            a_comp = st.number_input("a:", value=3, key="comp_a")
            b_comp = st.number_input("b:", value=5, key="comp_b")
            
            if st.button("Решить сравнение", key="solve_comp"):
                solutions = solve_comparison(a_comp, b_comp, mod_value)
                if solutions is not None:
                    st.success(f"Решения сравнения {a_comp}x ≡ {b_comp} (mod {mod_value}):")
                    st.write(f"Количество решений: {len(solutions)}")
                    st.write(f"Решения: {solutions}")
                else:
                    st.error("Сравнение не имеет решений")
        
        st.subheader("Решение системы сравнений")
        st.markdown("""
        **Система сравнений:**
        ```
        a₁ × x + y ≡ b₁ (mod m)
        a₂ × x + y ≡ b₂ (mod m)
        ```
        где x и y - неизвестные
        """)
        
        col_sys1, col_sys2 = st.columns(2)
        with col_sys1:
            a1_sys = st.number_input("a₁:", value=5, key="sys_a1")
            b1_sys = st.number_input("b₁:", value=10, key="sys_b1")
        with col_sys2:
            a2_sys = st.number_input("a₂:", value=7, key="sys_a2")
            b2_sys = st.number_input("b₂:", value=12, key="sys_b2")
        
        if st.button("Решить систему сравнений", key="solve_sys"):
            result = solve_comparison_system(a1_sys, a2_sys, b1_sys, b2_sys, mod_value)
            if result is not None:
                x_sol, y_sol = result
                st.success(f"Решение системы сравнений:")
                st.write(f"x = {x_sol}, y = {y_sol}")
                st.write(f"Проверка:")
                st.write(f"{a1_sys} × {x_sol} + {y_sol} = {a1_sys * x_sol + y_sol} ≡ {(a1_sys * x_sol + y_sol) % mod_value} (mod {mod_value}) [должно быть {b1_sys}]")
                st.write(f"{a2_sys} × {x_sol} + {y_sol} = {a2_sys * x_sol + y_sol} ≡ {(a2_sys * x_sol + y_sol) % mod_value} (mod {mod_value}) [должно быть {b2_sys}]")
            else:
                st.error("Система сравнений не имеет решений")
        
        st.subheader("Проверка взаимной простоты")
        num1 = st.number_input("Первое число:", min_value=1, value=15, key="gcd_a")
        num2 = st.number_input("Второе число:", min_value=1, value=32, key="gcd_b")
        
        if st.button("Проверить НОД", key="check_gcd"):
            gcd_val, x, y = extended_gcd(num1, num2)
            st.write(f"НОД({num1}, {num2}) = {gcd_val}")
            if gcd_val == 1:
                st.success("Числа взаимно просты")
            else:
                st.warning("Числа не взаимно просты")
            st.write(f"Коэффициенты Безу: {num1}×({x}) + {num2}×({y}) = {gcd_val}")
    
    # Вкладка 4: Шифрование/Расшифрование
    with tab4:
        st.header("Шифрование и расшифрование")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Шифрование")
            plain_text = st.text_area("Исходный текст:", height=100,
                                    placeholder="Введите текст для шифрования...",
                                    key="enc_text")
            a_enc = st.number_input("Ключ a:", min_value=1, max_value=ALPHABET_SIZE-1, value=3, key="enc_a")
            b_enc = st.number_input("Ключ b:", min_value=0, max_value=ALPHABET_SIZE-1, value=5, key="enc_b")
            
            if st.button("Зашифровать", key="encrypt_btn"):
                if plain_text:
                    if extended_gcd(a_enc, ALPHABET_SIZE)[0] == 1:
                        encrypted = affine_encrypt(plain_text, a_enc, b_enc)
                        st.success("Зашифрованный текст:")
                        st.code(encrypted)
                    else:
                        st.error("Ключ a должен быть взаимно прост с размером алфавита (32)")
        
        with col2:
            st.subheader("Расшифрование")
            cipher_text_dec = st.text_area("Зашифрованный текст:", height=100,
                                         placeholder="Введите текст для расшифрования...", 
                                         key="dec_text")
            a_dec = st.number_input("Ключ a:", min_value=1, max_value=ALPHABET_SIZE-1, value=3, key="dec_a")
            b_dec = st.number_input("Ключ b:", min_value=0, max_value=ALPHABET_SIZE-1, value=5, key="dec_b")
            
            if st.button("Расшифровать", key="decrypt_btn"):
                if cipher_text_dec:
                    decrypted = affine_decrypt(cipher_text_dec, a_dec, b_dec)
                    if decrypted is not None:
                        st.success("Расшифрованный текст:")
                        st.code(decrypted)
                    else:
                        st.error("Невозможно расшифровать: ключ a не имеет обратного элемента")
    
    # Вкладка 5: Теория
    with tab5:
        st.header("Теоретические сведения")
        
        st.subheader("Аффинный шифр")
        st.markdown("""
        **Формула шифрования:**
        ```
        y = (a × x + b) mod m
        ```
        где:
        - `x` - код символа открытого текста (0-31)
        - `y` - код символа шифртекста
        - `a, b` - ключ шифрования
        - `m` - мощность алфавита (32 для русского)
        
        **Формула расшифрования:**
        ```
        x = a⁻¹ × (y - b) mod m
        ```
        где `a⁻¹` - обратный элемент к `a` по модулю `m`
        """)
        
        st.subheader("Требования к ключу")
        st.markdown("""
        - Число `a` должно быть взаимно простым с `m` (НОД(a, m) = 1)
        - Число `b` может быть любым в диапазоне [0, m-1]
        - Для русского алфавита (m=32) возможные значения `a`: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
        """)
        
        st.subheader("Метод криптоанализа")
        st.markdown("""
        1. **Частотный анализ** - определение самых частых символов в шифртексте
        2. **Сопоставление** с эталонными частотами русского языка
        3. **Решение системы уравнений** для нахождения ключа
        4. **Проверка** полученных ключей на осмысленность текста
        
        **Система уравнений для нахождения ключа:**
        ```
        a × x₁ + b ≡ y₁ (mod m)
        a × x₂ + b ≡ y₂ (mod m)
        ```
        где (x₁, y₁) и (x₂, y₂) - предполагаемые соответствия символов.
        """)
        
        st.subheader("Таблица кодировки русского алфавита")
        coding_table = []
        for i, char in enumerate(RUSSIAN_ALPHABET):
            coding_table.append({'Буква': char, 'Код': i})
        
        # Разбиваем на 4 колонки для лучшего отображения
        cols = st.columns(4)
        for idx, col in enumerate(cols):
            with col:
                start = idx * 8
                end = start + 8
                for item in coding_table[start:end]:
                    st.write(f"**{item['Буква']}** = {item['Код']}")

if __name__ == "__main__":
    main_page()