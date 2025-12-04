import streamlit as st
import random
import pandas as pd
import io
import math

# ======== КОНСТАНТЫ И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ========

# Таблица кодировки русского алфавита
ALPHABET = {
    'А': 10, 'Б': 11, 'В': 12, 'Г': 13, 'Д': 14, 'Е': 15, 'Ж': 16, 'З': 17, 
    'И': 18, 'Й': 19, 'К': 20, 'Л': 21, 'М': 22, 'Н': 23, 'О': 24, 'П': 25,
    'Р': 26, 'С': 27, 'Т': 28, 'У': 29, 'Ф': 30, 'Х': 31, 'Ц': 32, 'Ч': 33,
    'Ш': 34, 'Щ': 35, 'Ъ': 36, 'Ы': 37, 'Ь': 38, 'Э': 39, 'Ю': 40, 'Я': 41,
    ' ': 99  # пробел
}

# Обратная таблица для декодирования
REVERSE_ALPHABET = {v: k for k, v in ALPHABET.items()}

# ======== МАТЕМАТИЧЕСКИЕ ФУНКЦИИ ========

def extended_gcd(a, b):
    """Расширенный алгоритм Евклида для нахождения НОД и коэффициентов"""
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

def mod_inverse(e, phi):
    """Нахождение обратного элемента по модулю"""
    gcd, x, _ = extended_gcd(e, phi)
    if gcd != 1:
        return None  # Обратного не существует
    return x % phi

def fast_pow(base, exp, mod):
    """Бинарное возведение в степень по модулю"""
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:  # если exp нечетное
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1  # exp = exp // 2
    return result

def is_prime(n):
    """Проверка числа на простоту"""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # Проверяем делимость на числа вида 6k ± 1
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# ======== ФУНКЦИИ RSA ========

def generate_keys(p, q, count=3):
    """Генерация count пар ключей (e, d) для заданных p и q"""
    n = p * q
    phi = (p - 1) * (q - 1)
    keys = []
    
    # Попытки найти взаимно простые e
    for _ in range(1000):  # ограничим попытки
        if len(keys) >= count:
            break
        e = random.randint(2, phi - 1)
        if math.gcd(e, phi) == 1:  # Используем math.gcd
            d = mod_inverse(e, phi)
            if d is not None:
                keys.append((e, d))
    
    return keys, n, phi

def text_to_digits(text):
    """Преобразование текста в строку цифр по таблице ALPHABET"""
    text = text.upper().strip()
    result = []
    for ch in text:
        if ch in ALPHABET:
            code = ALPHABET[ch]
            # Кодируем двухзначными числами, кроме пробела (99)
            result.append(f"{code:02d}" if code < 100 else "99")
    return ''.join(result)

def split_into_blocks(digit_str, n):
    """Разбиение цифровой строки на блоки, меньшие n"""
    blocks = []
    current_block = ""
    
    i = 0
    while i < len(digit_str):
        # Если текущий блок пустой
        if not current_block:
            # Проверяем, начинается ли с нуля (особый случай)
            if digit_str[i] == '0':
                # Переносим последнюю цифру из предыдущего блока
                if blocks:
                    last_block_str = str(blocks[-1])
                    if len(last_block_str) > 1:
                        moved_digit = last_block_str[-1]
                        blocks[-1] = int(last_block_str[:-1])
                        current_block = moved_digit
                        continue
                # Если это первый блок или предыдущий блок однозначный
                current_block = digit_str[i]
                i += 1
            else:
                current_block = digit_str[i]
                i += 1
            continue
        
        # Пробуем добавить следующую цифру
        test_block = current_block + digit_str[i]
        
        if int(test_block) < n:
            current_block = test_block
        else:
            # Текущий блок завершен
            blocks.append(int(current_block))
            current_block = digit_str[i]
        
        i += 1
    
    # Добавляем последний блок
    if current_block:
        blocks.append(int(current_block))
    
    # Сохраняем длины блоков для правильного восстановления
    lengths = [len(str(b)) for b in blocks]
    
    return blocks, lengths

def encrypt_message(text, e, n):
    """Шифрование текста"""
    # Преобразуем текст в цифры
    digit_str = text_to_digits(text)
    if not digit_str:
        return [], []
    
    # Разбиваем на блоки
    blocks, lengths = split_into_blocks(digit_str, n)
    
    # Шифруем каждый блок
    cipher_blocks = [fast_pow(block, e, n) for block in blocks]
    
    return cipher_blocks, lengths

def decrypt_message(cipher_blocks, d, n, original_lengths=None):
    """Расшифровка текста"""
    # Расшифровываем каждый блок
    plain_blocks = [fast_pow(block, d, n) for block in cipher_blocks]
    
    # Восстанавливаем цифровую строку
    digits = ""
    for i, block in enumerate(plain_blocks):
        block_str = str(block)
        if original_lengths and i < len(original_lengths):
            # Восстанавливаем оригинальную длину
            if len(block_str) < original_lengths[i]:
                block_str = block_str.zfill(original_lengths[i])
            elif len(block_str) > original_lengths[i]:
                block_str = block_str[-original_lengths[i]:]
        digits += block_str
    
    # Преобразуем цифры обратно в текст
    text = ""
    i = 0
    while i + 1 < len(digits):
        code = int(digits[i:i+2])
        if code in REVERSE_ALPHABET:
            text += REVERSE_ALPHABET[code]
        i += 2
    
    return text

# ======== ОСНОВНАЯ ФУНКЦИЯ STREAMLIT ========

def main_page():
    st.title("Лабораторная работа №4 — Шифрование RSA")
    st.markdown("**Цель:** Реализация алгоритма шифрования с открытым ключом RSA")
    
    # Инициализация состояния
    if 'rsa_keys' not in st.session_state:
        st.session_state.rsa_keys = []  # список кортежей (e, d)
    if 'rsa_n' not in st.session_state:
        st.session_state.rsa_n = None  
    if 'rsa_phi' not in st.session_state:
        st.session_state.rsa_phi = None  
    if 'rsa_last_lengths' not in st.session_state:
        st.session_state.rsa_last_lengths = None  
    if 'rsa_ciphertext' not in st.session_state:
        st.session_state.rsa_ciphertext = None  
    
    st.markdown("---")
    
    # СЕКЦИЯ 1: ГЕНЕРАЦИЯ КЛЮЧЕЙ
    st.header("1. Генерация ключей")
    
    col1, col2 = st.columns(2)
    
    with col1:
        p = st.number_input("Введите простое число p:", 
                           min_value=2, 
                           max_value=1000, 
                           value=151,
                           step=1,
                           help="Простое число из таблицы вариантов")
    
    with col2:
        q = st.number_input("Введите простое число q:", 
                           min_value=2, 
                           max_value=1000, 
                           value=283,
                           step=1,
                           help="Простое число из таблицы вариантов")
    
    if st.button("Сгенерировать 3 пары ключей", type="primary"):
        # Проверяем, что числа простые
        if not is_prime(p):
            st.error(f"Число p={p} не является простым!")
            return
        if not is_prime(q):
            st.error(f"Число q={q} не является простым!")
            return
        if p == q:
            st.error("Числа p и q должны быть разными!")
            return
        
        # Генерируем ключи
        keys, n, phi = generate_keys(p, q, count=3)
        
        if len(keys) < 3:
            st.warning(f"Удалось сгенерировать только {len(keys)} пар ключей")
        
        # Сохраняем в состоянии - 
        st.session_state.rsa_keys = keys
        st.session_state.rsa_n = n
        st.session_state.rsa_phi = phi
        
        st.success(f"Ключи сгенерированы! n = {n} (p×q = {p}×{q})")
    
    # Отображение текущих ключей - 
    if st.session_state.rsa_keys:
        st.subheader("Текущие ключи:")
        key_data = []
        # ИЗМЕНЕНО: st.session_state.rsa_keys вместо st.session_state.keys
        for i, (e, d) in enumerate(st.session_state.rsa_keys, 1):
            key_data.append({
                "№": i,
                "Открытый ключ (e)": e,
                "Закрытый ключ (d)": d,
                "Модуль (n)": st.session_state.rsa_n  
            })
        
        df_keys = pd.DataFrame(key_data)
        st.dataframe(df_keys, use_container_width=True, hide_index=True)
        
        # Кнопка для скачивания ключей
        csv_buffer = io.StringIO()
        df_keys.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Скачать ключи (CSV)",
            data=csv_buffer.getvalue(),
            file_name="rsa_keys.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # СЕКЦИЯ 2: ДОБАВЛЕНИЕ СВОЕГО КЛЮЧА - 
    st.header("2. Добавить свой ключ")
    
    if st.session_state.rsa_n is not None:  
        st.info(f"Текущий модуль n = {st.session_state.rsa_n}, φ(n) = {st.session_state.rsa_phi}")  
        
        e_input = st.number_input("Введите значение e (открытый ключ):",
                                 min_value=2,
                                 max_value=st.session_state.rsa_phi-1,  
                                 value=17,
                                 step=1,
                                 help=f"Должно быть взаимно простым с φ(n) = {st.session_state.rsa_phi}")  
        
        if st.button("Добавить ключ"):
            # Проверяем, что e взаимно просто с phi - ИСПОЛЬЗУЕМ math.gcd
            if math.gcd(e_input, st.session_state.rsa_phi) != 1:  
                st.error(f"e={e_input} не взаимно просто с φ(n)={st.session_state.rsa_phi}!")  
            else:
                d = mod_inverse(e_input, st.session_state.rsa_phi)  
                if d is None:
                    st.error("Не удалось найти обратный элемент!")
                else:
                    st.session_state.rsa_keys.append((e_input, d))  
                    st.success(f"Ключ добавлен! d = {d}")
    else:
        st.warning("Сначала сгенерируйте ключи (раздел 1)")
    
    st.markdown("---")
    
    # СЕКЦИЯ 3: ШИФРОВАНИЕ - 
    st.header("3. Шифрование сообщения")
    
    if st.session_state.rsa_keys:  
        # Выбор ключа для шифрования
        key_options = [f"Ключ {i}: e={e}" for i, (e, d) in enumerate(st.session_state.rsa_keys, 1)]  
        selected_key_idx = st.selectbox("Выберите ключ для шифрования:", 
                                       range(len(key_options)),
                                       format_func=lambda x: key_options[x])
        
        e_selected, d_selected = st.session_state.rsa_keys[selected_key_idx]  
        
        # Ввод текста
        text_to_encrypt = st.text_area("Введите текст для шифрования:",
                                      height=100,
                                      placeholder="Введите текст на русском языке...")
        
        if st.button("Зашифровать", type="primary"):
            if not text_to_encrypt.strip():
                st.warning("Введите текст для шифрования!")
            else:
                # Преобразуем текст в цифры
                digit_str = text_to_digits(text_to_encrypt)
                
                # Разбиваем на блоки
                blocks, lengths = split_into_blocks(digit_str, st.session_state.rsa_n)  
                
                # Шифруем
                cipher_blocks = [fast_pow(block, e_selected, st.session_state.rsa_n) for block in blocks]
                
                # Сохраняем для возможной расшифровки
                st.session_state.rsa_last_lengths = lengths
                st.session_state.rsa_ciphertext = cipher_blocks
                
                # Отображаем результаты
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Блоки открытого текста:")
                    for i, block in enumerate(blocks, 1):
                        st.write(f"Блок {i}: {block} (длина: {lengths[i-1]})")
                
                with col2:
                    st.subheader("Зашифрованные блоки:")
                    cipher_text = " ".join(map(str, cipher_blocks))
                    st.code(cipher_text, language="text")
                    
                
                # Показываем цифровое представление
                with st.expander("Подробности преобразования"):
                    st.write("**Текст в цифрах:**")
                    st.code(digit_str, language="text")
                    
                    # Отображаем таблицу кодировки
                    st.write("**Таблица кодировки:**")
                    table_data = []
                    for char, code in ALPHABET.items():
                        table_data.append({"Символ": char, "Код": code})
                    st.table(pd.DataFrame(table_data).set_index("Символ"))
    else:
        st.warning("Сначала сгенерируйте ключи (раздел 1)")
    
    st.markdown("---")
    
    # СЕКЦИЯ 4: РАСШИФРОВКА
    st.header("4. Расшифровка сообщения")
    
    if st.session_state.rsa_keys:
        # Выбор ключа для расшифровки
        key_options_dec = [f"Ключ {i}: d={d}" for i, (e, d) in enumerate(st.session_state.rsa_keys, 1)]
        selected_key_idx_dec = st.selectbox("Выберите ключ для расшифровки:", 
                                           range(len(key_options_dec)),
                                           format_func=lambda x: key_options_dec[x])
        
        e_selected_dec, d_selected_dec = st.session_state.rsa_keys[selected_key_idx_dec]
        
        # Ввод шифртекста
        cipher_input = st.text_area("Введите зашифрованные блоки (через пробел):",
                                   height=100,
                                   placeholder="Например: 1234 5678 9012",
                                   value=" ".join(map(str, st.session_state.rsa_ciphertext))
                                   if st.session_state.rsa_ciphertext else "")
        
        if st.button("Расшифровать", type="primary"):
            if not cipher_input.strip():
                st.warning("Введите шифртекст для расшифровки!")
            else:
                try:
                    # Парсим шифртекст
                    cipher_blocks = [int(x.strip()) for x in cipher_input.split()]
                    
                    lengths = None
                    
                    # Расшифровываем
                    decrypted_text = decrypt_message(cipher_blocks, d_selected_dec, 
                                                    st.session_state.rsa_n, lengths)
                    
                    # Отображаем результат
                    st.subheader("Расшифрованный текст:")
                    st.success(decrypted_text)
                    
                    # Показываем информацию
                    with st.expander("Подробности расшифровки"):
                        st.write(f"**Использованные параметры:**")
                        st.write(f"- Закрытый ключ d: {d_selected_dec}")
                        st.write(f"- Модуль n: {st.session_state.rsa_n}")
                        st.write(f"- Количество блоков: {len(cipher_blocks)}")
                        
                        if lengths:
                            st.write(f"- Длины блоков: {lengths}")
                        else:
                            st.write("- Длины блоков: автоматическое определение")
                        
                except ValueError as e:
                    st.error(f"Ошибка ввода данных: {e}")
                except Exception as e:
                    st.error(f"Ошибка при расшифровке: {e}")
    else:
        st.warning("Сначала сгенерируйте ключи (раздел 1)")