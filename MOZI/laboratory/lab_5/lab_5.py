import streamlit as st
import math


def gcd(a, b):
    """Обычный алгоритм Евклида для проверки НОД."""
    while b:
        a, b = b, a % b
    return a

def extended_gcd(a, b):
    """
    Расширенный алгоритм Евклида.
    Необходим для поиска d, так как решает уравнение: e*d + phi*y = 1
    """
    if a == 0:
        return b, 0, 1
    d_res, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return d_res, x, y

def mod_inverse(e, phi):
    """Нахождение секретного ключа d через расширенный алгоритм Евклида."""
    d_res, x, _ = extended_gcd(e, phi)
    if d_res != 1:
        return None 
    return x % phi

def fast_pow(base, exp, mod):
    """Быстрое (бинарное) возведение в степень по модулю."""
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def advanced_factorize_optimized(n):
    """
    Метод пробных делений.
    """
    temp_n = n
    factors = []
    
    # 1. Выделяем степени 2, 3, 5
    for p in [2, 3, 5]:
        while temp_n % p == 0:
            factors.append(p)
            temp_n //= p
            
    # 2. Групповой метод через НОД 
    primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    idx = 0
    while idx < len(primes) - 2 and temp_n > 1:
        qs = primes[idx] * primes[idx+1] * primes[idx+2]
        common = gcd(temp_n, qs)
        
        if common > 1:
            # Если НОД > 1, значит внутри группы есть делитель
            for p in [primes[idx], primes[idx+1], primes[idx+2]]:
                while temp_n % p == 0:
                    factors.append(p)
                    temp_n //= p
        idx += 3
        
    # 3. Добор оставшихся делителей (если n большое)
    if temp_n > 1:
        d = 53 # начинаем после списка primes
        while d * d <= temp_n:
            if temp_n % d == 0:
                factors.append(d)
                temp_n //= d
            else:
                d += 2
        if temp_n > 1:
            factors.append(temp_n)
            
    return factors

REV = {10:'А',11:'Б',12:'В',13:'Г',14:'Д',15:'Е',16:'Ж',17:'З',18:'И',19:'Й',
       20:'К',21:'Л',22:'М',23:'Н',24:'О',25:'П',26:'Р',27:'С',28:'Т',29:'У',
       30:'Ф',31:'Х',32:'Ц',33:'Ч',34:'Ш',35:'Щ',36:'Ъ',37:'Ы',38:'Ь',39:'Э',
       40:'Ю',41:'Я',99:' '}


def lab5():
    st.title("Лабораторная работа №5 — Криптоанализ RSA")
    st.write("Взлом через оптимизированный метод пробных делений.")
    
    col1, col2 = st.columns(2)
    with col1:
        e_input = st.number_input("Открытая экспонента (e):", value=251)
        n_input = st.number_input("Модуль (n):", value=49583)
    with col2:
        cipher_text = st.text_area("Зашифрованные блоки:", value="38076")

    if st.button("Взломать RSA", type="primary"):
        try:
            cipher_blocks = list(map(int, cipher_text.split()))
            
            # 1. Факторизация
            with st.status("Выполняется криптоанализ...") as s:
                st.write("Выполняем групповую факторизацию n...")
                factors = advanced_factorize_optimized(n_input)
                
                if len(factors) < 2:
                    st.error("Не удалось найти p и q.")
                    return
                
                p, q = factors[0], factors[1]
                st.write(f"Найдено: p={p}, q={q}")
                
                # 2. Эйлер
                phi = (p - 1) * (q - 1)
                
                # 3. Ключ d
                st.write("Вычисляем d через расширенный алгоритм Евклида...")
                d = mod_inverse(e_input, phi)
                
                # 4. Дешифровка
                plain_blocks = [fast_pow(c, d, n_input) for c in cipher_blocks]
                s.update(label="Взлом завершен!", state="complete")

            # Вывод результатов
            st.success(f"**Секретный ключ d:** {d}")
            
            # Декодирование текста
            all_digits = ''.join(str(num) for num in plain_blocks)
            decoded_text = ""
            for k in range(0, len(all_digits), 2):
                chunk = all_digits[k:k+2]
                if len(chunk) == 2:
                    code = int(chunk)
                    decoded_text += REV.get(code, '?')
            
            st.info(f"**Результат дешифрования:** {decoded_text}")
            st.code(f"Числовые блоки: {plain_blocks}")

        except Exception as ex:
            st.error(f"Ошибка при взломе: {ex}")

if __name__ == "__main__":
    lab5()