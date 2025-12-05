import random

ALPHABET = {
    'А': 10, 'Б': 11, 'В': 12, 'Г': 13, 'Д': 14, 'Е': 15, 'Ж': 16, 'З': 17, 'И': 18, 'Й': 19,
    'К': 20, 'Л': 21, 'М': 22, 'Н': 23, 'О': 24, 'П': 25, 'Р': 26, 'С': 27, 'Т': 28, 'У': 29,
    'Ф': 30, 'Х': 31, 'Ц': 32, 'Ч': 33, 'Ш': 34, 'Щ': 35, 'Ъ': 36, 'Ы': 37, 'Ь': 38, 'Э': 39,
    'Ю': 40, 'Я': 41, ' ': 99
}

REVERSE_ALPHABET = {v: k for k, v in ALPHABET.items()}

def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

def mod_inverse(e, phi):
    gcd, x, _ = extended_gcd(e, phi)
    if gcd != 1:
        return None
    return x % phi

def fast_pow(base, exp, mod):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def generate_random_keys(p, q, count=3):
    n = p * q
    phi = (p - 1) * (q - 1)
    keys = []
    attempts = 0
    while len(keys) < count and attempts < 10000:
        e = random.randint(2, phi - 1)
        if extended_gcd(e, phi)[0] == 1:
            d = mod_inverse(e, phi)
            if d is not None:
                keys.append((e, d))
        attempts += 1
    return keys, n, phi

def text_to_digits(text):
    text = text.upper()
    result = ''
    for ch in text:
        if ch in ALPHABET:
            result += f"{ALPHABET[ch]:02d}" if ALPHABET[ch] < 100 else '99'
    return result

def split_into_blocks(digit_string, n):
    blocks = []
    current = ''
    i = 0
    while i < len(digit_string):
        if not current:
            if digit_string[i] == '0' and blocks:
                last = str(blocks[-1])
                moved = last[-1]
                blocks[-1] = int(last[:-1])
                current = moved
            else:
                current = digit_string[i]
                i += 1
                continue
        temp = current + digit_string[i]
        if int(temp) >= n:
            blocks.append(int(current))
            current = digit_string[i]
        else:
            current = temp
        i += 1
    if current:
        blocks.append(int(current))
    lengths = [len(str(b)) for b in blocks]
    return list(zip(blocks, lengths))

def encrypt_blocks(blocks_with_len, e, n):
    return [fast_pow(m, e, n) for m, _ in blocks_with_len]

def decrypt_blocks(cipher_blocks, d, n):
    return [fast_pow(c, d, n) for c in cipher_blocks]

def digits_to_text(decrypted_numbers, lengths):
    digits = ''
    for num, length in zip(decrypted_numbers, lengths):
        s = str(num)
        if len(s) > length:
            s = s[-length:]
        elif len(s) < length:
            s = s.zfill(length)
        digits += s
    text = ''
    i = 0
    while i + 1 < len(digits):
        code = int(digits[i:i+2])
        text += REVERSE_ALPHABET.get(code, '?')
        i += 2
    return text

def main():
    saved_keys = []
    n_global = None
    phi_global = None
    last_block_lengths = None

    while True:
        print("\n")
        print("---МЕНЮ RSA---")
        print("1. Сгенерировать ключи по p и q (3 случайные пары)")
        print("2. Добавить свой ключ вручную (вводите только e)")
        print("3. Зашифровать сообщение")
        print("4. Расшифровать сообщение")
        print("0. Выход")

        choice = input("Выберите пункт: ").strip()

        if choice == '1':
            try:
                p = int(input("Введите простое число p: "))
                q = int(input("Введите простое число q: "))
                if p == q or p <= 1 or q <= 1:
                    print("p и q должны быть разными простыми числами > 1")
                    continue
            except ValueError:
                print("Введите целые числа!")
                continue

            keys, n_global, phi_global = generate_random_keys(p, q, count=3)
            if len(keys) < 3:
                print("Не удалось найти 3 подходящих ключа. Попробуйте другие p и q.")
                continue
            saved_keys = keys[:]
            print(f"\nn = {n_global} (p × q = {p} × {q})")
            print("Сгенерированные случайные пары ключей:")
            for i, (e, d) in enumerate(saved_keys, 1):
                print(f"{i}. e = {e:<12} d = {d}")

        elif choice == '2':
            if n_global is None or phi_global is None:
                print("Сначала сгенерируйте ключи (пункт 1)!")
                continue
            while True:
                try:
                    e = int(input("Введите e (открытый ключ): "))
                    if e <= 1 or e >= phi_global:
                        print(f"e должен быть в диапазоне от 2 до {phi_global-1}")
                        continue
                except ValueError:
                    print("Введите целое число!")
                    continue

                if extended_gcd(e, phi_global)[0] != 1:
                    print("e должно быть взаимно простым с φ(n)!")
                    continue

                d = mod_inverse(e, phi_global)
                if d is None:
                    print("Не удалось вычислить d. Попробуйте другое e.")
                    continue

                saved_keys.append((e, d))
                print(f"Ключ успешно добавлен!")
                print(f"   e = {e}")
                print(f"   d = {d} вычислено автоматически")
                break

        elif choice == '3':
            if not saved_keys or n_global is None:
                print("Сначала сгенерируйте ключи!")
                continue
            text = input("Введите текст для шифрования: ")
            digit_str = text_to_digits(text)
            if not digit_str:
                print("Текст пуст или содержит недопустимые символы")
                continue
            blocks_with_len = split_into_blocks(digit_str, n_global)
            print("\nБлоки открытого текста:")
            for m, _ in blocks_with_len:
                print(m)
            print("\nВыберите ключ для шифрования:")
            for i, (e, _) in enumerate(saved_keys, 1):
                print(f"{i}. e = {e}")
            try:
                idx = int(input("Номер ключа: ")) - 1
                e = saved_keys[idx][0]
            except:
                print("Неверный номер!")
                continue
            ciphertext = encrypt_blocks(blocks_with_len, e, n_global)
            last_block_lengths = [l for _, l in blocks_with_len]
            print("\nЗашифрованные блоки:")
            print(' '.join(map(str, ciphertext)))

        elif choice == '4':
            if not saved_keys or n_global is None:
                print("Сначала сгенерируйте ключи!")
                continue
            try:
                cipher_str = input("Введите зашифрованные блоки через пробел: ")
                ciphertext = [int(x) for x in cipher_str.split()]
            except:
                print("Ошибка ввода блоков!")
                continue
            print("\nВыберите ключ для расшифровки:")
            for i, (_, d) in enumerate(saved_keys, 1):
                print(f"{i}. d = {d}")
            try:
                idx = int(input("Номер ключа: ")) - 1
                d = saved_keys[idx][1]
            except:
                print("Неверный номер!")
                continue
            if last_block_lengths and len(last_block_lengths) == len(ciphertext):
                lengths = last_block_lengths
                print("Используются сохранённые длины блоков")
            else:
                print("Введите длины исходных блоков через пробел:")
                try:
                    lengths = [int(x) for x in input().split()]
                    if len(lengths) != len(ciphertext):
                        raise ValueError
                except:
                    print("Количество длин не совпадает!")
                    continue
            plaintext_nums = decrypt_blocks(ciphertext, d, n_global)
            result = digits_to_text(plaintext_nums, lengths)
            print("\nРасшифрованный текст:")
            print(result)

        elif choice == '0':
            print("До свидания!")
            break
        else:
            print("Неверный пункт меню!")

if __name__ == "__main__":
    main()