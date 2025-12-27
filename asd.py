import math

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

def generate_primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def advanced_factorize(n):
    if n % 2 == 0:
        return 2, n // 2
    if n % 3 == 0:
        return 3, n // 3
    if n % 5 == 0:
        return 5, n // 5

    primes = generate_primes(100000)

    i = 0
    while i + 2 < len(primes):
        q1, q2, q3 = primes[i], primes[i+1], primes[i+2]
        qs = q1 * q2 * q3
        if qs * qs > n:
            break
        d = math.gcd(n, qs)
        if d > 1:
            return d, n // d
        i += 3

    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            return p, n // p

    return None, None

REV = {10:'А',11:'Б',12:'В',13:'Г',14:'Д',15:'Е',16:'Ж',17:'З',18:'И',19:'Й',
       20:'К',21:'Л',22:'М',23:'Н',24:'О',25:'П',26:'Р',27:'С',28:'Т',29:'У',
       30:'Ф',31:'Х',32:'Ц',33:'Ч',34:'Ш',35:'Щ',36:'Ъ',37:'Ы',38:'Ь',39:'Э',
       40:'Ю',41:'Я',99:' '}

print("КРИПТОАНАЛИЗ ШИФРА RSA — взлом через факторизацию n\n")

e, n = map(int, input("Введите e и n через пробел: ").split())
cipher = list(map(int, input("Введите зашифрованные блоки C через пробел: ").split()))

print("Факторизация n (улучшенный метод пробного деления)...")
p, q = advanced_factorize(n)
if not p:
    print("Не удалось факторизовать n — шифр устойчив к атаке.")
    exit()

print(f"Найдено: p = {p}, q = {q}")

phi = (p - 1) * (q - 1)
d = mod_inverse(e, phi)
if d is None:
    print("Ошибка: не удалось найти d")
    exit()

print(f"φ(n) = {phi}")
print(f"Закрытый ключ d = {d}")

print("\nРасшифровка блоков...")
plain_blocks = [fast_pow(c, d, n) for c in cipher]

print("Числовые блоки открытого текста:")
print(' '.join(map(str, plain_blocks)))

all_digits = ''.join(str(num) for num in plain_blocks)
text = ""
i = 0
while i + 1 < len(all_digits):
    code = int(all_digits[i:i+2])
    text += REV.get(code, '?')
    i += 2

print("\nИсходное сообщение:")
print(text)
print("\nВзлом завершён успешно!")

# e,n: 251 46883
# C: 12351

# 7 22213
# 5891 9958 20480 8283 5963 5690