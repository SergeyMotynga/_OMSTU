"""
Лабораторная работа 5.

Задача: сравнить алгоритмы поиска подстроки в строке:
- наивный метод,
- Бойер-Мур,
- Рабин-Карп,
- Кнут-Моррис-Пратт.

Скрипт измеряет время, выводит таблицу и строит график.
"""

import random
import string
import time

import matplotlib.pyplot as plt


def naive_search(text, pattern):
    """
    Находит все вхождения pattern в text наивным методом.

    Идея:
    для каждого возможного старта i в тексте посимвольно
    сравниваем pattern с text[i:i+m].

    Параметры:
    text (str): исходная строка.
    pattern (str): искомая подстрока.

    Возвращает:
    list[int]: индексы всех вхождений pattern.

    Сложность:
    O(n*m) в худшем случае.
    """
    n = len(text)
    m = len(pattern)
    if m == 0:
        return list(range(n + 1))
    if m > n:
        return []

    matches = []
    for i in range(n - m + 1):
        j = 0
        while j < m and text[i + j] == pattern[j]:
            j += 1
        if j == m:
            matches.append(i)
    return matches


def _build_bad_char_table(pattern):
    """
    Строит таблицу последнего вхождения символа для Бойера-Мура.

    Параметры:
    pattern (str): шаблон.

    Возвращает:
    dict[str, int]: символ -> его последняя позиция в шаблоне.
    """
    table = {}
    for i, ch in enumerate(pattern):
        table[ch] = i
    return table


def _build_good_suffix_table(pattern):
    """
    Строит таблицу сдвигов good suffix для Бойера-Мура.

    Параметры:
    pattern (str): шаблон.

    Возвращает:
    list[int]: таблица сдвигов длины m+1.
    """
    m = len(pattern)
    shift = [0] * (m + 1)
    border = [0] * (m + 1)

    i = m
    j = m + 1
    border[i] = j

    while i > 0:
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if shift[j] == 0:
                shift[j] = j - i
            j = border[j]
        i -= 1
        j -= 1
        border[i] = j

    j = border[0]
    for i in range(m + 1):
        if shift[i] == 0:
            shift[i] = j
        if i == j:
            j = border[j]

    return shift


def boyer_moore_search(text, pattern):
    """
    Находит все вхождения pattern в text алгоритмом Бойера-Мура.

    Используются две эвристики:
    - bad character,
    - good suffix.

    Параметры:
    text (str): исходная строка.
    pattern (str): искомая подстрока.

    Возвращает:
    list[int]: индексы всех вхождений pattern.

    Сложность:
    - на практике обычно быстрее наивного,
    - худший случай O(n*m).
    """
    n = len(text)
    m = len(pattern)
    if m == 0:
        return list(range(n + 1))
    if m > n:
        return []

    bad_char = _build_bad_char_table(pattern)
    good_suffix = _build_good_suffix_table(pattern)

    matches = []
    s = 0
    while s <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1

        if j < 0:
            matches.append(s)
            s += good_suffix[0]
        else:
            bc_idx = bad_char.get(text[s + j], -1)
            bc_shift = max(1, j - bc_idx)
            gs_shift = good_suffix[j + 1]
            s += max(bc_shift, gs_shift)

    return matches


def rabin_karp_search(text, pattern, base=257, mod=1_000_000_007):
    """
    Находит все вхождения pattern в text алгоритмом Рабина-Карпа.

    Алгоритм использует rolling hash:
    - считает хеш шаблона,
    - считает хеш каждого окна длины m,
    - при совпадении хеша делает точную проверку строки.

    Параметры:
    text (str): исходная строка.
    pattern (str): искомая подстрока.
    base (int): основание полиномиального хеша.
    mod (int): модуль хеша.

    Возвращает:
    list[int]: индексы всех вхождений pattern.

    Сложность:
    - средний случай O(n+m),
    - худший O(n*m) при большом числе коллизий.
    """
    n = len(text)
    m = len(pattern)
    if m == 0:
        return list(range(n + 1))
    if m > n:
        return []

    power = pow(base, m - 1, mod)

    p_hash = 0
    w_hash = 0
    for i in range(m):
        p_hash = (p_hash * base + ord(pattern[i])) % mod
        w_hash = (w_hash * base + ord(text[i])) % mod

    matches = []
    for i in range(n - m + 1):
        if w_hash == p_hash and text[i:i + m] == pattern:
            matches.append(i)

        if i < n - m:
            left = ord(text[i])
            right = ord(text[i + m])
            w_hash = (w_hash - left * power) % mod
            w_hash = (w_hash * base + right) % mod

    return matches


def _kmp_prefix_function(pattern):
    """
    Вычисляет префикс-функцию для KMP.

    Параметры:
    pattern (str): шаблон.

    Возвращает:
    list[int]: массив префикс-функции.
    """
    m = len(pattern)
    pi = [0] * m
    j = 0

    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = pi[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        pi[i] = j

    return pi


def kmp_search(text, pattern):
    """
    Находит все вхождения pattern в text алгоритмом KMP.

    Параметры:
    text (str): исходная строка.
    pattern (str): искомая подстрока.

    Возвращает:
    list[int]: индексы всех вхождений pattern.

    Сложность:
    O(n+m).
    """
    n = len(text)
    m = len(pattern)
    if m == 0:
        return list(range(n + 1))
    if m > n:
        return []

    pi = _kmp_prefix_function(pattern)
    matches = []
    j = 0

    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = pi[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            matches.append(i - m + 1)
            j = pi[j - 1]

    return matches


def measure_ms(func, *args, repeats=7):
    """
    Измеряет среднее время выполнения функции в миллисекундах.

    Параметры:
    func (callable): функция, которую измеряем.
    *args: аргументы для функции.
    repeats (int): число повторных измерений.

    Возвращает:
    float: среднее время в миллисекундах.
    """
    func(*args)
    total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args)
        total += time.perf_counter() - t0
    return total / repeats * 1000


def make_text_and_pattern(n, m, alphabet=string.ascii_lowercase):
    """
    Генерирует тестовый текст длины n и шаблон длины m.

    Чтобы сравнение алгоритмов было корректным, в текст гарантированно
    вставляется хотя бы одно вхождение шаблона.

    Параметры:
    n (int): длина текста.
    m (int): длина шаблона.
    alphabet (str): набор символов для генерации.

    Возвращает:
    tuple[str, str]: (text, pattern).
    """
    text = [random.choice(alphabet) for _ in range(n)]
    pattern = "".join(random.choice(alphabet) for _ in range(m))

    if m <= n:
        pos = random.randint(0, n - m)
        text[pos:pos + m] = pattern

    return "".join(text), pattern


def main():
    """
    Запускает лабораторную работу 5.

    Что делает функция:
    1) Генерирует наборы данных разного размера.
    2) Измеряет время каждого алгоритма.
    3) Выводит таблицу результатов.
    4) Печатает теоретические оценки сложности.
    5) Сохраняет график сравнения.
    """
    random.seed(42)

    print("Лабораторная работа 5")
    print("Сравнение алгоритмов поиска подстроки")

    demo_cases = [
        ("abracadabra", "abra"),
        ("aaaaa", "aa"),
        ("abcabcabcabc", "abc"),
        ("the quick brown fox jumps over the lazy dog", "the"),
        ("mississippi", "issi"),
    ]

    print("\nДемонстрация на коротких примерах")
    for text, pattern in demo_cases:
        print(f"\nТекст: {text}")
        print(f"Шаблон: {pattern}")
        print(f"Naive: {naive_search(text, pattern)}")
        print(f"Boyer-Moore: {boyer_moore_search(text, pattern)}")
        print(f"Rabin-Karp: {rabin_karp_search(text, pattern)}")
        print(f"KMP: {kmp_search(text, pattern)}")

    text_sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 300_000]
    pattern_len = 20

    algos = {
        "Naive": naive_search,
        "Boyer-Moore": boyer_moore_search,
        "Rabin-Karp": rabin_karp_search,
        "KMP": kmp_search,
    }

    results = {name: [] for name in algos}

    print(f"\nДлина шаблона: m={pattern_len}")
    for n in text_sizes:
        text, pattern = make_text_and_pattern(n, pattern_len)
        repeats = 10 if n <= 10_000 else (6 if n <= 100_000 else 3)

        for name, algo in algos.items():
            results[name].append(measure_ms(algo, text, pattern, repeats=repeats))

        print(f"n={n}: замер выполнен, repeats={repeats}")

    names = list(algos.keys())

    print("\nРезультаты (мс):")
    header = f"{'n':>9}"
    for name in names:
        header += f"{name:>16}"
    print(header)

    for i, n in enumerate(text_sizes):
        row = f"{n:>9}"
        for name in names:
            row += f"{results[name][i]:>13.4f} ms"
        print(row)

    print("\nСложность:")
    print("Naive: O(n*m)")
    print("Boyer-Moore: средний случай быстрее, худший O(n*m)")
    print("Rabin-Karp: средний O(n+m), худший O(n*m)")
    print("KMP: O(n+m)")

    colors = {
        "Naive": "#e74c3c",
        "Boyer-Moore": "#3498db",
        "Rabin-Karp": "#f39c12",
        "KMP": "#2ecc71",
    }
    markers = {
        "Naive": "o",
        "Boyer-Moore": "s",
        "Rabin-Karp": "^",
        "KMP": "D",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("ЛР5: сравнение алгоритмов поиска подстроки", fontsize=13, fontweight="bold")

    for ax, yscale, title in [
        (axes[0], "linear", "Линейная шкала"),
        (axes[1], "log", "Логарифмическая шкала"),
    ]:
        for name in names:
            ax.plot(
                text_sizes,
                results[name],
                label=name,
                color=colors[name],
                marker=markers[name],
                linewidth=2,
                markersize=6,
            )
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        ax.set_title(title)
        ax.set_xlabel("Размер текста n")
        ax.set_ylabel("Время (мс)" + (" log" if yscale == "log" else ""))
        ax.grid(True, linestyle="--", alpha=0.6, which="both")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("task1_string_search_comparison.png", dpi=150, bbox_inches="tight")
    print("\nГрафик сохранен: task1_string_search_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
