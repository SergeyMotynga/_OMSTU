"""
Лабораторная работа №4
Генерация перестановок

Задача 1: Генерация всех перестановок 1..n через аналог кода Грэя
          (алгоритм Штейнхауса–Джонсона–Троттера)
Задача 2: Сравнительный анализ алгоритмов генерации случайных перестановок
          с замером времени выполнения
"""

import time
import math
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ═══════════════════════════════════════════════════════════════════════════════
# ЗАДАЧА 1 — Алгоритм Штейнхауса–Джонсона–Троттера (аналог кода Грэя)
# ═══════════════════════════════════════════════════════════════════════════════

def sjt_permutations(n):
    """
    Генерирует ВСЕ n! перестановок значений 1..n методом
    Штейнхауса–Джонсона–Троттера.

    Аналог кода Грэя для перестановок: каждая следующая перестановка
    отличается от предыдущей ровно одной транспозицией соседних элементов.

    Алгоритм:
      - каждому элементу назначается направление (-1 = влево, +1 = вправо);
      - «подвижный» элемент — тот, чей сосед в направлении движения меньше;
      - находим наибольший подвижный элемент и меняем его с соседом;
      - всем элементам, бо́льшим переставленного, меняем направление.
    """
    if n == 0:
        return [[]]
    if n == 1:
        return [[1]]

    perm = list(range(1, n + 1))
    dirs = [-1] * n           # -1 = влево, +1 = вправо
    result = [perm[:]]

    while True:
        # Поиск наибольшего подвижного элемента
        mob_pos = -1
        mob_val = -1
        for i in range(n):
            j = i + dirs[i]
            if 0 <= j < n and perm[i] > perm[j] and perm[i] > mob_val:
                mob_val = perm[i]
                mob_pos = i

        if mob_pos == -1:
            break   # нет подвижных элементов — все перестановки готовы

        # Транспозиция подвижного элемента с его соседом
        j = mob_pos + dirs[mob_pos]
        perm[mob_pos], perm[j] = perm[j], perm[mob_pos]
        dirs[mob_pos], dirs[j] = dirs[j], dirs[mob_pos]

        # Смена направления у всех элементов, бо́льших переставленного
        for i in range(n):
            if perm[i] > mob_val:
                dirs[i] = -dirs[i]

        result.append(perm[:])

    return result


def swap_position(prev, cur):
    """Возвращает индекс левой позиции транспозиции между соседними перестановками."""
    for i in range(len(prev)):
        if prev[i] != cur[i]:
            return i
    return -1


# ═══════════════════════════════════════════════════════════════════════════════
# ЗАДАЧА 2 — Алгоритмы генерации случайных перестановок
# ═══════════════════════════════════════════════════════════════════════════════

def fisher_yates(n):
    """
    Перемешивание Фишера–Йетса (алгоритм Кнута), O(n).
    Стандартный алгоритм: проходим с конца, на каждом шаге
    выбираем случайный элемент из ещё не рассмотренных и ставим его на место i.
    """
    a = list(range(1, n + 1))
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        a[i], a[j] = a[j], a[i]
    return a


def inside_out(n):
    """
    Fisher-Yates «изнутри наружу», O(n).
    Строит перестановку инкрементально, не инициализируя массив заранее:
    для каждого нового элемента i выбирает случайную позицию j ≤ i
    и помещает a[j] на место i, а новый элемент — на место j.
    """
    a = [0] * n
    for i in range(n):
        j = random.randint(0, i)
        if j != i:
            a[i] = a[j]
        a[j] = i + 1
    return a


def naive_sort_shuffle(n):
    """
    «Наивный» алгоритм: присвоить случайные вещественные ключи и отсортировать, O(n log n).
    Генерирует равномерно случайную перестановку, но медленнее Fisher-Yates.
    """
    a = list(range(1, n + 1))
    keys = [random.random() for _ in range(n)]
    a.sort(key=lambda x: keys[x - 1])
    return a


def python_builtin(n):
    """
    Встроенный random.shuffle (CPython реализует Fisher-Yates), O(n).
    Используется как эталонный замер стандартной библиотеки.
    """
    a = list(range(1, n + 1))
    random.shuffle(a)
    return a


# ═══════════════════════════════════════════════════════════════════════════════
# Вспомогательная функция замера
# ═══════════════════════════════════════════════════════════════════════════════

def measure_ms(func, *args, repeats=10):
    """Среднее время выполнения func(*args) в миллисекундах."""
    total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args)
        total += time.perf_counter() - t0
    return total / repeats * 1000


# ═══════════════════════════════════════════════════════════════════════════════
# ЗАДАЧА 1 — Вывод в терминал
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 68)
print("ЗАДАЧА 1: Генерация перестановок через аналог кода Грэя")
print("          Алгоритм Штейнхауса–Джонсона–Троттера (SJT)")
print("=" * 68)

# Демонстрация перестановок для n = 3 и n = 4
for demo_n in [3, 4]:
    perms = sjt_permutations(demo_n)
    print(f"\nn = {demo_n}  (всего {len(perms)} = {demo_n}! перестановок):")
    print(f"  {'Перестановка':<20}  {'Транспозиция'}")
    print(f"  {'-'*18}  {'-'*20}")
    for idx, p in enumerate(perms):
        perm_str = str(p)
        if idx == 0:
            tag = "(начальная)"
        else:
            pos = swap_position(perms[idx - 1], p)
            tag = f"своп позиций {pos + 1} ↔ {pos + 2}"
        print(f"  {perm_str:<20}  {tag}")

# Замер времени генерации всех перестановок
print("\n\nВремя генерации всех n! перестановок (SJT):\n")
print(f"  {'n':>4}  {'n!':>9}  {'Время (мс)':>12}  {'нс / перестановка':>18}")
print(f"  {'-'*4}  {'-'*9}  {'-'*12}  {'-'*18}")

SJT_SIZES = list(range(3, 12))   # до 11! = 39 916 800 перестановок
sjt_times = []

for n in SJT_SIZES:
    t0 = time.perf_counter()
    perms = sjt_permutations(n)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    fact = math.factorial(n)
    ns_each = elapsed_ms * 1e6 / fact
    sjt_times.append(elapsed_ms)
    print(f"  {n:>4}  {fact:>9}  {elapsed_ms:>12.3f}  {ns_each:>18.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# ЗАДАЧА 2 — Вывод в терминал
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 68)
print("ЗАДАЧА 2: Сравнительный анализ алгоритмов случайных перестановок")
print("=" * 68)

RAND_SIZES = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
REPEATS = 15

random_algos = {
    "Fisher-Yates":    fisher_yates,
    "Inside-Out F-Y":  inside_out,
    "Naive Sort":      naive_sort_shuffle,
    "Python built-in": python_builtin,
}

rand_results = {name: [] for name in random_algos}

print(f"\nЗамер (среднее по {REPEATS} запускам)...\n")
for n in RAND_SIZES:
    for name, func in random_algos.items():
        t = measure_ms(func, n, repeats=REPEATS)
        rand_results[name].append(t)
    print(f"  n = {n:>7}: готово")

# Таблица
col_w = 17
header_names = list(random_algos.keys())
total_w = 10 + col_w * len(header_names)

print(f"\n{'n':>9}", end="")
for name in header_names:
    print(f"{name[:col_w-2]:>{col_w}}", end="")
print()
print("-" * total_w)

for i, n in enumerate(RAND_SIZES):
    print(f"{n:>9}", end="")
    for name in header_names:
        t = rand_results[name][i]
        print(f"{t:>{col_w - 3}.4f} мс", end="")
    print()

print("-" * total_w)

print("\nСложность алгоритмов:")
complexities = {
    "Fisher-Yates":    "O(n)",
    "Inside-Out F-Y":  "O(n)",
    "Naive Sort":      "O(n log n)",
    "Python built-in": "O(n)",
}
for name, c in complexities.items():
    print(f"  {name:<18} {c}")


# ═══════════════════════════════════════════════════════════════════════════════
# Графики
# ═══════════════════════════════════════════════════════════════════════════════

colors = {
    "Fisher-Yates":    "#e74c3c",
    "Inside-Out F-Y":  "#2ecc71",
    "Naive Sort":      "#e67e22",
    "Python built-in": "#3498db",
}
markers = {
    "Fisher-Yates":    "o",
    "Inside-Out F-Y":  "s",
    "Naive Sort":      "^",
    "Python built-in": "D",
}

fig = plt.figure(figsize=(16, 14))
fig.suptitle(
    "Лабораторная работа №4 — Генерация перестановок\n"
    "Анализ алгоритмов и оценка времени выполнения",
    fontsize=14, fontweight="bold"
)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# ── График 1: SJT — время vs n (линейная шкала) ───────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
fact_vals = [math.factorial(n) for n in SJT_SIZES]
ax1.plot(SJT_SIZES, sjt_times, color="#9b59b6", marker="o",
         linewidth=2, markersize=7, label="SJT")
ax1.set_title("Задача 1: SJT — время генерации всех перестановок\n(линейная шкала)", fontsize=10)
ax1.set_xlabel("n", fontsize=10)
ax1.set_ylabel("Время (мс)", fontsize=10)
ax1.set_xticks(SJT_SIZES)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=9)

# ── График 2: SJT — нс / перестановку ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ns_per_perm = [sjt_times[i] * 1e6 / fact_vals[i] for i in range(len(SJT_SIZES))]
ax2.plot(SJT_SIZES, ns_per_perm, color="#1abc9c", marker="s",
         linewidth=2, markersize=7, label="нс / перестановка")
ax2.set_title("Задача 1: SJT — накладные расходы на одну перестановку", fontsize=10)
ax2.set_xlabel("n", fontsize=10)
ax2.set_ylabel("нс / перестановка", fontsize=10)
ax2.set_xticks(SJT_SIZES)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(fontsize=9)

# ── График 3: Случайные перестановки — линейная шкала ────────────────────
ax3 = fig.add_subplot(gs[1, 0])
for name, func in random_algos.items():
    ax3.plot(RAND_SIZES, rand_results[name],
             label=name, color=colors[name], marker=markers[name],
             linewidth=2, markersize=5)
ax3.set_title("Задача 2: Алгоритмы случайных перестановок\n(линейная шкала)", fontsize=10)
ax3.set_xlabel("n (размер перестановки)", fontsize=10)
ax3.set_ylabel("Время (мс)", fontsize=10)
ax3.set_xticks(RAND_SIZES)
ax3.tick_params(axis="x", rotation=35)
ax3.legend(fontsize=9)
ax3.grid(True, linestyle="--", alpha=0.6)

# ── График 4: Случайные перестановки — логарифмическая шкала ─────────────
ax4 = fig.add_subplot(gs[1, 1])
for name, func in random_algos.items():
    ax4.plot(RAND_SIZES, rand_results[name],
             label=name, color=colors[name], marker=markers[name],
             linewidth=2, markersize=5)
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_title("Задача 2: Алгоритмы случайных перестановок\n(лог. шкала)", fontsize=10)
ax4.set_xlabel("n (размер перестановки)", fontsize=10)
ax4.set_ylabel("Время (мс, log)", fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(True, linestyle="--", alpha=0.6, which="both")

plt.savefig("permutations_comparison.png", dpi=150, bbox_inches="tight")
print("\nГрафик сохранён в permutations_comparison.png")
plt.show()
