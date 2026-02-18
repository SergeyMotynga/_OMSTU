"""
Лабораторная работа №2
Сравнение алгоритмов сортировки: пузырьковая, гномья, вставками
Зависимость времени от числа элементов n на двух типах данных:
  - почти отсортированные
  - плохо отсортированные (обратный порядок)
"""

import time
import matplotlib.pyplot as plt


def bubble_sort(arr):
    a = arr[:]
    n = len(a)
    for i in range(n - 1):
        for j in range(n - 1 - i):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a


def gnome_sort(arr):
    a = arr[:]
    i = 0
    while i < len(a):
        if i == 0 or a[i] >= a[i - 1]:
            i += 1
        else:
            a[i], a[i - 1] = a[i - 1], a[i]
            i -= 1
    return a


def insertion_sort(arr):
    a = arr[:]
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


NEARLY_SORTED_BASE = [1, 2, 4, 3, 5, 6, 7, 9, 8, 10,
                      11, 12, 14, 13, 15, 17, 16, 18]

POORLY_SORTED_BASE = [18, 16, 17, 13, 15, 14, 10, 12,
                      11, 7, 9, 8, 4, 6, 5, 1, 3, 2]


def scale_array(base: list, target_n: int) -> list:
    """Масштабирует базовый массив до нужного размера n,
    сохраняя характер данных (почти/плохо отсортированные)."""
    base_len = len(base)
    if target_n <= base_len:
        return base[:target_n]
    result = []
    offset = 0
    max_val = max(base)
    while len(result) < target_n:
        result.extend([x + offset for x in base])
        offset += max_val
    return result[:target_n]


def measure_time(sort_func, arr, repeats: int = 5) -> float:
    """Возвращает среднее время сортировки в миллисекундах."""
    total = 0.0
    for _ in range(repeats):
        data = arr.copy()
        t0 = time.perf_counter()
        sort_func(data)
        total += time.perf_counter() - t0
    return (total / repeats) * 1000


SIZES = [18, 50, 100, 300, 500, 800, 1000, 2000, 3000, 5000]

algorithms = {
    "Пузырьковая": bubble_sort,
    "Гномья":      gnome_sort,
    "Вставками":   insertion_sort,
}

datasets = {
    "Почти отсортированные": NEARLY_SORTED_BASE,
    "Плохо отсортированные": POORLY_SORTED_BASE,
}

results = {ds_name: {alg_name: [] for alg_name in algorithms}
           for ds_name in datasets}

print("Замер времени сортировки...\n")
for ds_name, base in datasets.items():
    print(f"  Тип данных: {ds_name}")
    for n in SIZES:
        arr = scale_array(base, n)
        for alg_name, func in algorithms.items():
            t = measure_time(func, arr)
            results[ds_name][alg_name].append(t)
        print(f"    n={n:>5}: готово")
    print()


col_w = 14
alg_count = len(algorithms)
ds_count = len(datasets)
total_w = 8 + col_w * alg_count * ds_count

print(f"{'':8}", end="")
for ds_name in datasets:
    label = f"[ {ds_name} ]"
    print(f"{label:^{col_w * alg_count}}", end="")
print()

print(f"{'n':>6}  ", end="")
for ds_name in datasets:
    for alg_name in algorithms:
        print(f"{alg_name[:12]:>{col_w}}", end="")
print()

print("-" * total_w)

for i, n in enumerate(SIZES):
    print(f"{n:>6}  ", end="")
    for ds_name in datasets:
        for alg_name in algorithms:
            t = results[ds_name][alg_name][i]
            print(f"{t:>{col_w - 2}.4f}мс", end="")
    print()

print("-" * total_w)

print("\nИсходные массивы (18 элементов):")
print("  Почти отсортированные:", NEARLY_SORTED_BASE)
print("  Плохо отсортированные:", POORLY_SORTED_BASE)

colors = {"Пузырьковая": "#e74c3c", "Гномья": "#2ecc71", "Вставками": "#3498db"}
markers = {"Пузырьковая": "o", "Гномья": "s", "Вставками": "^"}

ds_names = list(datasets.keys())

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Сравнение алгоритмов сортировки\nЗависимость времени от числа элементов",
             fontsize=14, fontweight="bold")

for col, ds_name in enumerate(ds_names):
    ax_lin = axes[0][col]
    for alg_name in algorithms:
        times = results[ds_name][alg_name]
        ax_lin.plot(SIZES, times,
                    label=alg_name,
                    color=colors[alg_name],
                    marker=markers[alg_name],
                    linewidth=2, markersize=6)
    ax_lin.set_title(f"{ds_name}\n(линейная шкала)", fontsize=11)
    ax_lin.set_xlabel("n (число элементов)", fontsize=10)
    ax_lin.set_ylabel("Время (мс)", fontsize=10)
    ax_lin.legend(fontsize=9)
    ax_lin.grid(True, linestyle="--", alpha=0.6)
    ax_lin.set_xticks(SIZES)
    ax_lin.tick_params(axis="x", rotation=45)

    ax_log = axes[1][col]
    for alg_name in algorithms:
        times = results[ds_name][alg_name]
        ax_log.plot(SIZES, times,
                    label=alg_name,
                    color=colors[alg_name],
                    marker=markers[alg_name],
                    linewidth=2, markersize=6)
    ax_log.set_yscale("log")
    ax_log.set_title(f"{ds_name}\n(логарифмическая шкала)", fontsize=11)
    ax_log.set_xlabel("n (число элементов)", fontsize=10)
    ax_log.set_ylabel("Время (мс, log)", fontsize=10)
    ax_log.legend(fontsize=9)
    ax_log.grid(True, linestyle="--", alpha=0.6, which="both")
    ax_log.set_xticks(SIZES)
    ax_log.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("sorting_comparison.png", dpi=150, bbox_inches="tight")
print("\nГрафик сохранён в sorting_comparison.png")
plt.show()
