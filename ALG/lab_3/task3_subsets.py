"""
Задание 3 — Все возможные выборки из n элементов.
Перебрать выборки по 1 элементу, по 2, ..., по n элементов.
"""

import time
from itertools import combinations


def all_subsets(elements):
    """
    Генерирует все непустые подмножества элементов,
    сгруппированные по размеру выборки.
    """
    result = []
    for r in range(1, len(elements) + 1):
        for combo in combinations(elements, r):
            result.append(list(combo))
    return result


elements = ["стол", "стул", "шкаф"]

print("═" * 50)
print("Задание 3: Все возможные выборки")
print("═" * 50)

t0 = time.perf_counter()
subsets = all_subsets(elements)
elapsed = (time.perf_counter() - t0) * 1000

print(f"\nЭлементы: {elements}")
print(f"Всего вариантов: {len(subsets)}  |  время: {elapsed:.4f} мс\n")

for r in range(1, len(elements) + 1):
    group = [s for s in subsets if len(s) == r]
    print(f"Выборки по {r} элемент(а/ов):")
    for s in group:
        print(f"  {s}")
    print()
