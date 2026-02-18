"""
Задание 4 — Покупка канцелярских принадлежностей.
Определить, какие товары и в каком количестве может купить студент
в пределах бюджета при условии, что число наименований максимально.
"""

import time

BUDGET = 250

STUDENT_LIST = {
    "карандаш": 3,
    "ручка":    2,
    "тетрадь":  1,
    "ластик":   4,
    "линейка":  1,
    "маркер":   2,
}

PRICES = {
    "карандаш": 12,
    "ручка":    30,
    "тетрадь":  95,
    "ластик":    8,
    "линейка":  25,
    "маркер":   45,
}


def solve_stationery(budget, needs, prices):
    """
    Перебирает все 2^n подмножеств (bitmask).
    Находит подмножество с максимальным числом наименований,
    суммарная стоимость которого не превышает бюджет.
    """
    items = list(needs.keys())
    n = len(items)
    best_count = 0
    best_subset = []
    best_cost = 0

    for mask in range(1, 1 << n):
        subset = [items[i] for i in range(n) if mask & (1 << i)]
        cost = sum(prices[item] * needs[item] for item in subset)
        if cost <= budget and len(subset) > best_count:
            best_count = len(subset)
            best_subset = subset[:]
            best_cost = cost

    return best_subset, best_cost


print("═" * 55)
print("Задание 4: Покупка канцелярских принадлежностей")
print("═" * 55)

print(f"\nБюджет студента: {BUDGET} руб.")
print(f"\n{'Товар':<12} {'Кол-во':>6}  {'Цена':>6}  {'Итого':>7}")
print("-" * 38)
for item, qty in STUDENT_LIST.items():
    total = PRICES[item] * qty
    print(f"{item:<12} {qty:>6} шт  {PRICES[item]:>5} руб  {total:>6} руб")
print(f"\nПолная стоимость: {sum(PRICES[i]*STUDENT_LIST[i] for i in STUDENT_LIST)} руб")

t0 = time.perf_counter()
best, cost = solve_stationery(BUDGET, STUDENT_LIST, PRICES)
elapsed = (time.perf_counter() - t0) * 1000

print(f"\nОптимальная покупка: {len(best)} наименований на {cost} руб")
print(f"Время перебора: {elapsed:.4f} мс\n")
print(f"{'Товар':<12} {'Кол-во':>6}  {'Цена':>6}  {'Итого':>7}")
print("-" * 38)
for item in best:
    qty   = STUDENT_LIST[item]
    total = PRICES[item] * qty
    print(f"{item:<12} {qty:>6} шт  {PRICES[item]:>5} руб  {total:>6} руб")
print("-" * 38)
print(f"{'ИТОГО':<12} {'':>6}  {'':>6}  {cost:>6} руб")
print(f"Остаток: {BUDGET - cost} руб")
