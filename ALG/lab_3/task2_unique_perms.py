"""
Задание 2 — Уникальные перестановки с повторяющимися элементами.
Каждая сгенерированная подпоследовательность должна быть уникальной.
"""

import time


def unique_permutations(lst):
    """
    Генерирует все уникальные перестановки списка с повторами.
    Использует backtracking + множество seen для пропуска дублей.
    """
    result = []

    def backtrack(current, remaining):
        if not remaining:
            result.append(current[:])
            return
        seen = set()
        for i in range(len(remaining)):
            if remaining[i] in seen:
                continue
            seen.add(remaining[i])
            current.append(remaining[i])
            backtrack(current, remaining[:i] + remaining[i + 1:])
            current.pop()

    backtrack([], sorted(lst))
    return result


examples = [
    [1, 2, 1],
    [1, 2, 3],
    [1, 1, 1],
    [1, 1, 2, 2],
    [1, 2, 3, 4, 5],
]

print("═" * 50)
print("Задание 2: Уникальные перестановки с повторами")
print("═" * 50)

for seq in examples:
    t0 = time.perf_counter()
    perms = unique_permutations(seq)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\nПоследовательность: {seq}")
    print(f"Уникальных перестановок: {len(perms)}  |  время: {elapsed:.4f} мс")
    for p in perms:
        print(" ", p)
