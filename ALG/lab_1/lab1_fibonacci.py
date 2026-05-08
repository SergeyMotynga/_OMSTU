"""
Лабораторная работа №1
Сравнительный анализ вычисления n-го элемента последовательности Фибоначчи:
рекурсия vs итерация. График: n, время.
"""

import time
import matplotlib.pyplot as plt


def fib_recursive(n):
    """Вычисляет n-й элемент Фибоначчи рекурсивно."""
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)


def fib_iterative(n):
    """Вычисляет n-й элемент Фибоначчи итеративно."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def measure_time(func, n, repeats=5):
    """Возвращает среднее время вычисления в миллисекундах."""
    total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(n)
        total += time.perf_counter() - t0
    return (total / repeats) * 1000


SIZES = list(range(1, 41))

times_recursive = []
times_iterative = []

print("Замер времени...\n")
for n in SIZES:
    t_rec = measure_time(fib_recursive, n)
    t_iter = measure_time(fib_iterative, n)
    times_recursive.append(t_rec)
    times_iterative.append(t_iter)
    print(f"n={n:>2}: рекурсия={t_rec:.6f}мс  итерация={t_iter:.6f}мс")

print("\nЗначения Фибоначчи:")
for n in SIZES:
    print(f"  F({n:>2}) = {fib_iterative(n)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Сравнение рекурсии и итерации: числа Фибоначчи\nЗависимость времени от n",
             fontsize=13, fontweight="bold")

axes[0].plot(SIZES, times_recursive, color="#e74c3c", marker="o",
             linewidth=2, markersize=5, label="Рекурсия")
axes[0].plot(SIZES, times_iterative, color="#3498db", marker="^",
             linewidth=2, markersize=5, label="Итерация")
axes[0].set_title("Линейная шкала", fontsize=11)
axes[0].set_xlabel("n", fontsize=10)
axes[0].set_ylabel("Время (мс)", fontsize=10)
axes[0].legend(fontsize=9)
axes[0].grid(True, linestyle="--", alpha=0.6)

axes[1].plot(SIZES, times_recursive, color="#e74c3c", marker="o",
             linewidth=2, markersize=5, label="Рекурсия")
axes[1].plot(SIZES, times_iterative, color="#3498db", marker="^",
             linewidth=2, markersize=5, label="Итерация")
axes[1].set_yscale("log")
axes[1].set_title("Логарифмическая шкала", fontsize=11)
axes[1].set_xlabel("n", fontsize=10)
axes[1].set_ylabel("Время (мс, log)", fontsize=10)
axes[1].legend(fontsize=9)
axes[1].grid(True, linestyle="--", alpha=0.6, which="both")

plt.tight_layout()
plt.savefig("fibonacci_comparison.png", dpi=150, bbox_inches="tight")
print("\nГрафик сохранён в fibonacci_comparison.png")
plt.show()
