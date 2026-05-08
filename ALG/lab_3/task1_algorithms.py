"""
Задание 1 — Три алгоритма генерации перестановок.
Сравнение времени выполнения одного шага для n от 1 до 10000 с шагом 50.
"""

import time
import matplotlib.pyplot as plt


def narayana_next(perm):
    """
    Алгоритм Нарайяны: следующая перестановка в лексикографическом порядке.
    Изменяет perm на месте. Возвращает True если следующая перестановка существует.
    """
    n = len(perm)
    i = n - 2
    while i >= 0 and perm[i] >= perm[i + 1]:
        i -= 1
    if i < 0:
        return False
    j = n - 1
    while perm[j] <= perm[i]:
        j -= 1
    perm[i], perm[j] = perm[j], perm[i]
    perm[i + 1:] = perm[i + 1:][::-1]
    return True


def johnson_trotter_step(perm, direction):
    """
    Один шаг алгоритма Джонсона-Троттера.
    direction[v] — направление элемента v (-1 = влево, +1 = вправо).
    Изменяет perm и direction на месте. Возвращает True если шаг выполнен.
    """
    n = len(perm)
    mobile_pos, mobile_val = -1, -1
    for i in range(n):
        v = perm[i]
        j = i + direction[v]
        if 0 <= j < n and perm[j] < v and v > mobile_val:
            mobile_val, mobile_pos = v, i
    if mobile_pos == -1:
        return False
    j = mobile_pos + direction[mobile_val]
    perm[mobile_pos], perm[j] = perm[j], perm[mobile_pos]
    for v in range(mobile_val + 1, n + 1):
        direction[v] = -direction[v]
    return True


def inversion_increment(inv):
    """
    Инкремент вектора инверсий (факториальная система счисления).
    inv[i] ∈ [0, n-1-i]. Возвращает True если инкремент успешен.
    """
    n = len(inv)
    i = n - 1
    while i >= 0:
        inv[i] += 1
        if inv[i] <= n - 1 - i:
            return True
        inv[i] = 0
        i -= 1
    return False


def inversion_to_perm(inv):
    """
    Преобразует вектор инверсий в перестановку.
    """
    available = list(range(1, len(inv) + 1))
    perm = []
    for c in inv:
        perm.append(available[c])
        available.pop(c)
    return perm


def generate_all_narayana(n):
    perm = list(range(1, n + 1))
    result = [perm[:]]
    while narayana_next(perm):
        result.append(perm[:])
    return result


def generate_all_jt(n):
    perm = list(range(1, n + 1))
    direction = [0] + [-1] * n
    result = [perm[:]]
    while johnson_trotter_step(perm, direction):
        result.append(perm[:])
    return result


def generate_all_inversion(n):
    inv = [0] * n
    result = [inversion_to_perm(inv)]
    while inversion_increment(inv):
        result.append(inversion_to_perm(inv))
    return result



n_demo = 4

t0 = time.perf_counter()
perms_nar = generate_all_narayana(n_demo)
t_nar_demo = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
perms_jt = generate_all_jt(n_demo)
t_jt_demo = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
perms_inv = generate_all_inversion(n_demo)
t_inv_demo = (time.perf_counter() - t0) * 1000

print("═" * 55)
print(f"Демонстрация: все перестановки n={n_demo}  ({len(perms_nar)} штук)")
print("═" * 55)

print(f"\nНарайяна  ({t_nar_demo:.4f} мс):")
for p in perms_nar:
    print(" ", p)

print(f"\nДжонсон-Троттер  ({t_jt_demo:.4f} мс):")
for p in perms_jt:
    print(" ", p)

print(f"\nВектор инверсий  ({t_inv_demo:.4f} мс):")
for p in perms_inv:
    print(" ", p)



SIZES   = list(range(1, 10001, 50))
REPEATS = 5

t_narayana  = []
t_jt_times  = []
t_inv_times = []

print("\nЗамер времени одного шага (n = 1..10000, шаг 50)...")

for idx, n in enumerate(SIZES):
    identity = list(range(1, n + 1))
    dir_init = [0] + [-1] * n

    total = 0.0
    for _ in range(REPEATS):
        perm = identity[:]
        t0 = time.perf_counter()
        narayana_next(perm)
        total += time.perf_counter() - t0
    t_narayana.append(total / REPEATS * 1000)

    total = 0.0
    for _ in range(REPEATS):
        perm      = identity[:]
        direction = dir_init[:]
        t0 = time.perf_counter()
        johnson_trotter_step(perm, direction)
        total += time.perf_counter() - t0
    t_jt_times.append(total / REPEATS * 1000)

    total = 0.0
    for _ in range(REPEATS):
        inv = [0] * n
        t0 = time.perf_counter()
        inversion_increment(inv)
        total += time.perf_counter() - t0
    t_inv_times.append(total / REPEATS * 1000)

    if (idx + 1) % 50 == 0:
        print(f"  n={n}: готово")

print("Замер завершён\n")



fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Задание 1: Время одного шага алгоритма (n = 1..10000)",
    fontsize=12, fontweight="bold"
)

for ax, yscale, title in [
    (axes[0], "linear", "Линейная шкала"),
    (axes[1], "log",    "Логарифмическая шкала"),
]:
    ax.plot(SIZES, t_narayana,  color="#e74c3c", linewidth=2, label="Нарайяна")
    ax.plot(SIZES, t_jt_times,  color="#2ecc71", linewidth=2, label="Джонсон-Троттер")
    ax.plot(SIZES, t_inv_times, color="#3498db", linewidth=2, label="Вектор инверсий")
    ax.set_yscale(yscale)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("n", fontsize=10)
    ax.set_ylabel("Время (мс)" + (" log" if yscale == "log" else ""), fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6, which="both" if yscale == "log" else "major")

plt.tight_layout()
plt.savefig("task1_comparison.png", dpi=150, bbox_inches="tight")
print("График сохранён в task1_comparison.png")
plt.show()
