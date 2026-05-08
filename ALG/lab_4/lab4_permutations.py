"""
Лабораторная работа 4.

В этом файле решаются две задачи:
1) Генерация всех перестановок чисел от 1 до n через алгоритм Штейнхауса-Джонсона-Троттера (SJT).
2) Сравнение нескольких алгоритмов генерации случайной перестановки по времени выполнения.

Скрипт выводит результаты в консоль и строит графики.
"""

import math
import random
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def sjt_permutations_iter(n):
    """
    Генерирует все перестановки 1..n с помощью алгоритма SJT.

    Идея SJT:
    - каждому элементу назначается направление (влево или вправо),
    - на каждом шаге выбирается наибольший "подвижный" элемент,
    - он меняется местами с соседним элементом по направлению,
    - направления некоторых элементов разворачиваются.

    Параметры:
    n (int): размер перестановки.

    Возвращает:
    iterator[list[int]]: перестановки по одной, без хранения всех в памяти.

    Сложность:
    - генерация одной следующей перестановки требует O(n),
    - всего перестановок n!, поэтому полная генерация O(n * n!).
    """
    if n == 0:
        yield []
        return
    if n == 1:
        yield [1]
        return

    perm = list(range(1, n + 1))
    dirs = [-1] * n
    yield perm[:]

    while True:
        mob_pos = -1
        mob_val = -1

        for i in range(n):
            j = i + dirs[i]
            if 0 <= j < n and perm[i] > perm[j] and perm[i] > mob_val:
                mob_val = perm[i]
                mob_pos = i

        if mob_pos == -1:
            break

        j = mob_pos + dirs[mob_pos]
        perm[mob_pos], perm[j] = perm[j], perm[mob_pos]
        dirs[mob_pos], dirs[j] = dirs[j], dirs[mob_pos]

        for i in range(n):
            if perm[i] > mob_val:
                dirs[i] = -dirs[i]

        yield perm[:]


def sjt_permutations(n):
    """
    Возвращает все перестановки 1..n в виде списка.

    Это удобная обертка над sjt_permutations_iter.
    Используется там, где нужно сразу получить все значения,
    например для демонстрации в консоли.

    Параметры:
    n (int): размер перестановки.

    Возвращает:
    list[list[int]]: список всех перестановок.
    """
    return list(sjt_permutations_iter(n))


def swap_position(prev, cur):
    """
    Находит первую позицию, где две перестановки отличаются.

    Для соседних перестановок SJT это будет левая позиция обмена
    двух соседних элементов.

    Параметры:
    prev (list[int]): предыдущая перестановка.
    cur (list[int]): текущая перестановка.

    Возвращает:
    int: индекс позиции (0-based), либо -1 если различий нет.
    """
    for i in range(len(prev)):
        if prev[i] != cur[i]:
            return i
    return -1


def fisher_yates(n):
    """
    Генерирует случайную перестановку методом Fisher-Yates.

    Параметры:
    n (int): размер перестановки.

    Возвращает:
    list[int]: случайная перестановка чисел 1..n.

    Сложность:
    O(n).
    """
    a = list(range(1, n + 1))
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        a[i], a[j] = a[j], a[i]
    return a


def inside_out(n):
    """
    Генерирует случайную перестановку методом "inside-out".

    Это вариант Fisher-Yates, который строит перестановку с нуля,
    постепенно добавляя элементы.

    Параметры:
    n (int): размер перестановки.

    Возвращает:
    list[int]: случайная перестановка чисел 1..n.

    Сложность:
    O(n).
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
    Генерирует случайную перестановку через сортировку по случайным ключам.

    Алгоритм:
    - создаем числа 1..n,
    - даем каждому числу случайный ключ,
    - сортируем по этим ключам.

    Параметры:
    n (int): размер перестановки.

    Возвращает:
    list[int]: случайная перестановка чисел 1..n.

    Сложность:
    O(n log n).
    """
    a = list(range(1, n + 1))
    keys = [random.random() for _ in range(n)]
    a.sort(key=lambda x: keys[x - 1])
    return a


def python_builtin(n):
    """
    Генерирует случайную перестановку через random.shuffle.

    Параметры:
    n (int): размер перестановки.

    Возвращает:
    list[int]: случайная перестановка чисел 1..n.

    Сложность:
    O(n).
    """
    a = list(range(1, n + 1))
    random.shuffle(a)
    return a


def measure_ms(func, *args, repeats=10):
    """
    Измеряет среднее время выполнения функции в миллисекундах.

    Сначала выполняется один "прогревочный" запуск,
    затем repeats запусков для расчета среднего времени.

    Параметры:
    func (callable): функция для измерения.
    *args: аргументы функции func.
    repeats (int): число повторов.

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


def measure_sjt_generation_ms(n):
    """
    Измеряет время полной генерации всех перестановок SJT без накопления.

    Это важно, чтобы не хранить в памяти все n! перестановок,
    особенно для больших n.

    Параметры:
    n (int): размер перестановки.

    Возвращает:
    tuple[float, int]:
    - elapsed_ms: время генерации в миллисекундах,
    - count: количество сгенерированных перестановок.
    """
    count = 0
    t0 = time.perf_counter()
    for _ in sjt_permutations_iter(n):
        count += 1
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms, count


def repeats_for_size(n):
    """
    Подбирает число повторов замера по размеру задачи.

    Для маленьких n повторов больше (стабильнее среднее),
    для больших n повторов меньше (быстрее общий запуск).

    Параметры:
    n (int): размер входных данных.

    Возвращает:
    int: рекомендуемое число повторов.
    """
    if n <= 10_000:
        return 15
    if n <= 100_000:
        return 7
    return 3


def main():
    """
    Запускает демонстрацию и сравнение алгоритмов из лабораторной 4.

    Что делает функция:
    1) Печатает примеры SJT для n=3 и n=4.
    2) Измеряет время полной генерации n! перестановок SJT.
    3) Сравнивает 4 алгоритма случайной перестановки.
    4) Выводит таблицы и сохраняет график в файл.
    """
    random.seed(42)

    print("Лабораторная работа 4")
    print("Задача 1. Генерация перестановок методом SJT")

    for demo_n in [3, 4]:
        perms = sjt_permutations(demo_n)
        print(f"\nn={demo_n}, количество перестановок: {len(perms)}")
        print("Перестановка и переход")

        for idx, p in enumerate(perms):
            if idx == 0:
                tag = "начальная"
            else:
                pos = swap_position(perms[idx - 1], p)
                tag = f"своп позиций {pos + 1} и {pos + 2}"
            print(f"{p}  {tag}")

    print("\nВремя генерации всех n! перестановок методом SJT")
    print(f"{'n':>4} {'n!':>10} {'время, мс':>12} {'нс/перест.':>14}")

    sjt_sizes = list(range(3, 12))
    sjt_max_factorial = 10_000_000

    sjt_sizes_done = []
    sjt_fact_done = []
    sjt_times = []

    for n in sjt_sizes:
        fact = math.factorial(n)
        if fact > sjt_max_factorial:
            print(f"{n:>4} {fact:>10} {'пропуск':>12} {'слишком долго':>14}")
            continue

        elapsed_ms, count = measure_sjt_generation_ms(n)
        ns_each = elapsed_ms * 1e6 / count

        sjt_sizes_done.append(n)
        sjt_fact_done.append(fact)
        sjt_times.append(elapsed_ms)

        print(f"{n:>4} {fact:>10} {elapsed_ms:>12.3f} {ns_each:>14.2f}")

    print("\nЗадача 2. Сравнение алгоритмов случайных перестановок")

    rand_sizes = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]

    random_algos = {
        "Fisher-Yates": fisher_yates,
        "Inside-Out F-Y": inside_out,
        "Naive Sort": naive_sort_shuffle,
        "Python built-in": python_builtin,
    }

    rand_results = {name: [] for name in random_algos}

    for n in rand_sizes:
        reps = repeats_for_size(n)
        for name, func in random_algos.items():
            rand_results[name].append(measure_ms(func, n, repeats=reps))
        print(f"n={n}: замер выполнен, repeats={reps}")

    col_w = 17
    header_names = list(random_algos.keys())

    print("\nРезультаты (мс):")
    header = f"{'n':>9}"
    for name in header_names:
        header += f"{name[:col_w-2]:>{col_w}}"
    print(header)

    for i, n in enumerate(rand_sizes):
        row = f"{n:>9}"
        for name in header_names:
            row += f"{rand_results[name][i]:>{col_w - 3}.4f} мс"
        print(row)

    print("\nСложность алгоритмов:")
    complexities = {
        "Fisher-Yates": "O(n)",
        "Inside-Out F-Y": "O(n)",
        "Naive Sort": "O(n log n)",
        "Python built-in": "O(n)",
    }
    for name, c in complexities.items():
        print(f"{name:<18} {c}")

    colors = {
        "Fisher-Yates": "#e74c3c",
        "Inside-Out F-Y": "#2ecc71",
        "Naive Sort": "#e67e22",
        "Python built-in": "#3498db",
    }
    markers = {
        "Fisher-Yates": "o",
        "Inside-Out F-Y": "s",
        "Naive Sort": "^",
        "Python built-in": "D",
    }

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "Лабораторная работа 4\nАнализ алгоритмов генерации перестановок",
        fontsize=14,
        fontweight="bold",
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sjt_sizes_done, sjt_times, color="#9b59b6", marker="o", linewidth=2, markersize=7, label="SJT")
    ax1.set_title("SJT: время генерации всех перестановок", fontsize=10)
    ax1.set_xlabel("n", fontsize=10)
    ax1.set_ylabel("Время (мс)", fontsize=10)
    ax1.set_xticks(sjt_sizes_done)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    ns_per_perm = [sjt_times[i] * 1e6 / sjt_fact_done[i] for i in range(len(sjt_sizes_done))]
    ax2.plot(sjt_sizes_done, ns_per_perm, color="#1abc9c", marker="s", linewidth=2, markersize=7, label="нс/перест.")
    ax2.set_title("SJT: накладные расходы на одну перестановку", fontsize=10)
    ax2.set_xlabel("n", fontsize=10)
    ax2.set_ylabel("нс/перест.", fontsize=10)
    ax2.set_xticks(sjt_sizes_done)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(gs[1, 0])
    for name in random_algos:
        ax3.plot(
            rand_sizes,
            rand_results[name],
            label=name,
            color=colors[name],
            marker=markers[name],
            linewidth=2,
            markersize=5,
        )
    ax3.set_title("Случайные перестановки, линейная шкала", fontsize=10)
    ax3.set_xlabel("n", fontsize=10)
    ax3.set_ylabel("Время (мс)", fontsize=10)
    ax3.set_xticks(rand_sizes)
    ax3.tick_params(axis="x", rotation=35)
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.6)

    ax4 = fig.add_subplot(gs[1, 1])
    for name in random_algos:
        ax4.plot(
            rand_sizes,
            rand_results[name],
            label=name,
            color=colors[name],
            marker=markers[name],
            linewidth=2,
            markersize=5,
        )
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_title("Случайные перестановки, логарифмическая шкала", fontsize=10)
    ax4.set_xlabel("n", fontsize=10)
    ax4.set_ylabel("Время (мс, log)", fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.6, which="both")

    plt.savefig("permutations_comparison.png", dpi=150, bbox_inches="tight")
    print("\nГрафик сохранен: permutations_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
