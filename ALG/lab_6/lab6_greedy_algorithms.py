"""
Лабораторная работа 6.

Тема: жадные алгоритмы.

Задачи:
1) На прямой даны n отрезков. Нужно выбрать максимальное
   по размеру подмножество непересекающихся отрезков.
2) Для аудитории дан набор заявок (время начала и конца).
   Нужно принять максимальное количество заявок.
3) Отрезки [li, ri] покрывают интервал [L, R].
   Нужно выбрать минимальное подмножество отрезков,
   которое всё ещё покрывает [L, R].

Скрипт содержит:
- решения всех трёх задач,
- демонстрационные примеры,
- оценку времени выполнения,
- график сравнения.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

Interval = tuple[int, int]


def _normalize_intervals(intervals: Sequence[Interval]) -> list[Interval]:
    """Нормализует отрезки так, чтобы left <= right."""
    normalized: list[Interval] = []
    for left, right in intervals:
        if left <= right:
            normalized.append((left, right))
        else:
            normalized.append((right, left))
    return normalized


def max_non_overlapping_intervals(
    segments: Sequence[Interval],
    allow_touch: bool = True,
) -> list[Interval]:
    """
    Задача 1.
    Жадный выбор по возрастанию правой границы.

    Если allow_touch=True, отрезки [a, b] и [b, c]
    считаются совместимыми (касание разрешено).
    """
    sorted_segments = sorted(_normalize_intervals(segments), key=lambda seg: (seg[1], seg[0]))
    chosen: list[Interval] = []
    last_end = float("-inf")

    for left, right in sorted_segments:
        compatible = left >= last_end if allow_touch else left > last_end
        if compatible:
            chosen.append((left, right))
            last_end = right
    return chosen


def max_accepted_requests(requests: Sequence[Interval]) -> list[tuple[int, Interval]]:
    """
    Задача 2.
    Та же жадная идея интервального планирования, что и в задаче 1,
    но дополнительно возвращаются идентификаторы заявок.
    """
    indexed = [
        (start, finish, req_id)
        for req_id, (start, finish) in enumerate(_normalize_intervals(requests), start=1)
    ]
    indexed.sort(key=lambda item: (item[1], item[0], item[2]))

    accepted: list[tuple[int, Interval]] = []
    current_finish = float("-inf")

    for start, finish, req_id in indexed:
        if start >= current_finish:
            accepted.append((req_id, (start, finish)))
            current_finish = finish
    return accepted


def min_cover_intervals(
    intervals: Sequence[Interval],
    left_border: int,
    right_border: int,
) -> list[Interval] | None:
    """
    Задача 3.
    Минимальное число отрезков для покрытия [left_border, right_border].

    Возвращает:
    - список выбранных отрезков, если покрытие возможно,
    - None, если покрытие невозможно.
    """
    if left_border > right_border:
        raise ValueError("left_border должен быть <= right_border")
    if left_border == right_border:
        return []

    sorted_intervals = sorted(_normalize_intervals(intervals), key=lambda seg: (seg[0], -seg[1]))
    n = len(sorted_intervals)
    i = 0
    current = left_border
    chosen: list[Interval] = []

    while current < right_border:
        best_reach = current
        best_interval: Interval | None = None

        while i < n and sorted_intervals[i][0] <= current:
            left, right = sorted_intervals[i]
            if right > best_reach:
                best_reach = right
                best_interval = (left, right)
            i += 1

        if best_interval is None:
            return None

        chosen.append(best_interval)
        current = best_reach

    return chosen


def measure_ms(func, *args, repeats: int = 7) -> float:
    """Измеряет среднее время выполнения в миллисекундах."""
    func(*args)  # прогрев
    total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args)
        total += time.perf_counter() - t0
    return total / repeats * 1000.0


def generate_random_intervals(n: int, coord_factor: int = 5, max_len: int = 40) -> list[Interval]:
    """Генерирует случайные отрезки для замеров задач 1 и 2."""
    if n <= 0:
        return []

    right_bound = max(100, coord_factor * n)
    intervals: list[Interval] = []
    for _ in range(n):
        left = random.randint(0, right_bound)
        length = random.randint(1, max_len)
        intervals.append((left, left + length))
    return intervals


def generate_covering_intervals(n: int, left_border: int, right_border: int) -> list[Interval]:
    """Генерирует n отрезков с гарантированным покрытием [left_border, right_border]."""
    if n < 2:
        raise ValueError("Для генерации покрытия n должно быть >= 2")
    if left_border >= right_border:
        raise ValueError("left_border должен быть < right_border")

    span = right_border - left_border
    backbone_count = max(2, n // 4)
    backbone_count = min(backbone_count, n, span)

    intervals: list[Interval] = []
    current = left_border

    for idx in range(backbone_count):
        intervals_left = backbone_count - idx

        if intervals_left == 1:
            right = right_border
        else:
            max_step = (right_border - current) - (intervals_left - 1)
            step = random.randint(1, max_step)
            right = current + step

        jitter = max(1, (right - current) // 2)
        left = max(left_border, current - random.randint(0, jitter))
        intervals.append((left, right))
        current = right

    # Добавляем дополнительные «шумовые» отрезки,
    # при этом гарантированное покрытие уже обеспечено опорной цепочкой.
    while len(intervals) < n:
        left = random.randint(left_border, right_border - 1)
        max_extra = max(2, span // 10)
        right = random.randint(left + 1, min(right_border + max_extra, left + max_extra))
        intervals.append((left, right))

    return intervals


def print_demo_cases() -> None:
    """Выводит небольшие демонстрационные примеры для всех трёх задач."""
    print("Лабораторная работа 6: жадные алгоритмы")
    print("-" * 64)

    # Демонстрация задачи 1
    task1_segments = [
        (1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9),
        (6, 10), (8, 11), (8, 12), (2, 14), (12, 16),
    ]
    task1_answer = max_non_overlapping_intervals(task1_segments, allow_touch=True)
    print("Задача 1 (максимум непересекающихся отрезков):")
    print(f"  Отрезки: {task1_segments}")
    print(f"  Оптимальный набор ({len(task1_answer)}): {task1_answer}")
    print()

    # Демонстрация задачи 2
    task2_requests = [
        (9, 10), (9, 12), (10, 11), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)
    ]
    task2_answer = max_accepted_requests(task2_requests)
    print("Задача 2 (максимум заявок в аудиторию):")
    print(f"  Заявки: {list(enumerate(task2_requests, start=1))}")
    print(f"  Принятые заявки ({len(task2_answer)}): {task2_answer}")
    print()

    # Демонстрация задачи 3
    left_border, right_border = 0, 20
    task3_segments = [
        (-2, 3), (0, 5), (3, 9), (6, 12), (9, 15), (14, 20),
        (5, 11), (10, 18), (17, 22),
    ]
    task3_answer = min_cover_intervals(task3_segments, left_border, right_border)
    print("Задача 3 (минимальное покрытие интервала):")
    print(f"  Нужно покрыть: [{left_border}, {right_border}]")
    print(f"  Отрезки: {task3_segments}")
    if task3_answer is None:
        print("  Покрытие невозможно.")
    else:
        print(f"  Минимальное покрытие ({len(task3_answer)}): {task3_answer}")
    print("-" * 64)


def run_benchmark() -> tuple[list[int], dict[str, list[float]]]:
    """Запускает замеры времени для всех трёх задач на растущих n."""
    sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000]
    timing = {
        "Задача 1": [],
        "Задача 2": [],
        "Задача 3": [],
    }

    for n in sizes:
        repeats = 10 if n <= 10_000 else (6 if n <= 100_000 else 3)

        task1_segments = generate_random_intervals(n)
        task2_requests = generate_random_intervals(n)
        task3_segments = generate_covering_intervals(n, left_border=0, right_border=1_000_000)

        timing["Задача 1"].append(
            measure_ms(max_non_overlapping_intervals, task1_segments, True, repeats=repeats)
        )
        timing["Задача 2"].append(
            measure_ms(max_accepted_requests, task2_requests, repeats=repeats)
        )
        timing["Задача 3"].append(
            measure_ms(min_cover_intervals, task3_segments, 0, 1_000_000, repeats=repeats)
        )

        print(f"n={n}: замер выполнен (повторов={repeats})")

    return sizes, timing


def print_benchmark_table(sizes: Sequence[int], timing: dict[str, list[float]]) -> None:
    """Выводит таблицу времени выполнения (в мс)."""
    print("\nСравнение времени (мс):")
    header = f"{'n':>10}{'Задача 1':>14}{'Задача 2':>14}{'Задача 3':>14}"
    print(header)
    for i, n in enumerate(sizes):
        print(
            f"{n:>10}"
            f"{timing['Задача 1'][i]:>11.4f} ms"
            f"{timing['Задача 2'][i]:>11.4f} ms"
            f"{timing['Задача 3'][i]:>11.4f} ms"
        )


def plot_benchmark(sizes: Sequence[int], timing: dict[str, list[float]]) -> None:
    """Сохраняет график времени в линейной и логарифмической шкалах."""
    colors = {
        "Задача 1": "#1f77b4",
        "Задача 2": "#2ca02c",
        "Задача 3": "#d62728",
    }
    markers = {
        "Задача 1": "o",
        "Задача 2": "s",
        "Задача 3": "D",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ЛР6: сравнение времени жадных алгоритмов", fontsize=13, fontweight="bold")

    for ax, yscale, title in [
        (axes[0], "linear", "Линейная шкала"),
        (axes[1], "log", "Логарифмическая шкала"),
    ]:
        for task in ("Задача 1", "Задача 2", "Задача 3"):
            ax.plot(
                sizes,
                timing[task],
                color=colors[task],
                marker=markers[task],
                linewidth=2,
                markersize=6,
                label=task,
            )
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        ax.set_xlabel("Размер входа n")
        ax.set_ylabel("Время (мс)" + (" log" if yscale == "log" else ""))
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.6, which="both")
        ax.legend(fontsize=9)

    output_path = Path(__file__).with_name("lab6_greedy_runtime.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nГрафик сохранён: {output_path.name}")
    plt.show()


def main() -> None:
    """Запускает демонстрации и замеры для лабораторной работы 6."""
    random.seed(42)

    print_demo_cases()
    sizes, timing = run_benchmark()

    print_benchmark_table(sizes, timing)
    print("\nТеоретическая сложность:")
    print("Задача 1: O(n log n) за счёт сортировки (проход выбора O(n))")
    print("Задача 2: O(n log n) за счёт сортировки (проход выбора O(n))")
    print("Задача 3: O(n log n) за счёт сортировки (проход покрытия O(n))")

    plot_benchmark(sizes, timing)


if __name__ == "__main__":
    main()
