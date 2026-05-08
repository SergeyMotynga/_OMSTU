"""
Лабораторная работа 8.

Тема: жадные алгоритмы.

Задача 1:
Дан набор предметов, каждый предмет можно взять не более одного раза.
У предмета есть номер, вес и стоимость. Есть рюкзак вместимостью W (целое).
Требуется выбрать предметы так, чтобы суммарная стоимость была максимальной.

Решения:
1) Полный перебор (точное решение).
2) Жадный алгоритм (по убыванию value/weight).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@dataclass(frozen=True)
class Item:
    """Предмет рюкзака."""

    item_id: str
    weight: int
    value: int

    @property
    def density(self) -> float:
        """Стоимость за 1 кг."""
        return self.value / self.weight


def brute_force_knapsack(
    items: Sequence[Item],
    capacity: int,
) -> tuple[list[Item], int, int]:
    """
    Полный перебор всех подмножеств (0/1 knapsack, точный метод).

    Сложность:
    - Время: O(2^n * n)
    - Память: O(1) + размер ответа
    """
    n = len(items)
    best_value = 0
    best_weight = 0
    best_mask = 0

    for mask in range(1 << n):
        total_weight = 0
        total_value = 0
        feasible = True

        for i, item in enumerate(items):
            if mask & (1 << i):
                total_weight += item.weight
                if total_weight > capacity:
                    feasible = False
                    break
                total_value += item.value

        if not feasible:
            continue

        if (
            total_value > best_value
            or (total_value == best_value and total_weight < best_weight)
            or (
                total_value == best_value
                and total_weight == best_weight
                and mask < best_mask
            )
        ):
            best_value = total_value
            best_weight = total_weight
            best_mask = mask

    selected = [items[i] for i in range(n) if best_mask & (1 << i)]
    return selected, best_weight, best_value


def greedy_knapsack_by_density(
    items: Sequence[Item],
    capacity: int,
) -> tuple[list[Item], int, int]:
    """
    Жадный алгоритм для 0/1 рюкзака:
    сортировка по value/weight и последовательный отбор, пока помещается.

    Сложность:
    - Время: O(n log n)
    - Память: O(n) из-за сортировки
    """
    sorted_items = sorted(
        items,
        key=lambda it: (-it.density, -it.value, it.weight, it.item_id),
    )

    total_weight = 0
    total_value = 0
    selected: list[Item] = []

    for item in sorted_items:
        if total_weight + item.weight <= capacity:
            selected.append(item)
            total_weight += item.weight
            total_value += item.value

    return selected, total_weight, total_value


def measure_ms(func, *args, repeats: int = 5) -> float:
    """Среднее время выполнения функции в миллисекундах."""
    func(*args)  # прогрев
    total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args)
        total += time.perf_counter() - t0
    return total / repeats * 1000.0


def generate_random_items(
    n: int,
    weight_min: int = 1,
    weight_max: int = 30,
    value_min: int = 10,
    value_max: int = 250,
) -> list[Item]:
    """Случайная генерация предметов для экспериментов."""
    if n <= 0:
        return []
    if weight_min <= 0:
        raise ValueError("weight_min must be > 0")
    if weight_min > weight_max:
        raise ValueError("weight_min must be <= weight_max")
    if value_min > value_max:
        raise ValueError("value_min must be <= value_max")

    items: list[Item] = []
    for idx in range(1, n + 1):
        weight = random.randint(weight_min, weight_max)
        value = random.randint(value_min, value_max)
        items.append(Item(item_id=f"I{idx}", weight=weight, value=value))
    return items


def print_solution(method_name: str, selected: Sequence[Item], weight: int, value: int) -> None:
    """Печать выбранного набора."""
    ids = [item.item_id for item in selected]
    print(f"{method_name}:")
    print(f"  Выбранные предметы: {ids}")
    print(f"  Суммарный вес: {weight}")
    print(f"  Суммарная стоимость: {value}")


def print_demo_case() -> None:
    """
    Демонстрация на классическом примере,
    где жадный 0/1 рюкзак дает не оптимальный ответ.
    """
    print("Демонстрационный пример")
    capacity = 50
    items = [
        Item("A", weight=10, value=60),
        Item("B", weight=20, value=100),
        Item("C", weight=30, value=120),
    ]

    print(f"W = {capacity}")
    print("Предметы (id, вес, стоимость, стоимость/вес):")
    for item in items:
        print(
            f"  {item.item_id}: w={item.weight}, v={item.value}, "
            f"v/w={item.density:.2f}"
        )

    exact_sel, exact_w, exact_v = brute_force_knapsack(items, capacity)
    greedy_sel, greedy_w, greedy_v = greedy_knapsack_by_density(items, capacity)

    print_solution("Полный перебор (точно)", exact_sel, exact_w, exact_v)
    print_solution("Жадный алгоритм", greedy_sel, greedy_w, greedy_v)
    if greedy_v < exact_v:
        print(f"Жадный проиграл оптимуму на {exact_v - greedy_v} по стоимости.")
    else:
        print("В этом примере жадный совпал с оптимумом.")
    print()


def run_benchmark() -> tuple[list[int], dict[str, list[float]], list[float]]:
    """
    Сравнение времени полного перебора и жадного метода
    на возрастающем n (n ограничено из-за экспоненты 2^n).

    Возвращает:
    - sizes: размеры входа n
    - timing: таблица времен по методам
    - quality: качество жадного (% от оптимума)
    """
    sizes = [8, 10, 12, 14, 16, 18, 20]
    timing = {
        "Полный перебор": [],
        "Жадный": [],
    }
    quality: list[float] = []

    for n in sizes:
        items = generate_random_items(n)
        total_weight = sum(item.weight for item in items)
        capacity = max(1, total_weight // 2)

        repeats_bruteforce = 3 if n <= 14 else (2 if n <= 16 else 1)
        repeats_greedy = 10

        brute_ms = measure_ms(brute_force_knapsack, items, capacity, repeats=repeats_bruteforce)
        greedy_ms = measure_ms(
            greedy_knapsack_by_density,
            items,
            capacity,
            repeats=repeats_greedy,
        )

        timing["Полный перебор"].append(brute_ms)
        timing["Жадный"].append(greedy_ms)

        _, _, exact_value = brute_force_knapsack(items, capacity)
        _, _, greedy_value = greedy_knapsack_by_density(items, capacity)
        quality_percent = (greedy_value / exact_value * 100.0) if exact_value > 0 else 100.0
        quality.append(quality_percent)

        print(
            f"n={n}: замер выполнен "
            f"(перебор repeats={repeats_bruteforce}, жадный repeats={repeats_greedy})"
        )

    return sizes, timing, quality


def print_benchmark_table(
    sizes: Sequence[int],
    timing: dict[str, list[float]],
    quality: Sequence[float],
) -> None:
    """Печать таблицы времени и качества жадного метода."""
    print("\nСравнение времени (мс) и качества жадного:")
    header = (
        f"{'n':>6}"
        f"{'Полный перебор':>18}"
        f"{'Жадный':>14}"
        f"{'Качество жадного':>20}"
    )
    print(header)

    for i, n in enumerate(sizes):
        brute_ms = timing["Полный перебор"][i]
        greedy_ms = timing["Жадный"][i]
        q = quality[i]
        print(f"{n:>6}{brute_ms:>15.4f} ms{greedy_ms:>11.4f} ms{q:>16.2f} %")


def plot_benchmark(
    sizes: Sequence[int],
    timing: dict[str, list[float]],
    quality: Sequence[float],
) -> None:
    """Строит и сохраняет график времени и качества."""
    if plt is None:
        print("\nmatplotlib не установлен, график пропущен.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("ЛР8: 0/1 рюкзак, полный перебор vs жадный", fontsize=13, fontweight="bold")

    axes[0].plot(
        sizes,
        timing["Полный перебор"],
        marker="o",
        linewidth=2,
        label="Полный перебор",
        color="#d62728",
    )
    axes[0].plot(
        sizes,
        timing["Жадный"],
        marker="s",
        linewidth=2,
        label="Жадный",
        color="#1f77b4",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Число предметов n")
    axes[0].set_ylabel("Время (мс, log)")
    axes[0].set_title("Скорость алгоритмов")
    axes[0].grid(True, linestyle="--", alpha=0.6, which="both")
    axes[0].legend(fontsize=9)

    axes[1].plot(
        sizes,
        quality,
        marker="D",
        linewidth=2,
        color="#2ca02c",
        label="Качество жадного",
    )
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("Число предметов n")
    axes[1].set_ylabel("Качество (% от оптимума)")
    axes[1].set_title("Точность жадного решения")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=9)

    output_path = Path(__file__).with_name("lab8_knapsack_runtime.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик сохранен: {output_path.name}")


def print_complexity_notes() -> None:
    """Печать теоретической сложности."""
    print("\nТеоретическая сложность:")
    print("Полный перебор: O(2^n * n), память O(1) + ответ.")
    print("Жадный алгоритм: O(n log n) из-за сортировки, память O(n).")
    print(
        "Примечание: для 0/1 рюкзака жадный метод может быть не оптимальным, "
        "но работает существенно быстрее."
    )


def main() -> None:
    """Запуск ЛР8: демонстрация, замеры времени, оценка сложности."""
    random.seed(42)

    print("Лабораторная работа 8")
    print("Задача 1: рюкзак (полный перебор и жадный алгоритм)")
    print("-" * 68)

    print_demo_case()

    sizes, timing, quality = run_benchmark()
    print_benchmark_table(sizes, timing, quality)
    print_complexity_notes()
    plot_benchmark(sizes, timing, quality)


if __name__ == "__main__":
    main()

