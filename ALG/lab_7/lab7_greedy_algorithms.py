"""
Лабораторная работа 7.

Тема: жадные алгоритмы.

Задача 1:
Имеются m дней и n заказов. Каждый заказ выполняется ровно за 1 день и имеет номер, стоимость и дедлайн. В каждый день можно выполнять только один заказ. Нужно выбрать набор заказов с максимальной суммарной стоимостью.

Задача 2:
Имеется n детей, у каждого есть номер и возраст.
Нужно разбить детей на минимальное число групп так, чтобы
в каждой группе разница возрастов была не больше 2 лет.
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
class Order:
    """Заказ для задачи 1."""

    order_id: str
    deadline: int
    value: int


@dataclass(frozen=True)
class Child:
    """Ребенок для задачи 2."""

    child_id: str
    age: int


def schedule_orders_max_value(
    days: int,
    orders: Sequence[Order],
) -> tuple[list[Order | None], list[Order], int]:
    """
    Жадный алгоритм для задачи 1.

    Идея:
    1) Сортируем заказы по убыванию стоимости.
    2) Каждый заказ ставим в самый поздний свободный день <= deadline.

    Возвращает:
    - расписание по дням (длина = days, может содержать None),
    - список выбранных заказов в календарном порядке,
    - максимальную суммарную стоимость.
    """
    if days <= 0:
        return [], [], 0

    useful_orders = [order for order in orders if order.deadline > 0 and order.value > 0]
    useful_orders.sort(key=lambda item: (-item.value, item.deadline, item.order_id))

    parent = list(range(days + 1))
    schedule: list[Order | None] = [None] * (days + 1)
    total_value = 0

    def find_set(day: int) -> int:
        while parent[day] != day:
            parent[day] = parent[parent[day]]
            day = parent[day]
        return day

    for order in useful_orders:
        latest_possible_day = min(order.deadline, days)
        free_day = find_set(latest_possible_day)

        if free_day > 0:
            schedule[free_day] = order
            total_value += order.value
            parent[free_day] = find_set(free_day - 1)

    selected_in_order = [item for item in schedule[1:] if item is not None]
    return schedule[1:], selected_in_order, total_value


def group_children_min_count(children: Sequence[Child]) -> list[list[Child]]:
    """
    Жадный алгоритм для задачи 2.

    Идея:
    1) Сортируем детей по возрасту.
    2) Берем самого младшего нераспределенного ребенка и создаем новую группу.
    3) Добавляем в эту группу всех следующих детей с возрастом <= (age_start + 2).
    4) Повторяем, пока дети не закончатся.

    Такой подход минимизирует число групп.
    """
    if not children:
        return []

    sorted_children = sorted(children, key=lambda item: (item.age, item.child_id))
    groups: list[list[Child]] = []

    i = 0
    n = len(sorted_children)
    while i < n:
        group_start_age = sorted_children[i].age
        max_age_in_group = group_start_age + 2
        current_group: list[Child] = []

        while i < n and sorted_children[i].age <= max_age_in_group:
            current_group.append(sorted_children[i])
            i += 1

        groups.append(current_group)

    return groups


def measure_ms(func, *args, repeats: int = 7) -> float:
    """Среднее время выполнения в миллисекундах."""
    func(*args)  # прогрев
    total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args)
        total += time.perf_counter() - t0
    return total / repeats * 1000.0


def generate_random_orders(n: int, days: int) -> list[Order]:
    """Генерация случайных заказов для бенчмарка задачи 1."""
    if n <= 0:
        return []
    if days <= 0:
        raise ValueError("days must be positive")

    orders: list[Order] = []
    for idx in range(1, n + 1):
        deadline = random.randint(1, max(days * 2, 1))
        value = random.randint(1, 10_000)
        orders.append(Order(order_id=f"J{idx}", deadline=deadline, value=value))
    return orders


def generate_random_children(n: int, min_age: int = 4, max_age: int = 14) -> list[Child]:
    """Генерация случайных детей для бенчмарка задачи 2."""
    if n <= 0:
        return []
    if min_age > max_age:
        raise ValueError("min_age must be <= max_age")

    children: list[Child] = []
    for idx in range(1, n + 1):
        age = random.randint(min_age, max_age)
        children.append(Child(child_id=f"C{idx}", age=age))
    return children


def print_demo_task_1() -> None:
    """Демонстрация задачи 1 на примере из условия."""
    print("Задача 1: расписание заказов")

    days = 3
    orders = [
        Order("A", deadline=2, value=40),
        Order("B", deadline=1, value=25),
        Order("C", deadline=2, value=30),
        Order("D", deadline=1, value=15),
        Order("E", deadline=3, value=20),
    ]

    schedule, selected, total_value = schedule_orders_max_value(days, orders)

    print("Вход:")
    print("  m = 3 дня")
    print("  Заказы:")
    print("  A(d=2,v=40), B(d=1,v=25), C(d=2,v=30), D(d=1,v=15), E(d=3,v=20)")

    print("Оптимальный выбор (в порядке дней):", [order.order_id for order in selected])
    print("Расписание:")
    for day, order in enumerate(schedule, start=1):
        if order is None:
            print(f"  День {day}: -")
        else:
            print(f"  День {day}: {order.order_id} (d={order.deadline}, v={order.value})")
    print(f"Суммарная стоимость: {total_value}")
    print()


def print_demo_task_2() -> None:
    """Демонстрация задачи 2."""
    print("Задача 2: группировка детей по возрасту")

    children = [
        Child("1", age=4),
        Child("2", age=5),
        Child("3", age=6),
        Child("4", age=7),
        Child("5", age=9),
        Child("6", age=10),
        Child("7", age=11),
        Child("8", age=13),
    ]

    groups = group_children_min_count(children)

    print("Вход (номер, возраст):")
    print(" ", [(child.child_id, child.age) for child in children])
    print(f"Минимальное число групп: {len(groups)}")
    for idx, group in enumerate(groups, start=1):
        as_pairs = [(child.child_id, child.age) for child in group]
        print(f"  Группа {idx}: {as_pairs}")
    print()


def run_benchmark() -> tuple[list[int], dict[str, list[float]]]:
    """Замеры времени для обеих задач."""
    sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000]
    timing = {
        "Задача 1": [],
        "Задача 2": [],
    }

    for n in sizes:
        days = max(1, n // 8)
        repeats = 10 if n <= 10_000 else (6 if n <= 100_000 else 3)

        orders = generate_random_orders(n, days=days)
        children = generate_random_children(n)

        timing["Задача 1"].append(
            measure_ms(schedule_orders_max_value, days, orders, repeats=repeats)
        )
        timing["Задача 2"].append(
            measure_ms(group_children_min_count, children, repeats=repeats)
        )

        print(f"n={n}: замер выполнен (repeats={repeats})")

    return sizes, timing


def print_benchmark_table(sizes: Sequence[int], timing: dict[str, list[float]]) -> None:
    """Вывод таблицы времени."""
    print("\nСравнение времени (мс):")
    header = f"{'n':>10}{'Задача 1':>14}{'Задача 2':>14}"
    print(header)

    for i, n in enumerate(sizes):
        t1 = timing["Задача 1"][i]
        t2 = timing["Задача 2"][i]
        print(f"{n:>10}{t1:>11.4f} ms{t2:>11.4f} ms")


def plot_benchmark(sizes: Sequence[int], timing: dict[str, list[float]]) -> None:
    """Построение и сохранение графика времени."""
    if plt is None:
        print("\nmatplotlib не установлен, график пропущен.")
        return

    colors = {"Задача 1": "#1f77b4", "Задача 2": "#2ca02c"}
    markers = {"Задача 1": "o", "Задача 2": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ЛР7: жадные алгоритмы, сравнение времени", fontsize=13, fontweight="bold")

    for ax, yscale, title in [
        (axes[0], "linear", "Линейная шкала"),
        (axes[1], "log", "Логарифмическая шкала"),
    ]:
        for name in ("Задача 1", "Задача 2"):
            ax.plot(
                sizes,
                timing[name],
                color=colors[name],
                marker=markers[name],
                linewidth=2,
                markersize=6,
                label=name,
            )
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        ax.set_xlabel("Размер входа n")
        ax.set_ylabel("Время (мс)" + (" log" if yscale == "log" else ""))
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.6, which="both")
        ax.legend(fontsize=9)

    output_path = Path(__file__).with_name("lab7_greedy_runtime.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик сохранен: {output_path.name}")


def print_complexity_notes() -> None:
    """Теоретическая оценка сложности."""
    print("\nТеоретическая сложность:")
    print(
        "Задача 1: O(n log n + n * alpha(m)) ~= O(n log n), "
        "где alpha - обратная функция Аккермана."
    )
    print("Задача 2: O(n log n) из-за сортировки, после нее один линейный проход O(n).")


def main() -> None:
    """Запуск демонстраций и замеров лабораторной работы 7."""
    random.seed(42)

    print("Лабораторная работа 7: жадные алгоритмы")
    print("-" * 64)

    print_demo_task_1()
    print_demo_task_2()

    sizes, timing = run_benchmark()
    print_benchmark_table(sizes, timing)
    print_complexity_notes()
    plot_benchmark(sizes, timing)


if __name__ == "__main__":
    main()
