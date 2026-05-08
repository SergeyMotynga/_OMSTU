"""
Лабораторная работа 9.

Тема: жадные алгоритмы (раскраска графа).

Постановки из задания:
1) Распределение работ между механизмами:
   - есть работы a1..an и механизмы b1..bm,
   - каждая работа требует набор механизмов,
   - один механизм не может одновременно обслуживать несколько работ,
   - нужно минимизировать общее время выполнения (при равной длительности работ).
2) Размещение грузов по контейнерам:
   - некоторые грузы несовместимы,
   - нужно минимизировать число контейнеров.

Обе задачи сводятся к раскраске графа конфликтов:
- вершины: работы (или грузы),
- ребро: пара не может выполняться/размещаться вместе,
- цвет: временной слот (или контейнер).
"""

from __future__ import annotations

import random
import re
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


Graph = dict[str, set[str]]
_LABEL_RE = re.compile(r"^([^\d]+)(\d+)$")


def natural_label_key(label: str) -> tuple[str, int, str]:
    """Ключ сортировки меток вида a10, b2, g15."""
    match = _LABEL_RE.match(label)
    if match:
        return match.group(1), int(match.group(2)), label
    return label, -1, label


def ensure_undirected_graph(graph: Mapping[str, Iterable[str]]) -> Graph:
    """
    Строит неориентированный граф без петель.

    Если ребро u-v указано только с одной стороны, симметричная сторона
    добавляется автоматически.
    """
    undirected: Graph = {node: set(neighbors) for node, neighbors in graph.items()}

    for node in list(undirected):
        undirected[node].discard(node)
        for neighbor in list(undirected[node]):
            if neighbor == node:
                continue
            if neighbor not in undirected:
                undirected[neighbor] = set()
            undirected[neighbor].add(node)

    return undirected


def build_conflict_graph_from_mechanisms(
    mechanism_to_works: Mapping[str, Iterable[str]],
) -> Graph:
    """
    Строит граф конфликтов работ по матрице "механизм -> работы".

    Ребро между работами ai и aj существует, если есть хотя бы один механизм,
    который требуется обеим работам.
    """
    all_works: set[str] = set()
    for works in mechanism_to_works.values():
        all_works.update(works)

    graph: Graph = {work: set() for work in all_works}

    for works in mechanism_to_works.values():
        work_list = sorted(set(works), key=natural_label_key)
        for i in range(len(work_list)):
            wi = work_list[i]
            for j in range(i + 1, len(work_list)):
                wj = work_list[j]
                graph[wi].add(wj)
                graph[wj].add(wi)

    return graph


def greedy_coloring(graph: Mapping[str, Iterable[str]]) -> tuple[dict[str, int], list[str]]:
    """
    Жадная раскраска в порядке Welsh-Powell:
    1) сортировка вершин по убыванию степени;
    2) вершине присваивается минимальный допустимый цвет.

    Возвращает:
    - colors: словарь вершина -> номер цвета (с 1),
    - order: фактический порядок обхода вершин.
    """
    undirected = ensure_undirected_graph(graph)
    order = sorted(
        undirected.keys(),
        key=lambda node: (-len(undirected[node]), natural_label_key(node)),
    )

    colors: dict[str, int] = {}
    for node in order:
        forbidden_colors = {colors[nbr] for nbr in undirected[node] if nbr in colors}
        color = 1
        while color in forbidden_colors:
            color += 1
        colors[node] = color

    return colors, order


def color_classes(colors: Mapping[str, int]) -> dict[int, list[str]]:
    """Преобразует раскраску вершина->цвет в группы цвет->список вершин."""
    classes: dict[int, list[str]] = {}
    for node, color in colors.items():
        classes.setdefault(color, []).append(node)
    for color in classes:
        classes[color].sort(key=natural_label_key)
    return dict(sorted(classes.items()))


def verify_coloring(graph: Mapping[str, Iterable[str]], colors: Mapping[str, int]) -> bool:
    """Проверяет корректность раскраски: соседние вершины не должны иметь одинаковый цвет."""
    undirected = ensure_undirected_graph(graph)
    for node, neighbors in undirected.items():
        for nbr in neighbors:
            if colors.get(node) == colors.get(nbr):
                return False
    return True


def solve_work_distribution(
    mechanism_to_works: Mapping[str, Iterable[str]],
) -> tuple[Graph, dict[str, int], dict[int, list[str]]]:
    """Решает задачу распределения работ между механизмами через раскраску графа."""
    conflict_graph = build_conflict_graph_from_mechanisms(mechanism_to_works)
    colors, _ = greedy_coloring(conflict_graph)
    classes = color_classes(colors)
    return conflict_graph, colors, classes


def solve_cargo_placement(
    incompatibility_graph: Mapping[str, Iterable[str]],
) -> tuple[Graph, dict[str, int], dict[int, list[str]]]:
    """Решает задачу размещения грузов по контейнерам через раскраску графа несовместимости."""
    graph = ensure_undirected_graph(incompatibility_graph)
    colors, _ = greedy_coloring(graph)
    classes = color_classes(colors)
    return graph, colors, classes


def measure_ms(func, *args, repeats: int = 5) -> float:
    """Среднее время выполнения функции в миллисекундах."""
    func(*args)  # прогрев
    total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args)
        total += time.perf_counter() - t0
    return total / repeats * 1000.0


def generate_random_work_mechanism_instance(
    num_works: int,
    num_mechanisms: int,
    min_mechanisms_per_work: int = 2,
    max_mechanisms_per_work: int = 4,
) -> dict[str, set[str]]:
    """
    Случайный генератор данных для постановки "работы/механизмы".

    Формат результата: {mechanism: {works...}}.
    """
    if num_works <= 0 or num_mechanisms <= 0:
        return {}
    if min_mechanisms_per_work <= 0:
        raise ValueError("min_mechanisms_per_work must be > 0")
    if min_mechanisms_per_work > max_mechanisms_per_work:
        raise ValueError("min_mechanisms_per_work must be <= max_mechanisms_per_work")

    mechanisms = [f"b{i}" for i in range(1, num_mechanisms + 1)]
    mechanism_to_works: dict[str, set[str]] = {m: set() for m in mechanisms}

    for idx in range(1, num_works + 1):
        work = f"a{idx}"
        degree = random.randint(min_mechanisms_per_work, max_mechanisms_per_work)
        degree = min(degree, num_mechanisms)
        selected_mechanisms = random.sample(mechanisms, k=degree)
        for mechanism in selected_mechanisms:
            mechanism_to_works[mechanism].add(work)

    return mechanism_to_works


def generate_random_incompatibility_graph(num_nodes: int, avg_degree: int = 8) -> Graph:
    """
    Случайный разреженный граф несовместимости для постановки "грузы/контейнеры".

    Генерация выполняется за O(n * avg_degree), без полного O(n^2) перебора пар.
    """
    if num_nodes <= 0:
        return {}
    if avg_degree <= 0:
        raise ValueError("avg_degree must be > 0")

    nodes = [f"g{i}" for i in range(1, num_nodes + 1)]
    graph: Graph = {node: set() for node in nodes}

    min_deg = max(1, avg_degree - 2)
    max_deg = max(min_deg, avg_degree + 2)

    for idx, node in enumerate(nodes):
        degree_target = random.randint(min_deg, max_deg)
        for _ in range(degree_target):
            j = random.randrange(num_nodes)
            if j == idx:
                continue
            other = nodes[j]
            graph[node].add(other)
            graph[other].add(node)

    return graph


def solve_work_distribution_instance(mechanism_to_works: Mapping[str, Iterable[str]]) -> int:
    """Вспомогательная функция для замеров: возвращает число цветовых классов."""
    _, _, classes = solve_work_distribution(mechanism_to_works)
    return len(classes)


def solve_cargo_instance(incompatibility_graph: Mapping[str, Iterable[str]]) -> int:
    """Вспомогательная функция для замеров: возвращает число контейнеров (цветов)."""
    _, _, classes = solve_cargo_placement(incompatibility_graph)
    return len(classes)


def print_demo_task_1() -> None:
    """
    Демонстрация задачи 1 с данными из изображения в методичке (ЛР9).
    """
    print("Задача 1: распределение работ между механизмами")

    mechanism_to_works = {
        "b1": {"a1", "a4", "a6", "a10"},
        "b2": {"a2", "a3", "a7", "a9"},
        "b3": {"a2", "a5", "a8"},
        "b4": {"a6", "a9"},
        "b5": {"a1", "a5", "a9"},
        "b6": {"a4", "a8", "a10"},
        "b7": {"a3", "a4", "a6"},
        "b8": {"a1", "a8"},
        "b9": {"a3", "a4", "a6", "a7", "a10"},
    }

    conflict_graph, colors, slots = solve_work_distribution(mechanism_to_works)
    ok = verify_coloring(conflict_graph, colors)

    print("Матрица требований (механизм -> работы):")
    for mechanism in sorted(mechanism_to_works, key=natural_label_key):
        works = sorted(mechanism_to_works[mechanism], key=natural_label_key)
        print(f"  {mechanism}: {works}")

    print("\nПолученный конфликтный граф работ (степени вершин):")
    for work in sorted(conflict_graph, key=natural_label_key):
        print(f"  {work}: degree={len(conflict_graph[work])}")

    print("\nЖадная раскраска (work -> слот):")
    for work in sorted(colors, key=natural_label_key):
        print(f"  {work} -> t{colors[work]}")

    print("\nПараллельное расписание по временным слотам:")
    for slot, works in slots.items():
        print(f"  Слот {slot}: {works}")

    duration_per_work = 1.0
    total_time = len(slots) * duration_per_work
    print(f"\nЧисло слотов = {len(slots)}")
    print(f"Общее время = {len(slots)} * {duration_per_work:g} = {total_time:g}")
    print(f"Проверка корректности раскраски: {'OK' if ok else 'ERROR'}")
    print()


def print_demo_task_2() -> None:
    """
    Демонстрация задачи 2 (размещение грузов по контейнерам).
    """
    print("Задача 2: размещение грузов по контейнерам")

    # Небольшой пример несовместимости грузов.
    incompatibility: Graph = {
        "g1": {"g2", "g5", "g8"},
        "g2": {"g1", "g3", "g6"},
        "g3": {"g2", "g4", "g7"},
        "g4": {"g3", "g8"},
        "g5": {"g1", "g6", "g7"},
        "g6": {"g2", "g5", "g8"},
        "g7": {"g3", "g5", "g8"},
        "g8": {"g1", "g4", "g6", "g7"},
    }

    graph, colors, containers = solve_cargo_placement(incompatibility)
    ok = verify_coloring(graph, colors)

    print("Ребра несовместимости (граф):")
    for node in sorted(graph, key=natural_label_key):
        neighbors = sorted(graph[node], key=natural_label_key)
        print(f"  {node}: {neighbors}")

    print("\nРазмещение грузов (cargo -> container):")
    for cargo in sorted(colors, key=natural_label_key):
        print(f"  {cargo} -> c{colors[cargo]}")

    print("\nКомпоновка по контейнерам:")
    for container, cargos in containers.items():
        print(f"  Контейнер {container}: {cargos}")

    print(f"\nМинимизация по жадному алгоритму: использовано контейнеров = {len(containers)}")
    print(f"Проверка корректности раскраски: {'OK' if ok else 'ERROR'}")
    print()


def run_benchmark() -> tuple[list[int], dict[str, list[float]]]:
    """
    Бенчмарк времени для двух постановок (на растущих n).
    """
    sizes = [500, 1_000, 2_000, 5_000, 10_000, 20_000]
    timing = {
        "Работы/механизмы": [],
        "Грузы/контейнеры": [],
    }

    for n in sizes:
        repeats = 8 if n <= 2_000 else (4 if n <= 10_000 else 2)

        num_mechanisms = max(50, n // 4)
        mechanism_to_works = generate_random_work_mechanism_instance(
            num_works=n,
            num_mechanisms=num_mechanisms,
            min_mechanisms_per_work=2,
            max_mechanisms_per_work=4,
        )
        cargo_graph = generate_random_incompatibility_graph(n, avg_degree=8)

        t_work = measure_ms(
            solve_work_distribution_instance,
            mechanism_to_works,
            repeats=repeats,
        )
        t_cargo = measure_ms(
            solve_cargo_instance,
            cargo_graph,
            repeats=repeats,
        )

        timing["Работы/механизмы"].append(t_work)
        timing["Грузы/контейнеры"].append(t_cargo)
        print(f"n={n}: замер выполнен (repeats={repeats})")

    return sizes, timing


def print_benchmark_table(sizes: Sequence[int], timing: Mapping[str, Sequence[float]]) -> None:
    """Печать таблицы времени выполнения."""
    print("\nСравнение времени (мс):")
    header = f"{'n':>10}{'Работы/механизмы':>20}{'Грузы/контейнеры':>20}"
    print(header)

    for i, n in enumerate(sizes):
        t_work = timing["Работы/механизмы"][i]
        t_cargo = timing["Грузы/контейнеры"][i]
        print(f"{n:>10}{t_work:>17.4f} ms{t_cargo:>17.4f} ms")


def plot_benchmark(sizes: Sequence[int], timing: Mapping[str, Sequence[float]]) -> None:
    """Построение и сохранение графика времени работы алгоритма."""
    if plt is None:
        print("\nmatplotlib не установлен, график пропущен.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ЛР9: жадная раскраска графа, сравнение времени", fontsize=13, fontweight="bold")

    colors = {
        "Работы/механизмы": "#1f77b4",
        "Грузы/контейнеры": "#2ca02c",
    }
    markers = {
        "Работы/механизмы": "o",
        "Грузы/контейнеры": "s",
    }

    for ax, yscale, title in [
        (axes[0], "linear", "Линейная шкала"),
        (axes[1], "log", "Логарифмическая шкала"),
    ]:
        for name in ("Работы/механизмы", "Грузы/контейнеры"):
            ax.plot(
                sizes,
                timing[name],
                marker=markers[name],
                color=colors[name],
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

    output_path = Path(__file__).with_name("lab9_graph_coloring_runtime.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик сохранен: {output_path.name}")


def print_complexity_notes() -> None:
    """Теоретическая оценка сложности."""
    print("\nТеоретическая сложность:")
    print(
        "1) Построение графа конфликтов (работы/механизмы): "
        "O(sum_k |W_k|^2), где W_k — работы, использующие механизм k."
    )
    print(
        "2) Жадная раскраска (Welsh-Powell): O(V log V + E) "
        "для разреженных графов, в худшем случае O(V^2)."
    )
    print(
        "3) Размещение грузов по контейнерам — та же раскраска графа несовместимости: "
        "O(V log V + E)."
    )


def main() -> None:
    """Запуск демонстраций, замеров времени и построения графика."""
    random.seed(42)

    print("Лабораторная работа 9: жадные алгоритмы (раскраска графа)")
    print("-" * 72)

    print_demo_task_1()
    print_demo_task_2()

    sizes, timing = run_benchmark()
    print_benchmark_table(sizes, timing)
    print_complexity_notes()
    plot_benchmark(sizes, timing)


if __name__ == "__main__":
    main()
