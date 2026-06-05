from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable, Sequence
from xml.sax.saxutils import escape


VARIANT = 17

CAUCHY_A = -1.0
CAUCHY_B = 1.0
CAUCHY_H = 0.4
CAUCHY_Y0 = 1.0

BVP_A = 0.5
BVP_B = 0.8
BVP_H = 0.1


@dataclass
class Point:
    x: float
    y: float


@dataclass
class CauchyNode:
    i: int
    x: float
    y: float
    exact: float
    absolute_error: float


@dataclass
class CauchyStep:
    i: int
    x: float
    y: float
    x_next: float
    y_next: float
    slope: float | None = None
    predictor: float | None = None
    corrected_slope: float | None = None
    k0: float | None = None
    k1: float | None = None
    k2: float | None = None
    k3: float | None = None


@dataclass
class CauchySolution:
    key: str
    name: str
    short_name: str
    order: int
    h: float
    nodes: list[CauchyNode]
    steps: list[CauchyStep]
    max_absolute_error: float
    richardson_error_h2: float


@dataclass
class TridiagonalRow:
    k: int
    x: float
    a: float
    b: float
    c: float
    d: float


@dataclass
class SweepRow:
    k: int
    denominator: float
    u: float
    v: float


@dataclass
class BvpNode:
    k: int
    x: float
    y: float
    residual: float


@dataclass
class BvpSolution:
    nodes: list[BvpNode]
    coefficients: list[TridiagonalRow]
    sweep: list[SweepRow]
    left_derivative: float
    right_derivative: float
    max_residual: float


@dataclass
class GraphSeries:
    name: str
    color: str
    points: list[Point]
    width: float = 2.6
    dasharray: str | None = None


def f_cauchy(x: float, y: float) -> float:
    return y - 5.0 * x


def exact_cauchy(x: float) -> float:
    return math.exp(x + 1.0) + 5.0 * x + 5.0


def p_bvp(x: float) -> float:
    return 2.0 * x


def q_bvp(_: float) -> float:
    return 1.0


def rhs_bvp(_: float) -> float:
    return 1.0


def _fmt(value: float, digits: int = 6) -> str:
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}"


def _fmt_compact(value: float, digits: int = 6) -> str:
    return _fmt(value, digits).rstrip("0").rstrip(".")


def _fmt_ru(value: float, digits: int = 6) -> str:
    return _fmt(value, digits).replace(".", ",").replace("-", "−")


def _fmt_ru_compact(value: float, digits: int = 6) -> str:
    return _fmt_compact(value, digits).replace(".", ",").replace("-", "−")


def _fmt_ru_sign(value: float, digits: int = 6) -> str:
    sign = "+" if value >= 0.0 else "−"
    return f" {sign} {_fmt_ru(abs(value), digits)}"


def grid_nodes(left: float, right: float, h: float) -> list[float]:
    count = int(round((right - left) / h))
    return [round(left + i * h, 10) for i in range(count + 1)]


def solve_cauchy_raw(method_key: str, h: float) -> tuple[list[CauchyNode], list[CauchyStep]]:
    xs = grid_nodes(CAUCHY_A, CAUCHY_B, h)
    y = CAUCHY_Y0
    nodes = [
        CauchyNode(
            i=0,
            x=xs[0],
            y=y,
            exact=exact_cauchy(xs[0]),
            absolute_error=abs(y - exact_cauchy(xs[0])),
        )
    ]
    steps: list[CauchyStep] = []

    for i in range(len(xs) - 1):
        x = xs[i]
        x_next = xs[i + 1]
        if method_key == "euler":
            slope = f_cauchy(x, y)
            y_next = y + h * slope
            steps.append(
                CauchyStep(
                    i=i,
                    x=x,
                    y=y,
                    x_next=x_next,
                    y_next=y_next,
                    slope=slope,
                )
            )
        elif method_key == "modified_euler":
            slope = f_cauchy(x, y)
            predictor = y + h * slope
            corrected_slope = f_cauchy(x_next, predictor)
            y_next = y + h * (slope + corrected_slope) / 2.0
            steps.append(
                CauchyStep(
                    i=i,
                    x=x,
                    y=y,
                    x_next=x_next,
                    y_next=y_next,
                    slope=slope,
                    predictor=predictor,
                    corrected_slope=corrected_slope,
                )
            )
        elif method_key == "runge_kutta":
            k0 = h * f_cauchy(x, y)
            k1 = h * f_cauchy(x + h / 2.0, y + k0 / 2.0)
            k2 = h * f_cauchy(x + h / 2.0, y + k1 / 2.0)
            k3 = h * f_cauchy(x + h, y + k2)
            y_next = y + (k0 + 2.0 * k1 + 2.0 * k2 + k3) / 6.0
            steps.append(
                CauchyStep(
                    i=i,
                    x=x,
                    y=y,
                    x_next=x_next,
                    y_next=y_next,
                    k0=k0,
                    k1=k1,
                    k2=k2,
                    k3=k3,
                )
            )
        else:
            raise ValueError(f"Unknown method: {method_key}")

        y = y_next
        exact = exact_cauchy(x_next)
        nodes.append(
            CauchyNode(
                i=i + 1,
                x=x_next,
                y=y,
                exact=exact,
                absolute_error=abs(y - exact),
            )
        )

    return nodes, steps


def richardson_error(method_key: str, order: int) -> float:
    coarse_nodes, _ = solve_cauchy_raw(method_key, CAUCHY_H)
    fine_nodes, _ = solve_cauchy_raw(method_key, CAUCHY_H / 2.0)
    denominator = 2**order - 1
    return max(
        abs(fine_nodes[2 * i].y - coarse_nodes[i].y) / denominator
        for i in range(len(coarse_nodes))
    )


def calculate_cauchy_solutions() -> list[CauchySolution]:
    specs = [
        ("euler", "Метод Эйлера", "Эйлер", 1),
        ("modified_euler", "Модифицированный метод Эйлера", "Мод. Эйлер", 2),
        ("runge_kutta", "Метод Рунге-Кутта 4-го порядка", "Рунге-Кутта", 4),
    ]
    solutions: list[CauchySolution] = []
    for key, name, short_name, order in specs:
        nodes, steps = solve_cauchy_raw(key, CAUCHY_H)
        solutions.append(
            CauchySolution(
                key=key,
                name=name,
                short_name=short_name,
                order=order,
                h=CAUCHY_H,
                nodes=nodes,
                steps=steps,
                max_absolute_error=max(node.absolute_error for node in nodes),
                richardson_error_h2=richardson_error(key, order),
            )
        )
    return solutions


def build_bvp_coefficients() -> list[TridiagonalRow]:
    xs = grid_nodes(BVP_A, BVP_B, BVP_H)
    h = BVP_H
    h2 = h * h
    rows = [
        TridiagonalRow(k=1, x=xs[0], a=0.0, b=-1.0 / h, c=1.0 / h, d=1.0)
    ]

    for index in range(1, len(xs) - 1):
        x = xs[index]
        p = p_bvp(x)
        q = q_bvp(x)
        rows.append(
            TridiagonalRow(
                k=index + 1,
                x=x,
                a=1.0 / h2 - p / (2.0 * h),
                b=-2.0 / h2 + q,
                c=1.0 / h2 + p / (2.0 * h),
                d=rhs_bvp(x),
            )
        )

    rows.append(
        TridiagonalRow(k=len(xs), x=xs[-1], a=-1.0 / h, b=1.0 / h, c=0.0, d=3.0)
    )
    return rows


def solve_tridiagonal(rows: Sequence[TridiagonalRow]) -> tuple[list[float], list[SweepRow]]:
    sweep: list[SweepRow] = []
    for index, row in enumerate(rows):
        if index == 0:
            denominator = row.b
            u = -row.c / denominator
            v = row.d / denominator
        else:
            prev = sweep[-1]
            denominator = row.a * prev.u + row.b
            u = -row.c / denominator
            v = (row.d - row.a * prev.v) / denominator
        sweep.append(SweepRow(k=row.k, denominator=denominator, u=u, v=v))

    ys = [0.0] * len(rows)
    ys[-1] = sweep[-1].v
    for index in range(len(rows) - 2, -1, -1):
        ys[index] = sweep[index].u * ys[index + 1] + sweep[index].v
    return ys, sweep


def row_residual(row: TridiagonalRow, ys: Sequence[float]) -> float:
    index = row.k - 1
    left = row.a * ys[index - 1] if index > 0 else 0.0
    center = row.b * ys[index]
    right = row.c * ys[index + 1] if index < len(ys) - 1 else 0.0
    return left + center + right - row.d


def calculate_bvp_solution() -> BvpSolution:
    xs = grid_nodes(BVP_A, BVP_B, BVP_H)
    coefficients = build_bvp_coefficients()
    ys, sweep = solve_tridiagonal(coefficients)
    nodes = [
        BvpNode(k=index + 1, x=x, y=y, residual=row_residual(coefficients[index], ys))
        for index, (x, y) in enumerate(zip(xs, ys))
    ]
    left_derivative = (ys[1] - ys[0]) / BVP_H
    right_derivative = (ys[-1] - ys[-2]) / BVP_H
    return BvpSolution(
        nodes=nodes,
        coefficients=coefficients,
        sweep=sweep,
        left_derivative=left_derivative,
        right_derivative=right_derivative,
        max_residual=max(abs(node.residual) for node in nodes),
    )


def cauchy_table_markdown(solutions: Sequence[CauchySolution]) -> str:
    lines = [
        "| i | xᵢ | точное y(xᵢ) | Эйлер | ошибка | мод. Эйлер | ошибка | Рунге-Кутта | ошибка |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index in range(len(solutions[0].nodes)):
        exact = solutions[0].nodes[index].exact
        cells = [
            str(index),
            _fmt_ru(solutions[0].nodes[index].x, 3),
            _fmt_ru(exact, 9),
        ]
        for solution in solutions:
            node = solution.nodes[index]
            cells.append(_fmt_ru(node.y, 9))
            cells.append(_fmt_ru(node.absolute_error, 9))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cauchy_first_step_block(solution: CauchySolution) -> list[str]:
    step = solution.steps[0]
    h = _fmt_ru(CAUCHY_H, 3)
    if solution.key == "euler":
        return [
            f"**{solution.name}:**",
            "",
            "`yᵢ₊₁ = yᵢ + h·f(xᵢ, yᵢ)`",
            "",
            f"`f(x₀, y₀) = {_fmt_ru(step.slope or 0.0, 9)}`",
            f"`y₁ = {_fmt_ru(step.y, 9)} + {h}·{_fmt_ru(step.slope or 0.0, 9)} = {_fmt_ru(step.y_next, 9)}`",
        ]
    if solution.key == "modified_euler":
        return [
            f"**{solution.name}:**",
            "",
            "`ỹᵢ₊₁ = yᵢ + h·f(xᵢ, yᵢ)`",
            "`yᵢ₊₁ = yᵢ + h·(f(xᵢ, yᵢ) + f(xᵢ₊₁, ỹᵢ₊₁)) / 2`",
            "",
            f"`ỹ₁ = {_fmt_ru(step.predictor or 0.0, 9)}`",
            f"`f(x₁, ỹ₁) = {_fmt_ru(step.corrected_slope or 0.0, 9)}`",
            f"`y₁ = {_fmt_ru(step.y_next, 9)}`",
        ]
    return [
        f"**{solution.name}:**",
        "",
        "`yᵢ₊₁ = yᵢ + (k₀ + 2k₁ + 2k₂ + k₃) / 6`",
        "",
        f"`k₀ = {_fmt_ru(step.k0 or 0.0, 9)}`",
        f"`k₁ = {_fmt_ru(step.k1 or 0.0, 9)}`",
        f"`k₂ = {_fmt_ru(step.k2 or 0.0, 9)}`",
        f"`k₃ = {_fmt_ru(step.k3 or 0.0, 9)}`",
        f"`y₁ = {_fmt_ru(step.y_next, 9)}`",
    ]


def cauchy_error_table_markdown(solutions: Sequence[CauchySolution]) -> str:
    lines = [
        "| метод | порядок p | max точная ошибка при h | оценка Δ для h/2 |",
        "|:---|---:|---:|---:|",
    ]
    for solution in solutions:
        lines.append(
            f"| {solution.name} | {solution.order} | "
            f"{_fmt_ru(solution.max_absolute_error, 9)} | {_fmt_ru(solution.richardson_error_h2, 9)} |"
        )
    return "\n".join(lines)


def bvp_coefficients_table_markdown(solution: BvpSolution) -> str:
    lines = [
        "| k | xₖ | aₖ | bₖ | cₖ | dₖ | Uₖ | Vₖ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row, sweep in zip(solution.coefficients, solution.sweep):
        lines.append(
            f"| {row.k} | {_fmt_ru(row.x, 3)} | {_fmt_ru(row.a, 6)} | "
            f"{_fmt_ru(row.b, 6)} | {_fmt_ru(row.c, 6)} | {_fmt_ru(row.d, 6)} | "
            f"{_fmt_ru(sweep.u, 9)} | {_fmt_ru(sweep.v, 9)} |"
        )
    return "\n".join(lines)


def bvp_solution_table_markdown(solution: BvpSolution) -> str:
    lines = [
        "| k | xₖ | yₖ | невязка строки |",
        "|---:|---:|---:|---:|",
    ]
    for node in solution.nodes:
        lines.append(
            f"| {node.k} | {_fmt_ru(node.x, 3)} | {_fmt_ru(node.y, 9)} | "
            f"{_fmt_ru(node.residual, 12)} |"
        )
    return "\n".join(lines)


def build_report(cauchy_solutions: Sequence[CauchySolution], bvp_solution: BvpSolution) -> str:
    best = min(cauchy_solutions, key=lambda item: item.max_absolute_error)
    lines: list[str] = []
    lines.append(f"# Лабораторная работа №8, вариант {VARIANT}")
    lines.append("")
    lines.append("## Задание")
    lines.append("")
    lines.append("В работе решаются две задачи из блока численного решения обыкновенных дифференциальных уравнений.")
    lines.append("")
    lines.append("**№ 7.1. Задача Коши:**")
    lines.append("")
    lines.append("`y' = y − 5x`, `y(−1) = 1`, `x ∈ [−1; 1]`, `h = 0,4`.")
    lines.append("")
    lines.append("Нужно найти сеточное решение методом Эйлера, модифицированным методом Эйлера и методом Рунге-Кутта 4-го порядка.")
    lines.append("")
    lines.append("**№ 7.2. Краевая задача:**")
    lines.append("")
    lines.append("`y'' + 2xy' + y = 1`, `y'(0,5) = 1`, `y'(0,8) = 3`, `h = 0,1`.")
    lines.append("")
    lines.append("Нужно построить конечно-разностную схему и решить полученную трехдиагональную систему методом прогонки.")
    lines.append("")

    lines.append("## 1) Задача Коши")
    lines.append("")
    lines.append("Правая часть уравнения:")
    lines.append("")
    lines.append("`f(x, y) = y − 5x`")
    lines.append("")
    lines.append("Для контроля точности удобно выписать аналитическое решение:")
    lines.append("")
    lines.append("`y' − y = −5x`")
    lines.append("")
    lines.append("`y = C·eˣ + 5x + 5`")
    lines.append("")
    lines.append("Из условия `y(−1)=1` получаем `C=e`, поэтому:")
    lines.append("")
    lines.append("`y(x) = eˣ⁺¹ + 5x + 5`")
    lines.append("")
    lines.append("Сетка:")
    lines.append("")
    lines.append(f"`xᵢ = −1 + i·{_fmt_ru(CAUCHY_H, 1)}`, `i = 0, ..., 5`.")
    lines.append("")
    lines.append("Первые шаги трех методов:")
    lines.append("")
    for solution in cauchy_solutions:
        lines.extend(cauchy_first_step_block(solution))
        lines.append("")
    lines.append("Итоговая таблица значений:")
    lines.append("")
    lines.append(cauchy_table_markdown(cauchy_solutions))
    lines.append("")
    lines.append("Оценка точности по правилу Рунге выполнялась сравнением расчетов с шагами `h` и `h/2`:")
    lines.append("")
    lines.append("`Δₕ/₂ = max|yᵢ^(h/2) − yᵢ^(h)| / (2ᵖ − 1)`")
    lines.append("")
    lines.append(cauchy_error_table_markdown(cauchy_solutions))
    lines.append("")
    lines.append("График решений:")
    lines.append("")
    lines.append("![cauchy solutions](cauchy_solutions.svg)")
    lines.append("")
    lines.append("График абсолютных ошибок:")
    lines.append("")
    lines.append("![cauchy errors](cauchy_errors.svg)")
    lines.append("")
    lines.append(
        f"На этой сетке наименьшую максимальную ошибку дал **{best.name}**: "
        f"`{_fmt_ru(best.max_absolute_error, 9)}`."
    )
    lines.append("")

    lines.append("## 2) Краевая задача")
    lines.append("")
    lines.append("Решение оформим в той же последовательности, что и в примере методички.")
    lines.append("")
    lines.append("**1. Определение сетки.**")
    lines.append("")
    lines.append("Отрезок `[0,5; 0,8]` делится с шагом `h = 0,1`:")
    lines.append("")
    lines.append("`x₁=0,5`, `x₂=0,6`, `x₃=0,7`, `x₄=0,8`.")
    lines.append("")
    lines.append("Здесь `x₁` и `x₄` — краевые точки, `x₂` и `x₃` — внутренние точки.")
    lines.append("")
    lines.append("**2. Определение сеточной функции.**")
    lines.append("")
    lines.append("В каждом узле вводим значение функции:")
    lines.append("")
    lines.append("| узел | x₁ | x₂ | x₃ | x₄ |")
    lines.append("|:---|---:|---:|---:|---:|")
    lines.append("| x | 0,5 | 0,6 | 0,7 | 0,8 |")
    lines.append("| y | y₁ = y(x₁) | y₂ = y(x₂) | y₃ = y(x₃) | y₄ = y(x₄) |")
    lines.append("")
    lines.append("**3. Аппроксимация уравнения.**")
    lines.append("")
    lines.append("Исходная задача:")
    lines.append("")
    lines.append("`y'' + 2xy' + y = 1`")
    lines.append("")
    lines.append("`y'(0,5)=1`, `y'(0,8)=3`.")
    lines.append("")
    lines.append("Для внутренних узлов используются центральные разности:")
    lines.append("")
    lines.append("`y'ₖ ≈ (yₖ₊₁ − yₖ₋₁)/(2h)`")
    lines.append("")
    lines.append("`y''ₖ ≈ (yₖ₋₁ − 2yₖ + yₖ₊₁)/h²`")
    lines.append("")
    lines.append("Для краевых условий используются односторонние разности.")
    lines.append("")
    lines.append("При `x₁ = 0,5`:")
    lines.append("")
    lines.append("`(y₂ − y₁) / 0,1 = 1`")
    lines.append("")
    lines.append("При `x₂ = 0,6`:")
    lines.append("")
    lines.append("`(y₁ − 2y₂ + y₃) / 0,1² + 2·0,6·(y₃ − y₁) / (2·0,1) + y₂ = 1`")
    lines.append("")
    lines.append("При `x₃ = 0,7`:")
    lines.append("")
    lines.append("`(y₂ − 2y₃ + y₄) / 0,1² + 2·0,7·(y₄ − y₂) / (2·0,1) + y₃ = 1`")
    lines.append("")
    lines.append("При `x₄ = 0,8`:")
    lines.append("")
    lines.append("`(y₄ − y₃) / 0,1 = 3`")
    lines.append("")
    lines.append("После приведения подобных членов получаем систему:")
    lines.append("")
    lines.append("`−10y₁ + 10y₂ = 1`")
    lines.append("")
    lines.append("`94y₁ − 199y₂ + 106y₃ = 1`")
    lines.append("")
    lines.append("`93y₂ − 199y₃ + 107y₄ = 1`")
    lines.append("")
    lines.append("`−10y₃ + 10y₄ = 3`")
    lines.append("")
    lines.append("**4. Решение системы методом прогонки.**")
    lines.append("")
    lines.append("Коэффициенты системы `aₖyₖ₋₁ + bₖyₖ + cₖyₖ₊₁ = dₖ` и прогоночные коэффициенты:")
    lines.append("")
    lines.append(bvp_coefficients_table_markdown(bvp_solution))
    lines.append("")
    lines.append("Обратный ход прогонки:")
    lines.append("")
    lines.append(f"`y₄ = V₄ = {_fmt_ru(bvp_solution.nodes[3].y, 9)}`")
    lines.append("")
    lines.append(
        f"`y₃ = U₃·y₄ + V₃ = {_fmt_ru(bvp_solution.sweep[2].u, 9)}·"
        f"({_fmt_ru(bvp_solution.nodes[3].y, 9)}){_fmt_ru_sign(bvp_solution.sweep[2].v, 9)} "
        f"= {_fmt_ru(bvp_solution.nodes[2].y, 9)}`"
    )
    lines.append("")
    lines.append(
        f"`y₂ = U₂·y₃ + V₂ = {_fmt_ru(bvp_solution.sweep[1].u, 9)}·"
        f"({_fmt_ru(bvp_solution.nodes[2].y, 9)}){_fmt_ru_sign(bvp_solution.sweep[1].v, 9)} "
        f"= {_fmt_ru(bvp_solution.nodes[1].y, 9)}`"
    )
    lines.append("")
    lines.append(
        f"`y₁ = U₁·y₂ + V₁ = {_fmt_ru(bvp_solution.sweep[0].u, 9)}·"
        f"({_fmt_ru(bvp_solution.nodes[1].y, 9)}){_fmt_ru_sign(bvp_solution.sweep[0].v, 9)} "
        f"= {_fmt_ru(bvp_solution.nodes[0].y, 9)}`"
    )
    lines.append("")
    lines.append("Сеточную функцию `yₖ = y(xₖ)` записываем в виде таблицы:")
    lines.append("")
    lines.append(bvp_solution_table_markdown(bvp_solution))
    lines.append("")
    lines.append("Проверка граничных производных:")
    lines.append("")
    lines.append(f"`(y₂ − y₁)/h = {_fmt_ru(bvp_solution.left_derivative, 9)}`")
    lines.append("")
    lines.append(f"`(y₄ − y₃)/h = {_fmt_ru(bvp_solution.right_derivative, 9)}`")
    lines.append("")
    lines.append(f"Максимальная невязка строк системы: `{_fmt_ru(bvp_solution.max_residual, 12)}`.")
    lines.append("")
    lines.append("График сеточного решения:")
    lines.append("")
    lines.append("![bvp solution](bvp_solution.svg)")
    lines.append("")
    return "\n".join(lines)


def _tick_step(v_min: float, v_max: float, target_ticks: int) -> float:
    span = v_max - v_min
    if span <= 0.0:
        return 1.0
    raw_step = span / max(target_ticks, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = factor * magnitude
        if step >= raw_step:
            return step
    return 10.0 * magnitude


def _build_ticks(v_min: float, v_max: float, target_ticks: int = 7) -> list[float]:
    step = _tick_step(v_min, v_max, target_ticks)
    start = math.floor(v_min / step) * step
    end = math.ceil(v_max / step) * step
    ticks: list[float] = []
    value = start
    while value <= end + 1e-12:
        ticks.append(round(value, 10))
        value += step
    return ticks


def sample_points(func: Callable[[float], float], left: float, right: float, count: int = 450) -> list[Point]:
    return [
        Point(
            x=left + (right - left) * i / (count - 1),
            y=func(left + (right - left) * i / (count - 1)),
        )
        for i in range(count)
    ]


def write_xy_svg(
    path: Path,
    title: str,
    series: Sequence[GraphSeries],
    node_points: Sequence[Point] = (),
    y_zero_baseline: bool = False,
) -> None:
    width = 980
    height = 620
    margin_left = 88
    margin_right = 38
    margin_top = 74
    margin_bottom = 78
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_points = [point for item in series for point in item.points] + list(node_points)
    x_min = min(point.x for point in all_points)
    x_max = max(point.x for point in all_points)
    y_min = min(point.y for point in all_points)
    y_max = max(point.y for point in all_points)
    if y_zero_baseline:
        y_min = min(0.0, y_min)
        y_max = max(0.0, y_max)
    x_padding = (x_max - x_min) * 0.04 if x_max > x_min else 1.0
    y_padding = (y_max - y_min) * 0.12 if y_max > y_min else 1.0
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding

    def sx(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_width

    def sy(y: float) -> float:
        return margin_top + (y_max - y) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="24" font-family="Arial" fill="#1f2933">{escape(title)}</text>',
    ]

    for tick in _build_ticks(x_min, x_max, 7):
        x = sx(tick)
        if margin_left <= x <= width - margin_right:
            parts.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="#e6edf3"/>')
            parts.append(f'<text x="{x:.2f}" y="{height - margin_bottom + 22}" text-anchor="middle" font-size="12" font-family="Arial" fill="#415166">{_fmt_ru_compact(tick, 2)}</text>')
    for tick in _build_ticks(y_min, y_max, 8):
        y = sy(tick)
        if margin_top <= y <= height - margin_bottom:
            parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e6edf3"/>')
            parts.append(f'<text x="{margin_left - 14}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#415166">{_fmt_ru_compact(tick, 2)}</text>')

    parts.append(f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#5c6b7a" stroke-width="1.4"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#5c6b7a" stroke-width="1.4"/>')
    parts.append(f'<text x="{width - margin_right + 10}" y="{height - margin_bottom + 5}" font-size="16" font-family="Arial" fill="#1f2933">x</text>')
    parts.append(f'<text x="{margin_left + 8}" y="{margin_top - 16}" font-size="16" font-family="Arial" fill="#1f2933">y</text>')

    legend_x = margin_left + 12
    legend_y = margin_top + 18
    for index, item in enumerate(series):
        x = legend_x + index * 168
        dash = f' stroke-dasharray="{item.dasharray}"' if item.dasharray else ""
        parts.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 30}" y2="{legend_y}" stroke="{item.color}" stroke-width="{item.width}" stroke-linecap="round"{dash}/>')
        parts.append(f'<text x="{x + 40}" y="{legend_y + 4}" font-size="12" font-family="Arial" fill="#1f2933">{escape(item.name)}</text>')

    for item in series:
        points = " ".join(f"{sx(point.x):.2f},{sy(point.y):.2f}" for point in item.points)
        dash = f' stroke-dasharray="{item.dasharray}"' if item.dasharray else ""
        parts.append(f'<polyline points="{points}" fill="none" stroke="{item.color}" stroke-width="{item.width}" stroke-linecap="round" stroke-linejoin="round"{dash}/>')

    for index, point in enumerate(node_points):
        x = sx(point.x)
        y = sy(point.y)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="#111827"/>')
        parts.append(f'<text x="{x + 6:.2f}" y="{y - 7:.2f}" font-size="11" font-family="Arial" fill="#111827">x{index + 1}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_cauchy_csv(path: Path, solutions: Sequence[CauchySolution]) -> None:
    headers = ["i", "x", "exact"]
    for solution in solutions:
        headers.extend([solution.key, f"{solution.key}_error"])
    rows = [",".join(headers)]
    for index in range(len(solutions[0].nodes)):
        values: list[float | int] = [
            index,
            solutions[0].nodes[index].x,
            solutions[0].nodes[index].exact,
        ]
        for solution in solutions:
            values.extend([solution.nodes[index].y, solution.nodes[index].absolute_error])
        rows.append(",".join(str(value) for value in values))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_bvp_csv(path: Path, solution: BvpSolution) -> None:
    headers = ["k", "x", "a", "b", "c", "d", "U", "V", "y", "residual"]
    rows = [",".join(headers)]
    for row, sweep, node in zip(solution.coefficients, solution.sweep, solution.nodes):
        values = [
            row.k,
            row.x,
            row.a,
            row.b,
            row.c,
            row.d,
            sweep.u,
            sweep.v,
            node.y,
            node.residual,
        ]
        rows.append(",".join(str(value) for value in values))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_results_json(path: Path, cauchy_solutions: Sequence[CauchySolution], bvp_solution: BvpSolution) -> None:
    payload = {
        "variant": VARIANT,
        "cauchy_problem": {
            "equation": "y' = y - 5x",
            "initial_condition": {"x": CAUCHY_A, "y": CAUCHY_Y0},
            "interval": [CAUCHY_A, CAUCHY_B],
            "h": CAUCHY_H,
            "exact_solution": "exp(x + 1) + 5x + 5",
            "solutions": [asdict(solution) for solution in cauchy_solutions],
        },
        "boundary_value_problem": {
            "equation": "y'' + 2xy' + y = 1",
            "boundary_conditions": {"left": "y'(0.5)=1", "right": "y'(0.8)=3"},
            "interval": [BVP_A, BVP_B],
            "h": BVP_H,
            "solution": asdict(bvp_solution),
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_graphs(results_dir: Path, cauchy_solutions: Sequence[CauchySolution], bvp_solution: BvpSolution) -> None:
    cauchy_series = [
        GraphSeries(
            name="точное",
            color="#111827",
            points=sample_points(exact_cauchy, CAUCHY_A, CAUCHY_B),
            width=2.7,
            dasharray="7 5",
        )
    ]
    colors = {
        "euler": "#d9480f",
        "modified_euler": "#0b7285",
        "runge_kutta": "#6741d9",
    }
    for solution in cauchy_solutions:
        cauchy_series.append(
            GraphSeries(
                name=solution.short_name,
                color=colors[solution.key],
                points=[Point(node.x, node.y) for node in solution.nodes],
                width=2.4,
            )
        )
    write_xy_svg(results_dir / "cauchy_solutions.svg", "Задача Коши: сравнение методов", cauchy_series)

    error_series = [
        GraphSeries(
            name=solution.short_name,
            color=colors[solution.key],
            points=[Point(node.x, node.absolute_error) for node in solution.nodes],
            width=2.5,
        )
        for solution in cauchy_solutions
    ]
    write_xy_svg(results_dir / "cauchy_errors.svg", "Абсолютная ошибка методов", error_series, y_zero_baseline=True)

    bvp_points = [Point(node.x, node.y) for node in bvp_solution.nodes]
    bvp_series = [
        GraphSeries(
            name="разностное решение",
            color="#0b7285",
            points=bvp_points,
            width=2.8,
        )
    ]
    write_xy_svg(results_dir / "bvp_solution.svg", "Краевая задача: сеточное решение", bvp_series, bvp_points)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cauchy_solutions = calculate_cauchy_solutions()
    bvp_solution = calculate_bvp_solution()

    (results_dir / "report.md").write_text(build_report(cauchy_solutions, bvp_solution), encoding="utf-8")
    write_cauchy_csv(results_dir / "cauchy_solutions.csv", cauchy_solutions)
    write_bvp_csv(results_dir / "bvp_solution.csv", bvp_solution)
    write_results_json(results_dir / "results.json", cauchy_solutions, bvp_solution)
    write_graphs(results_dir, cauchy_solutions, bvp_solution)

    print(f"Lab 8 variant {VARIANT} solved.")
    print(f"Results directory: {results_dir}")
    for solution in cauchy_solutions:
        print(
            f"{solution.name}: "
            f"y({CAUCHY_B})={solution.nodes[-1].y:.12f}, "
            f"max_error={solution.max_absolute_error:.12f}"
        )
    print(
        "Boundary value problem:",
        f"y({BVP_A})={bvp_solution.nodes[0].y:.12f}",
        f"y({BVP_B})={bvp_solution.nodes[-1].y:.12f}",
        f"max_residual={bvp_solution.max_residual:.3e}",
    )


if __name__ == "__main__":
    main()
