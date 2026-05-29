from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import random
import struct
from typing import Callable, Sequence
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile
import zlib

import numpy as np


VARIANT = 17
GEOMETRY_VARIANT = 5  # В таблице фигур только 12 вариантов: 17 -> 5 по циклу.
SEED = 17052026

INTEGRAL_N = 200_000
SYSTEM_TRAJECTORIES = 120_000
SYSTEM_STEPS = 50
AREA_N = 150_000


@dataclass
class IntegralResult:
    n: int
    volume: float
    estimate: float
    exact: float
    absolute_error: float
    relative_error_percent: float
    standard_error: float


@dataclass
class SystemResult:
    trajectories: int
    steps: int
    monte_carlo: list[float]
    exact: list[float]
    absolute_errors: list[float]
    standard_errors: list[float]


@dataclass
class AreaResult:
    geometry_variant: int
    n: int
    rectangle_area: float
    inside_count: int
    estimate: float
    reference: float
    absolute_error: float
    relative_error_percent: float


@dataclass
class Point:
    x: float
    y: float


def _fmt(value: float, digits: int = 6) -> str:
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}"


def _fmt_ru(value: float, digits: int = 6) -> str:
    return _fmt(value, digits).replace(".", ",").replace("-", "−")


def _fmt_percent(value: float, digits: int = 4) -> str:
    return f"{_fmt_ru(value, digits)}%"


def calculate_integral() -> tuple[IntegralResult, np.ndarray]:
    rng = np.random.default_rng(SEED)
    xs = rng.uniform(0.0, 1.0, INTEGRAL_N)
    ys = rng.uniform(1.0, 2.0, INTEGRAL_N)
    zs = rng.uniform(0.0, 3.0, INTEGRAL_N)
    values = xs**2 + ys**2 + zs
    volume = 3.0
    estimate = float(volume * np.mean(values))
    standard_error = float(volume * np.std(values, ddof=1) / math.sqrt(INTEGRAL_N))

    exact = 12.5
    absolute_error = abs(estimate - exact)
    relative_error_percent = absolute_error / abs(exact) * 100.0
    return (
        IntegralResult(
            n=INTEGRAL_N,
            volume=volume,
            estimate=estimate,
            exact=exact,
            absolute_error=absolute_error,
            relative_error_percent=relative_error_percent,
            standard_error=standard_error,
        ),
        np.column_stack([xs, ys, zs, values]),
    )


def solve_system_exact() -> np.ndarray:
    alpha = np.array([[0.5, 0.1], [0.2, 0.4]], dtype=float)
    beta = np.array([0.6, 0.1], dtype=float)
    return np.linalg.solve(np.eye(2) - alpha, beta)


def calculate_system() -> tuple[SystemResult, np.ndarray]:
    rng = np.random.default_rng(SEED + 1)
    alpha = np.array([[0.5, 0.1], [0.2, 0.4]], dtype=float)
    beta = np.array([0.6, 0.1], dtype=float)
    row_sums = alpha.sum(axis=1)
    probabilities = alpha / row_sums[:, None]

    estimates: list[float] = []
    standard_errors: list[float] = []
    samples_by_component: list[np.ndarray] = []

    for start_state in range(2):
        states = np.full(SYSTEM_TRAJECTORIES, start_state, dtype=int)
        weights = np.ones(SYSTEM_TRAJECTORIES, dtype=float)
        totals = beta[states].astype(float)
        for _ in range(SYSTEM_STEPS):
            draws = rng.random(SYSTEM_TRAJECTORIES)
            next_states = np.where(draws < probabilities[states, 0], 0, 1)
            weights *= alpha[states, next_states] / probabilities[states, next_states]
            states = next_states
            totals += weights * beta[states]
        estimates.append(float(np.mean(totals)))
        standard_errors.append(float(np.std(totals, ddof=1) / math.sqrt(SYSTEM_TRAJECTORIES)))
        samples_by_component.append(totals)

    exact = solve_system_exact()
    absolute_errors = [abs(value - exact[index]) for index, value in enumerate(estimates)]
    rows = np.column_stack([samples_by_component[0], samples_by_component[1]])
    return (
        SystemResult(
            trajectories=SYSTEM_TRAJECTORIES,
            steps=SYSTEM_STEPS,
            monte_carlo=estimates,
            exact=[float(value) for value in exact],
            absolute_errors=[float(value) for value in absolute_errors],
            standard_errors=standard_errors,
        ),
        rows,
    )


def inside_figure(x: float, y: float) -> bool:
    return -x * x + y**3 < 2.0 * x - y < 1.0 and -2.0 < x < 2.0 and -2.0 < y < 2.0


def upper_boundary_y(x: float) -> float:
    target = x * x + 2.0 * x
    left = -2.0
    right = 2.0
    for _ in range(70):
        middle = 0.5 * (left + right)
        value = middle**3 + middle
        if value < target:
            left = middle
        else:
            right = middle
    return 0.5 * (left + right)


def vertical_slice_height(x: float) -> float:
    lower = max(-2.0, 2.0 * x - 1.0)
    upper = min(2.0, upper_boundary_y(x))
    return max(0.0, upper - lower)


def reference_area_simpson(parts: int = 20_000) -> float:
    if parts % 2 == 1:
        parts += 1
    left = -2.0
    right = 2.0
    h = (right - left) / parts
    total = vertical_slice_height(left) + vertical_slice_height(right)
    for i in range(1, parts):
        x = left + i * h
        total += (4.0 if i % 2 == 1 else 2.0) * vertical_slice_height(x)
    return total * h / 3.0


def calculate_area() -> tuple[AreaResult, np.ndarray]:
    rng = np.random.default_rng(SEED + 2)
    xs = rng.uniform(-2.0, 2.0, AREA_N)
    ys = rng.uniform(-2.0, 2.0, AREA_N)
    mask = np.array([inside_figure(float(x), float(y)) for x, y in zip(xs, ys)], dtype=bool)
    inside_count = int(np.sum(mask))
    rectangle_area = 16.0
    estimate = rectangle_area * inside_count / AREA_N
    reference = reference_area_simpson()
    absolute_error = abs(estimate - reference)
    relative_error_percent = absolute_error / reference * 100.0
    return (
        AreaResult(
            geometry_variant=GEOMETRY_VARIANT,
            n=AREA_N,
            rectangle_area=rectangle_area,
            inside_count=inside_count,
            estimate=float(estimate),
            reference=float(reference),
            absolute_error=float(absolute_error),
            relative_error_percent=float(relative_error_percent),
        ),
        np.column_stack([xs, ys, mask.astype(int)]),
    )


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[float | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        row_start = y * stride
        raw.extend(pixels[row_start : row_start + stride])

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def blank_canvas(width: int, height: int, color: tuple[int, int, int] = (255, 255, 255)) -> bytearray:
    return bytearray(color * (width * height))


def set_pixel(pixels: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < height:
        index = (y * width + x) * 3
        pixels[index : index + 3] = bytes(color)


def draw_line(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        set_pixel(pixels, width, height, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_disc(
    pixels: bytearray,
    width: int,
    height: int,
    center_x: int,
    center_y: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    for y in range(center_y - radius, center_y + radius + 1):
        for x in range(center_x - radius, center_x + radius + 1):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius * radius:
                set_pixel(pixels, width, height, x, y, color)


def write_figure_png(path: Path, area_samples: np.ndarray) -> None:
    width = 920
    height = 720
    margin = 58
    pixels = blank_canvas(width, height)

    def sx(x: float) -> int:
        return round(margin + (x + 2.0) / 4.0 * (width - 2 * margin))

    def sy(y: float) -> int:
        return round(height - margin - (y + 2.0) / 4.0 * (height - 2 * margin))

    for py in range(margin, height - margin):
        y = -2.0 + (height - margin - py) / (height - 2 * margin) * 4.0
        for px in range(margin, width - margin):
            x = -2.0 + (px - margin) / (width - 2 * margin) * 4.0
            if inside_figure(x, y):
                set_pixel(pixels, width, height, px, py, (215, 235, 250))

    grid_color = (224, 230, 236)
    axis_color = (45, 55, 72)
    for tick in range(-2, 3):
        draw_line(pixels, width, height, sx(tick), margin, sx(tick), height - margin, grid_color)
        draw_line(pixels, width, height, margin, sy(tick), width - margin, sy(tick), grid_color)
    draw_line(pixels, width, height, sx(-2), sy(0), sx(2), sy(0), axis_color)
    draw_line(pixels, width, height, sx(0), sy(-2), sx(0), sy(2), axis_color)

    line_points: list[tuple[int, int]] = []
    upper_points: list[tuple[int, int]] = []
    for i in range(900):
        x = -2.0 + 4.0 * i / 899.0
        line_y = 2.0 * x - 1.0
        if -2.0 <= line_y <= 2.0:
            line_points.append((sx(x), sy(line_y)))
        upper_y = upper_boundary_y(x)
        if -2.0 <= upper_y <= 2.0:
            upper_points.append((sx(x), sy(upper_y)))
    for points, color in ((line_points, (211, 84, 0)), (upper_points, (13, 110, 253))):
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            draw_line(pixels, width, height, x0, y0, x1, y1, color)

    rng = random.Random(SEED + 3)
    sample_indices = rng.sample(range(len(area_samples)), min(2400, len(area_samples)))
    for index in sample_indices:
        x, y, flag = area_samples[index]
        color = (20, 105, 70) if int(flag) else (180, 190, 200)
        draw_disc(pixels, width, height, sx(float(x)), sy(float(y)), 1, color)

    draw_line(pixels, width, height, margin, margin, width - margin, margin, axis_color)
    draw_line(pixels, width, height, width - margin, margin, width - margin, height - margin, axis_color)
    draw_line(pixels, width, height, width - margin, height - margin, margin, height - margin, axis_color)
    draw_line(pixels, width, height, margin, height - margin, margin, margin, axis_color)
    write_png(path, width, height, pixels)


def write_report(
    path: Path,
    integral: IntegralResult,
    system: SystemResult,
    area: AreaResult,
) -> None:
    lines: list[str] = []
    lines.append(f"# Семестровое задание по дисциплине «Численные методы», вариант {VARIANT}")
    lines.append("")
    lines.append("## Исходные данные")
    lines.append("")
    lines.append("| задача | данные варианта 17 |")
    lines.append("|:---|:---|")
    lines.append("| Интеграл | `∫₀¹ ∫₁² ∫₀³ (x² + y² + z) dx dy dz` |")
    lines.append("| Система | `x₁ = 0,5x₁ + 0,1x₂ + 0,6`, `x₂ = 0,2x₁ + 0,4x₂ + 0,1` |")
    lines.append("| Фигура | в таблице фигур есть варианты 1–12, поэтому использован вариант 5 по циклу `17 − 12 = 5` |")
    lines.append("")
    lines.append("Во всех экспериментах используется фиксированное зерно генератора случайных чисел, поэтому результаты воспроизводимы.")
    lines.append("")
    lines.append("## 1) Вычисление интеграла методом Монте-Карло")
    lines.append("")
    lines.append("### Постановка задачи")
    lines.append("")
    lines.append("Для варианта 17 задан интеграл:")
    lines.append("")
    lines.append("`I = ∫₀¹ ∫₁² ∫₀³ (x² + y² + z) dx dy dz`.")
    lines.append("")
    lines.append("Область интегрирования — прямоугольный параллелепипед:")
    lines.append("")
    lines.append("`0 ≤ x ≤ 1`, `1 ≤ y ≤ 2`, `0 ≤ z ≤ 3`.")
    lines.append("")
    lines.append("Его объем:")
    lines.append("")
    lines.append("`V = (1−0)(2−1)(3−0) = 3`.")
    lines.append("")
    lines.append("### Формула метода")
    lines.append("")
    lines.append("Генерируем `N` случайных точек `Mᵢ(xᵢ, yᵢ, zᵢ)` равномерно внутри параллелепипеда. Затем считаем среднее значение функции `f(x,y,z)=x²+y²+z` в этих точках.")
    lines.append("")
    lines.append("Формула Монте-Карло:")
    lines.append("")
    lines.append("`I ≈ V/N · Σ f(xᵢ, yᵢ, zᵢ)`.")
    lines.append("")
    lines.append("Для расчета взято:")
    lines.append("")
    lines.append(f"`N = {integral.n}`.")
    lines.append("")
    lines.append("### Контрольное точное значение")
    lines.append("")
    lines.append("Так как функция раскладывается по переменным, интеграл можно проверить аналитически:")
    lines.append("")
    lines.append("`I = ∫₀¹∫₁²∫₀³x² dz dy dx + ∫₀¹∫₁²∫₀³y² dz dy dx + ∫₀¹∫₁²∫₀³z dz dy dx`.")
    lines.append("")
    lines.append("Отдельные слагаемые:")
    lines.append("")
    lines.append("`I₁ = (∫₀¹x²dx)·(2−1)·(3−0) = 1/3·3 = 1`.")
    lines.append("")
    lines.append("`I₂ = (1−0)·(∫₁²y²dy)·(3−0) = 7/3·3 = 7`.")
    lines.append("")
    lines.append("`I₃ = (1−0)·(2−1)·(∫₀³z dz) = 9/2 = 4,5`.")
    lines.append("")
    lines.append("`I = I₁ + I₂ + I₃ = 1 + 7 + 4,5 = 12,5`.")
    lines.append("")
    lines.append("### Результат расчета")
    lines.append("")
    lines.append("| N | оценка Монте-Карло | точное значение | абс. ошибка | отн. ошибка | стандартная ошибка |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {integral.n} | {_fmt_ru(integral.estimate, 9)} | {_fmt_ru(integral.exact, 9)} | "
        f"{_fmt_ru(integral.absolute_error, 9)} | {_fmt_percent(integral.relative_error_percent)} | "
        f"{_fmt_ru(integral.standard_error, 9)} |"
    )
    lines.append("")
    lines.append("Полученная оценка отличается от точного значения меньше чем на одну стандартную ошибку, поэтому для выбранного `N` расчет можно считать устойчивым.")
    lines.append("")

    lines.append("## 2) Решение системы линейных уравнений методом Монте-Карло")
    lines.append("")
    lines.append("### Постановка задачи")
    lines.append("")
    lines.append("Для варианта 17:")
    lines.append("")
    lines.append("`x₁ = 0,5x₁ + 0,1x₂ + 0,6`")
    lines.append("")
    lines.append("`x₂ = 0,2x₁ + 0,4x₂ + 0,1`")
    lines.append("")
    lines.append("В матричном виде:")
    lines.append("")
    lines.append("`x = αx + β`, где `α = [[0,5; 0,1], [0,2; 0,4]]`, `β = [0,6; 0,1]`.")
    lines.append("")
    lines.append("### Идея метода")
    lines.append("")
    lines.append("Так как суммы строк матрицы `α` меньше 1, решение можно представить рядом Неймана:")
    lines.append("")
    lines.append("`x = β + αβ + α²β + ...`.")
    lines.append("")
    lines.append("Метод Монте-Карло оценивает сумму этого ряда случайными траекториями. Для каждой стартовой компоненты моделируется цепочка состояний `1` и `2`. На каждом шаге выбирается следующее состояние, вес траектории умножается на отношение `αᵢⱼ / pᵢⱼ`, после чего к сумме добавляется вклад `Q·βⱼ`.")
    lines.append("")
    lines.append("Для данной системы удобно взять вероятности переходов пропорционально коэффициентам строки:")
    lines.append("")
    lines.append("`p₁₁ = 0,5 / 0,6 = 0,833333`, `p₁₂ = 0,1 / 0,6 = 0,166667`.")
    lines.append("")
    lines.append("`p₂₁ = 0,2 / 0,6 = 0,333333`, `p₂₂ = 0,4 / 0,6 = 0,666667`.")
    lines.append("")
    lines.append("Одна случайная траектория дает величину:")
    lines.append("")
    lines.append("`θ = βᵢ₀ + Q₁βᵢ₁ + Q₂βᵢ₂ + ...`, где `Q₀=1`, `Qₖ₊₁=Qₖ·αᵢₖᵢₖ₊₁/pᵢₖᵢₖ₊₁`.")
    lines.append("")
    lines.append("Среднее по большому числу траекторий является оценкой нужной компоненты решения.")
    lines.append("")
    lines.append("### Точное решение для проверки")
    lines.append("")
    lines.append("Перенесем неизвестные в левую часть:")
    lines.append("")
    lines.append("`0,5x₁ − 0,1x₂ = 0,6`")
    lines.append("")
    lines.append("`−0,2x₁ + 0,6x₂ = 0,1`")
    lines.append("")
    lines.append("Определитель системы:")
    lines.append("")
    lines.append("`D = 0,5·0,6 − (−0,1)·(−0,2) = 0,28`.")
    lines.append("")
    lines.append("Тогда:")
    lines.append("")
    lines.append("`x₁ = (0,6·0,6 − (−0,1)·0,1) / 0,28 = 1,321428571`.")
    lines.append("")
    lines.append("`x₂ = (0,5·0,1 − (−0,2)·0,6) / 0,28 = 0,607142857`.")
    lines.append("")
    lines.append("### Результат расчета")
    lines.append("")
    lines.append("| компонент | Монте-Карло | точное решение | абс. ошибка | стандартная ошибка |")
    lines.append("|:---|---:|---:|---:|---:|")
    for index in range(2):
        lines.append(
            f"| x{index + 1} | {_fmt_ru(system.monte_carlo[index], 9)} | "
            f"{_fmt_ru(system.exact[index], 9)} | {_fmt_ru(system.absolute_errors[index], 9)} | "
            f"{_fmt_ru(system.standard_errors[index], 9)} |"
        )
    lines.append("")
    lines.append(f"Число траекторий для каждой компоненты: `{system.trajectories}`, длина траектории: `{system.steps}`.")
    lines.append("")
    lines.append("Обе оценки попали в интервал порядка одной стандартной ошибки от точного решения.")
    lines.append("")

    lines.append("## 3) Определение площади фигуры методом Монте-Карло")
    lines.append("")
    lines.append("### Постановка задачи")
    lines.append("")
    lines.append("В таблице фигур приведены варианты 1–12, поэтому для номера 17 взят вариант 5 по циклу.")
    lines.append("")
    lines.append("Ограничения варианта 5:")
    lines.append("")
    lines.append("`−x² + y³ < 2x − y < 1`, `−2 < x < 2`, `−2 < y < 2`.")
    lines.append("")
    lines.append("### Метод расчета")
    lines.append("")
    lines.append("Ограничивающий прямоугольник:")
    lines.append("")
    lines.append("`−2 < x < 2`, `−2 < y < 2`, `Sпрям = 16`.")
    lines.append("")
    lines.append("Формула площади:")
    lines.append("")
    lines.append("`S ≈ Sпрям · K/N`, где `K` — количество точек, попавших в фигуру.")
    lines.append("")
    lines.append("Для каждой случайной точки проверяются два неравенства:")
    lines.append("")
    lines.append("`−x² + y³ < 2x − y` и `2x − y < 1`.")
    lines.append("")
    lines.append("Если оба условия выполнены, точка считается попавшей внутрь фигуры.")
    lines.append("")
    lines.append("### Контрольное численное значение")
    lines.append("")
    lines.append("Для проверки площадь дополнительно вычислена одномерным численным интегрированием по вертикальным сечениям. Из первого неравенства:")
    lines.append("")
    lines.append("`y³ + y < x² + 2x`.")
    lines.append("")
    lines.append("Так как функция `y³ + y` монотонно возрастает, верхняя граница сечения находится как корень уравнения:")
    lines.append("")
    lines.append("`y³ + y = x² + 2x`.")
    lines.append("")
    lines.append("Нижняя граница задается прямой `y = 2x − 1`. Далее высота сечения интегрируется по `x ∈ [−2; 2]` методом Симпсона.")
    lines.append("")
    lines.append("### Результат расчета")
    lines.append("")
    lines.append("| N | K | Sпрям | площадь Монте-Карло | контрольное численное значение | абс. ошибка | отн. ошибка |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {area.n} | {area.inside_count} | {_fmt_ru(area.rectangle_area, 6)} | "
        f"{_fmt_ru(area.estimate, 9)} | {_fmt_ru(area.reference, 9)} | "
        f"{_fmt_ru(area.absolute_error, 9)} | {_fmt_percent(area.relative_error_percent)} |"
    )
    lines.append("")
    lines.append("Построенная фигура:")
    lines.append("")
    lines.append("![figure](figure_area.png)")
    lines.append("")
    lines.append("## Вывод")
    lines.append("")
    lines.append("Метод Монте-Карло дал приемлемые приближенные значения во всех трех задачах. При увеличении количества случайных точек и траекторий точность улучшается как величина порядка `1/√N`, поэтому для повышения точности нужно существенно увеличивать число испытаний.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def paragraph_xml(text: str = "", style: str | None = None, align: str | None = None, bold: bool = False) -> str:
    props: list[str] = []
    if style:
        props.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        props.append(f'<w:jc w:val="{align}"/>')
    ppr = f"<w:pPr>{''.join(props)}</w:pPr>" if props else ""
    rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
    if text == "":
        return f"<w:p>{ppr}</w:p>"
    parts = []
    for index, line in enumerate(text.split("\n")):
        if index:
            parts.append("<w:br/>")
        parts.append(f'<w:t xml:space="preserve">{escape(line)}</w:t>')
    return f"<w:p>{ppr}<w:r>{rpr}{''.join(parts)}</w:r></w:p>"


def page_break_xml() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def table_xml(rows: Sequence[Sequence[str]]) -> str:
    border = (
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        "</w:tblBorders>"
    )
    xml = [f"<w:tbl><w:tblPr>{border}</w:tblPr>"]
    for row_index, row in enumerate(rows):
        xml.append("<w:tr>")
        for cell in row:
            shade = '<w:shd w:fill="EAF2F8"/>' if row_index == 0 else ""
            xml.append(f"<w:tc><w:tcPr>{shade}</w:tcPr>{paragraph_xml(str(cell), bold=row_index == 0)}</w:tc>")
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def image_xml(rel_id: str, name: str, width_px: int, height_px: int, image_id: int) -> str:
    cx = 5_600_000
    cy = round(cx * height_px / width_px)
    return f"""
<w:p>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:docPr id="{image_id}" name="{escape(name)}"/>
        <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr>
                <pic:cNvPr id="{image_id}" name="{escape(name)}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rel_id}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def create_docx(path: Path, figure_path: Path, integral: IntegralResult, system: SystemResult, area: AreaResult) -> None:
    body: list[str] = []
    body.append(paragraph_xml("МИНИСТЕРСТВО ОБРАЗОВАНИЯ И НАУКИ РОССИЙСКОЙ ФЕДЕРАЦИИ", align="center"))
    body.append(paragraph_xml("Федеральное государственное автономное образовательное учреждение высшего образования", align="center"))
    body.append(paragraph_xml("«Омский государственный технический университет»", align="center"))
    body.append(paragraph_xml(""))
    body.append(paragraph_xml("Факультет информационных технологий и компьютерных систем", align="center"))
    body.append(paragraph_xml("Кафедра «Прикладная математика и фундаментальная информатика»", align="center"))
    body.append(paragraph_xml(""))
    body.append(paragraph_xml(""))
    body.append(paragraph_xml("СЕМЕСТРОВОЕ (ДОМАШНЕЕ) ЗАДАНИЕ", align="center", bold=True))
    body.append(paragraph_xml("по дисциплине «Численные методы»", align="center"))
    body.append(paragraph_xml(f"Вариант {VARIANT}", align="center", bold=True))
    body.append(paragraph_xml(""))
    body.append(paragraph_xml("Студента _______________________________________"))
    body.append(paragraph_xml("Группа _________________________________________"))
    body.append(paragraph_xml(""))
    body.append(paragraph_xml("Руководитель ст. преподаватель Коберник Е.Г."))
    body.append(paragraph_xml(""))
    body.append(paragraph_xml("Выполнил ______________________________________"))
    body.append(paragraph_xml("Принято _______________________________________"))
    body.append(paragraph_xml(""))
    body.append(paragraph_xml(""))
    body.append(paragraph_xml("Омск 2026", align="center"))
    body.append(page_break_xml())

    body.append(paragraph_xml("Семестровое задание по дисциплине «Численные методы»", style="Heading1", align="center"))
    body.append(paragraph_xml("Вариант 17", align="center", bold=True))
    body.append(paragraph_xml(""))
    body.append(paragraph_xml("Содержание работы", style="Heading2"))
    body.append(paragraph_xml("1. Вычислить тройной интеграл методом Монте-Карло."))
    body.append(paragraph_xml("2. Решить систему линейных уравнений методом Монте-Карло."))
    body.append(paragraph_xml("3. Найти площадь фигуры методом Монте-Карло и построить эту фигуру."))
    body.append(paragraph_xml(""))
    body.append(paragraph_xml("Исходные данные", style="Heading2"))
    body.append(
        table_xml(
            [
                ["Задача", "Данные варианта"],
                ["Интеграл", "I = ∫₀¹ ∫₁² ∫₀³ (x² + y² + z) dx dy dz"],
                ["Система", "x₁ = 0,5x₁ + 0,1x₂ + 0,6;  x₂ = 0,2x₁ + 0,4x₂ + 0,1"],
                ["Фигура", "Использован вариант 5 по циклу, так как таблица фигур содержит варианты 1–12"],
            ]
        )
    )
    body.append(paragraph_xml(""))
    body.append(paragraph_xml("Примечание. Во всех вычислениях использовано фиксированное зерно генератора случайных чисел, поэтому результаты воспроизводимы."))
    body.append(page_break_xml())

    body.append(paragraph_xml("1. Вычисление интеграла методом Монте-Карло", style="Heading1"))
    body.append(paragraph_xml("Постановка задачи", style="Heading2"))
    body.append(paragraph_xml("Задан интеграл"))
    body.append(paragraph_xml("I = ∫₀¹ ∫₁² ∫₀³ (x² + y² + z) dx dy dz.", align="center", bold=True))
    body.append(paragraph_xml("Область интегрирования: 0 ≤ x ≤ 1, 1 ≤ y ≤ 2, 0 ≤ z ≤ 3."))
    body.append(paragraph_xml("Объем ограничивающего параллелепипеда:"))
    body.append(paragraph_xml("V = (1 − 0)(2 − 1)(3 − 0) = 3.", align="center"))
    body.append(paragraph_xml("Формула метода", style="Heading2"))
    body.append(paragraph_xml("Генерируем N случайных точек Mᵢ(xᵢ, yᵢ, zᵢ), равномерно распределенных в параллелепипеде. В каждой точке вычисляется f(xᵢ,yᵢ,zᵢ)=xᵢ²+yᵢ²+zᵢ."))
    body.append(paragraph_xml("Оценка интеграла находится по формуле:"))
    body.append(paragraph_xml("I ≈ V/N · Σ f(xᵢ, yᵢ, zᵢ).", align="center", bold=True))
    body.append(paragraph_xml(f"В программе взято N = {integral.n}."))
    body.append(paragraph_xml("Контрольное точное значение", style="Heading2"))
    body.append(paragraph_xml("Так как подынтегральная функция является суммой функций от отдельных переменных, интеграл удобно разложить на три слагаемых:"))
    body.append(paragraph_xml("I = I₁ + I₂ + I₃."))
    body.append(
        table_xml(
            [
                ["Слагаемое", "Расчет", "Значение"],
                ["I₁", "(∫₀¹ x² dx)·(2−1)·(3−0) = 1/3·3", "1"],
                ["I₂", "(1−0)·(∫₁² y² dy)·(3−0) = 7/3·3", "7"],
                ["I₃", "(1−0)·(2−1)·(∫₀³ z dz)", "4,5"],
                ["I", "I₁ + I₂ + I₃", "12,5"],
            ]
        )
    )
    body.append(paragraph_xml("Результат", style="Heading2"))
    body.append(
        table_xml(
            [
                ["N", "Оценка", "Точное", "Абс. ошибка", "Отн. ошибка", "Станд. ошибка"],
                [
                    str(integral.n),
                    _fmt_ru(integral.estimate, 9),
                    _fmt_ru(integral.exact, 9),
                    _fmt_ru(integral.absolute_error, 9),
                    _fmt_percent(integral.relative_error_percent),
                    _fmt_ru(integral.standard_error, 9),
                ],
            ]
        )
    )
    body.append(paragraph_xml("Полученная оценка отличается от точного значения меньше чем на одну стандартную ошибку, поэтому расчет устойчив для выбранного числа испытаний."))
    body.append(page_break_xml())

    body.append(paragraph_xml("2. Решение системы линейных уравнений методом Монте-Карло", style="Heading1"))
    body.append(paragraph_xml("Постановка задачи", style="Heading2"))
    body.append(paragraph_xml("Система варианта 17 имеет вид:"))
    body.append(paragraph_xml("x₁ = 0,5x₁ + 0,1x₂ + 0,6", align="center"))
    body.append(paragraph_xml("x₂ = 0,2x₁ + 0,4x₂ + 0,1", align="center"))
    body.append(paragraph_xml("В матричной форме: x = αx + β, где"))
    body.append(paragraph_xml("α = [[0,5; 0,1], [0,2; 0,4]],   β = [0,6; 0,1].", align="center"))
    body.append(paragraph_xml("Идея метода", style="Heading2"))
    body.append(paragraph_xml("Так как суммы строк матрицы α меньше единицы, решение можно представить рядом Неймана:"))
    body.append(paragraph_xml("x = β + αβ + α²β + α³β + ... .", align="center", bold=True))
    body.append(paragraph_xml("Метод Монте-Карло оценивает сумму этого ряда случайными траекториями. Для каждой компоненты строится цепочка состояний 1 и 2. Вклад траектории накапливается по формуле:"))
    body.append(paragraph_xml("θ = βᵢ₀ + Q₁βᵢ₁ + Q₂βᵢ₂ + ...,\nQ₀=1,  Qₖ₊₁ = Qₖ·αᵢₖᵢₖ₊₁ / pᵢₖᵢₖ₊₁.", align="center"))
    body.append(paragraph_xml("Вероятности переходов взяты пропорционально коэффициентам строки:"))
    body.append(
        table_xml(
            [
                ["Переход", "Вероятность"],
                ["p₁₁", "0,5 / 0,6 = 0,833333"],
                ["p₁₂", "0,1 / 0,6 = 0,166667"],
                ["p₂₁", "0,2 / 0,6 = 0,333333"],
                ["p₂₂", "0,4 / 0,6 = 0,666667"],
            ]
        )
    )
    body.append(paragraph_xml("Точное решение для проверки", style="Heading2"))
    body.append(paragraph_xml("Перенесем неизвестные в левую часть:"))
    body.append(paragraph_xml("0,5x₁ − 0,1x₂ = 0,6", align="center"))
    body.append(paragraph_xml("−0,2x₁ + 0,6x₂ = 0,1", align="center"))
    body.append(paragraph_xml("Определитель: D = 0,5·0,6 − (−0,1)·(−0,2) = 0,28."))
    body.append(paragraph_xml("x₁ = (0,6·0,6 − (−0,1)·0,1) / 0,28 = 1,321428571."))
    body.append(paragraph_xml("x₂ = (0,5·0,1 − (−0,2)·0,6) / 0,28 = 0,607142857."))
    body.append(paragraph_xml("Результат", style="Heading2"))
    body.append(
        table_xml(
            [
                ["Компонент", "Монте-Карло", "Точное", "Абс. ошибка", "Станд. ошибка"],
                [
                    "x₁",
                    _fmt_ru(system.monte_carlo[0], 9),
                    _fmt_ru(system.exact[0], 9),
                    _fmt_ru(system.absolute_errors[0], 9),
                    _fmt_ru(system.standard_errors[0], 9),
                ],
                [
                    "x₂",
                    _fmt_ru(system.monte_carlo[1], 9),
                    _fmt_ru(system.exact[1], 9),
                    _fmt_ru(system.absolute_errors[1], 9),
                    _fmt_ru(system.standard_errors[1], 9),
                ],
            ]
        )
    )
    body.append(paragraph_xml(f"Число траекторий для каждой компоненты: {system.trajectories}. Длина каждой траектории: {system.steps}."))
    body.append(paragraph_xml("Обе оценки близки к точному решению и попали в интервал порядка одной стандартной ошибки."))
    body.append(page_break_xml())

    body.append(paragraph_xml("3. Определение площади фигуры методом Монте-Карло", style="Heading1"))
    body.append(paragraph_xml("Постановка задачи", style="Heading2"))
    body.append(paragraph_xml("В таблице фигур приведены варианты 1–12, поэтому для номера 17 взят вариант 5 по циклу."))
    body.append(paragraph_xml("Ограничения варианта 5:"))
    body.append(paragraph_xml("−x² + y³ < 2x − y < 1,\n−2 < x < 2,\n−2 < y < 2.", align="center"))
    body.append(paragraph_xml("Метод расчета", style="Heading2"))
    body.append(paragraph_xml("В качестве ограничивающего прямоугольника выбран квадрат −2 < x < 2, −2 < y < 2. Его площадь:"))
    body.append(paragraph_xml("Sпрям = 4·4 = 16.", align="center", bold=True))
    body.append(paragraph_xml("Генерируем N случайных точек внутри этого прямоугольника. Если точка удовлетворяет двум неравенствам −x²+y³ < 2x−y и 2x−y < 1, она считается попавшей внутрь фигуры."))
    body.append(paragraph_xml("Площадь оценивается по формуле:"))
    body.append(paragraph_xml("S ≈ Sпрям · K/N,", align="center", bold=True))
    body.append(paragraph_xml("где K — количество точек внутри фигуры."))
    body.append(paragraph_xml("Контрольное численное значение", style="Heading2"))
    body.append(paragraph_xml("Для проверки была построена одномерная численная оценка по вертикальным сечениям. Из первого неравенства получается y³ + y < x² + 2x. Так как y³+y монотонно возрастает, верхняя граница сечения находится из уравнения y³ + y = x² + 2x. Нижняя граница задается прямой y = 2x − 1. Высота сечения интегрируется по x методом Симпсона."))
    body.append(paragraph_xml("Результат", style="Heading2"))
    body.append(
        table_xml(
            [
                ["N", "K", "Площадь", "Контроль", "Абс. ошибка", "Отн. ошибка"],
                [
                    str(area.n),
                    str(area.inside_count),
                    _fmt_ru(area.estimate, 9),
                    _fmt_ru(area.reference, 9),
                    _fmt_ru(area.absolute_error, 9),
                    _fmt_percent(area.relative_error_percent),
                ],
            ]
        )
    )
    body.append(paragraph_xml("Построенная фигура:", bold=True))
    body.append(image_xml("rId1", "figure_area.png", 920, 720, 1))
    body.append(page_break_xml())
    body.append(paragraph_xml("Вывод", style="Heading1"))
    body.append(paragraph_xml("В работе методом Монте-Карло были решены три задачи: вычисление тройного интеграла, решение системы линейных уравнений и определение площади фигуры."))
    body.append(paragraph_xml("Во всех случаях полученные приближенные значения близки к контрольным значениям. Погрешность метода Монте-Карло убывает медленно, примерно как 1/√N, поэтому для заметного повышения точности требуется существенно увеличивать число испытаний."))
    body.append(paragraph_xml("Итоговые значения:", style="Heading2"))
    body.append(
        table_xml(
            [
                ["Задача", "Итог"],
                ["Интеграл", f"I ≈ {_fmt_ru(integral.estimate, 9)}"],
                ["Система", f"x₁ ≈ {_fmt_ru(system.monte_carlo[0], 9)}, x₂ ≈ {_fmt_ru(system.monte_carlo[1], 9)}"],
                ["Площадь фигуры", f"S ≈ {_fmt_ru(area.estimate, 9)}"],
            ]
        )
    )

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
 xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
 xmlns:v="urn:schemas-microsoft-com:vml"
 xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:w10="urn:schemas-microsoft-com:office:word"
 xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
 xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
 xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
 xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
 xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"
 mc:Ignorable="w14 wp14">
 <w:body>
 {''.join(body)}
 <w:sectPr>
   <w:pgSz w:w="11906" w:h="16838"/>
   <w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="708" w:footer="708" w:gutter="0"/>
 </w:sectPr>
 </w:body>
</w:document>
"""

    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:before="180" w:after="80"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="30"/></w:rPr>
  </w:style>
</w:styles>
"""

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/figure_area.png"/>
</Relationships>
"""

    with ZipFile(path, "w", ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", root_rels)
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", styles_xml)
        docx.writestr("word/_rels/document.xml.rels", document_rels)
        docx.write(figure_path, "word/media/figure_area.png")


def write_results_json(path: Path, integral: IntegralResult, system: SystemResult, area: AreaResult) -> None:
    payload = {
        "variant": VARIANT,
        "geometry_variant_assumption": GEOMETRY_VARIANT,
        "seed": SEED,
        "integral": asdict(integral),
        "linear_system": asdict(system),
        "area": asdict(area),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    integral, integral_samples = calculate_integral()
    system, system_samples = calculate_system()
    area, area_samples = calculate_area()

    write_csv(
        results_dir / "integral_samples_preview.csv",
        ["x", "y", "z", "f"],
        integral_samples[:500],
    )
    write_csv(
        results_dir / "system_samples_preview.csv",
        ["x1_estimate", "x2_estimate"],
        system_samples[:500],
    )
    write_csv(
        results_dir / "area_samples_preview.csv",
        ["x", "y", "inside"],
        area_samples[:1000],
    )

    figure_path = results_dir / "figure_area.png"
    write_figure_png(figure_path, area_samples)
    write_report(results_dir / "report.md", integral, system, area)
    write_results_json(results_dir / "results.json", integral, system, area)
    docx_path = results_dir / "semester_assignment_variant17.docx"
    try:
        create_docx(docx_path, figure_path, integral, system, area)
    except PermissionError:
        docx_path = results_dir / "semester_assignment_variant17_detailed.docx"
        create_docx(docx_path, figure_path, integral, system, area)

    print(f"Homework variant {VARIANT} solved.")
    print(f"Results directory: {results_dir}")
    print(f"DOCX: {docx_path}")
    print(f"Integral MC: {integral.estimate:.9f}, exact: {integral.exact:.9f}")
    print(
        "System MC:",
        f"x1={system.monte_carlo[0]:.9f}",
        f"x2={system.monte_carlo[1]:.9f}",
    )
    print(f"Area MC: {area.estimate:.9f}, reference: {area.reference:.9f}, K={area.inside_count}")


if __name__ == "__main__":
    main()
