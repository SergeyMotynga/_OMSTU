from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


VARIANT = 17

X_VALUES = np.array([0.219, 0.551, 0.883, 1.215, 1.547, 1.878, 2.210, 2.542, 2.874])
Y_VALUES = np.array([-2.151, -1.270, -0.201, 0.771, 1.626, 2.479, 3.249, 3.841, 4.617])


SUBSCRIPT = str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋")
SUPERSCRIPT = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Approximation:
    law_number: int
    name: str
    formula: str
    parameters: dict[str, float]
    normal_matrix: list[list[float]]
    normal_rhs: list[float]
    predicted: list[float]
    residual_squares: list[float]
    residual_sum: float


@dataclass
class GraphSeries:
    name: str
    color: str
    points: list[Point]
    width: float = 2.7
    dasharray: str | None = None


@dataclass
class CandidateCurve:
    law_number: int
    formula: str
    params: dict[str, float]
    residual_sum: float
    points: list[Point]


def subscript(value: int | str) -> str:
    return str(value).translate(SUBSCRIPT)


def superscript(value: int | str) -> str:
    return str(value).translate(SUPERSCRIPT)


def _fmt(value: float, digits: int = 6) -> str:
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}"


def _fmt_ru(value: float, digits: int = 6) -> str:
    return _fmt(value, digits).replace(".", ",").replace("-", "−")


def _fmt_ru_sign(value: float, digits: int = 6) -> str:
    sign = "+" if value >= 0 else "−"
    return f" {sign} {_fmt_ru(abs(value), digits)}"


def points_markdown(xs: np.ndarray, ys: np.ndarray) -> str:
    lines = ["| i | xᵢ | yᵢ |", "|---:|---:|---:|"]
    for i, (x, y) in enumerate(zip(xs, ys)):
        lines.append(f"| {i} | {_fmt_ru(float(x), 3)} | {_fmt_ru(float(y), 3)} |")
    return "\n".join(lines)


def normal_system_markdown(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> str:
    lines: list[str] = []
    for row, value in zip(matrix, rhs):
        left = "   ".join(_fmt_ru(float(v), 6).rjust(12) for v in row)
        lines.append(f"{left}  |  {_fmt_ru(float(value), 6)}")
    return "\n".join(lines)


def substituted_equations_markdown(
    matrix: Sequence[Sequence[float]],
    rhs: Sequence[float],
    variables: Sequence[str],
) -> str:
    lines: list[str] = []
    for row, value in zip(matrix, rhs):
        terms: list[str] = []
        for coefficient, variable in zip(row, variables):
            coefficient_text = _fmt_ru(abs(float(coefficient)), 6)
            term = f"{coefficient_text}·{variable}"
            if not terms:
                terms.append(term if coefficient >= 0 else f"−{term}")
            else:
                sign = "+" if coefficient >= 0 else "−"
                terms.append(f"{sign} {term}")
        lines.append(f"{' '.join(terms)} = {_fmt_ru(float(value), 6)}")
    return "\n".join(lines)


def inline_equations_markdown(equations: Sequence[str]) -> str:
    return "<br>\n".join(f"`{equation}`" for equation in equations)


def solve_normal(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return np.linalg.solve(matrix, rhs)


def fit_quadratic(xs: np.ndarray, ys: np.ndarray) -> Approximation:
    n = len(xs)
    sx = float(np.sum(xs))
    sx2 = float(np.sum(xs**2))
    sx3 = float(np.sum(xs**3))
    sx4 = float(np.sum(xs**4))
    sy = float(np.sum(ys))
    sxy = float(np.sum(xs * ys))
    sx2y = float(np.sum((xs**2) * ys))

    matrix = np.array(
        [
            [sx4, sx3, sx2],
            [sx3, sx2, sx],
            [sx2, sx, float(n)],
        ],
        dtype=float,
    )
    rhs = np.array([sx2y, sxy, sy], dtype=float)
    a, b, c = solve_normal(matrix, rhs)
    predicted = a * xs**2 + b * xs + c
    residual_squares = (predicted - ys) ** 2

    return Approximation(
        law_number=1,
        name="квадратичный закон",
        formula="y = ax² + bx + c",
        parameters={"a": float(a), "b": float(b), "c": float(c)},
        normal_matrix=matrix.tolist(),
        normal_rhs=rhs.tolist(),
        predicted=predicted.tolist(),
        residual_squares=residual_squares.tolist(),
        residual_sum=float(np.sum(residual_squares)),
    )


def rational_residual_for_a(alpha: float, xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    if np.any(np.abs(xs + alpha) < 1e-12):
        return float("inf"), 0.0, 0.0
    t = 1.0 / (xs + alpha)
    matrix = np.array(
        [
            [float(np.sum(t * t)), float(np.sum(t))],
            [float(np.sum(t)), float(len(xs))],
        ],
        dtype=float,
    )
    rhs = np.array([float(np.sum(ys * t)), float(np.sum(ys))], dtype=float)
    b, c = solve_normal(matrix, rhs)
    predicted = b * t + c
    return float(np.sum((predicted - ys) ** 2)), float(b), float(c)


def golden_minimize(
    func: Callable[[float], float], left: float, right: float, iterations: int = 120
) -> float:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = func(x1)
    f2 = func(x2)
    for _ in range(iterations):
        if right - left < 1e-12:
            break
        if f1 > f2:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + ratio * (right - left)
            f2 = func(x2)
        else:
            right = x2
            x2 = x1
            f2 = f1
            x1 = right - ratio * (right - left)
            f1 = func(x1)
    return 0.5 * (left + right)


def fit_rational_shift(xs: np.ndarray, ys: np.ndarray) -> Approximation:
    def residual(alpha: float) -> float:
        value, _, _ = rational_residual_for_a(alpha, xs, ys)
        return value

    grid = np.linspace(0.001, 50.0, 5000)
    values = np.array([residual(float(alpha)) for alpha in grid])
    best_index = int(np.argmin(values))
    step = grid[1] - grid[0]
    left = max(0.001, float(grid[best_index] - 5.0 * step))
    right = float(grid[best_index] + 5.0 * step)
    alpha = golden_minimize(residual, left, right)

    residual_sum, b, c = rational_residual_for_a(alpha, xs, ys)
    t = 1.0 / (xs + alpha)
    matrix = np.array(
        [
            [float(np.sum(t * t)), float(np.sum(t))],
            [float(np.sum(t)), float(len(xs))],
        ],
        dtype=float,
    )
    rhs = np.array([float(np.sum(ys * t)), float(np.sum(ys))], dtype=float)
    predicted = b * t + c
    residual_squares = (predicted - ys) ** 2

    return Approximation(
        law_number=5,
        name="дробно-линейный закон со сдвигом",
        formula="y = b/(x + a) + c",
        parameters={"a": float(alpha), "b": float(b), "c": float(c)},
        normal_matrix=matrix.tolist(),
        normal_rhs=rhs.tolist(),
        predicted=predicted.tolist(),
        residual_squares=residual_squares.tolist(),
        residual_sum=residual_sum,
    )


def fit_power_shift_candidate(xs: np.ndarray, ys: np.ndarray) -> tuple[int, str, str, dict[str, float], float]:
    # y = b*x^a + c. For fixed c, ln(y-c) = a*ln(x) + ln(b).
    upper = float(np.min(ys) - 1e-8)

    def solve_for_c(c: float) -> tuple[float, float, float]:
        if np.any(ys - c <= 0.0):
            return float("inf"), 0.0, 0.0
        design = np.column_stack([np.log(xs), np.ones_like(xs)])
        theta, *_ = np.linalg.lstsq(design, np.log(ys - c), rcond=None)
        a = float(theta[0])
        b = float(math.exp(theta[1]))
        predicted = b * xs**a + c
        return float(np.sum((predicted - ys) ** 2)), a, b

    def residual(c: float) -> float:
        return solve_for_c(c)[0]

    c = golden_minimize(residual, float(np.min(ys) - 20.0), upper)
    s, a, b = solve_for_c(c)
    return 3, "степенной закон со сдвигом", "y = b·xᵃ + c", {"a": a, "b": b, "c": c}, s


def fit_power_shift(xs: np.ndarray, ys: np.ndarray) -> Approximation:
    # y = b*x^a + c. For fixed c:
    # z_i = ln(y_i - c), t_i = ln(x_i), z = a*t + ln(b).
    upper = float(np.min(ys) - 1e-8)

    def solve_for_c(c: float) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
        if np.any(ys - c <= 0.0):
            return float("inf"), 0.0, 0.0, 0.0, np.zeros((2, 2)), np.zeros(2)
        t = np.log(xs)
        z = np.log(ys - c)
        matrix = np.array(
            [
                [float(np.sum(t * t)), float(np.sum(t))],
                [float(np.sum(t)), float(len(xs))],
            ],
            dtype=float,
        )
        rhs = np.array([float(np.sum(t * z)), float(np.sum(z))], dtype=float)
        a, ln_b = solve_normal(matrix, rhs)
        b = float(math.exp(ln_b))
        predicted = b * xs**a + c
        residual_sum = float(np.sum((predicted - ys) ** 2))
        return residual_sum, float(a), b, float(ln_b), matrix, rhs

    def residual(c: float) -> float:
        return solve_for_c(c)[0]

    c = golden_minimize(residual, float(np.min(ys) - 20.0), upper)
    residual_sum, a, b, ln_b, matrix, rhs = solve_for_c(c)
    predicted = b * xs**a + c
    residual_squares = (predicted - ys) ** 2

    return Approximation(
        law_number=3,
        name="степенной закон со сдвигом",
        formula="y = b·xᵃ + c",
        parameters={"a": a, "b": b, "c": float(c), "ln_b": ln_b},
        normal_matrix=matrix.tolist(),
        normal_rhs=rhs.tolist(),
        predicted=predicted.tolist(),
        residual_squares=residual_squares.tolist(),
        residual_sum=residual_sum,
    )


def fit_exp_shift_candidate(xs: np.ndarray, ys: np.ndarray) -> tuple[int, str, str, dict[str, float], float]:
    # y = b*e^(a*x) + c. For fixed c, ln(y-c) = a*x + ln(b).
    upper = float(np.min(ys) - 1e-8)

    def solve_for_c(c: float) -> tuple[float, float, float]:
        if np.any(ys - c <= 0.0):
            return float("inf"), 0.0, 0.0
        design = np.column_stack([xs, np.ones_like(xs)])
        theta, *_ = np.linalg.lstsq(design, np.log(ys - c), rcond=None)
        a = float(theta[0])
        b = float(math.exp(theta[1]))
        predicted = b * np.exp(a * xs) + c
        return float(np.sum((predicted - ys) ** 2)), a, b

    def residual(c: float) -> float:
        return solve_for_c(c)[0]

    c = golden_minimize(residual, float(np.min(ys) - 20.0), upper)
    s, a, b = solve_for_c(c)
    return 4, "экспоненциальный закон со сдвигом", "y = b·eᵃˣ + c", {"a": a, "b": b, "c": c}, s


def fit_gaussian_like_candidate(xs: np.ndarray, ys: np.ndarray) -> tuple[int, str, str, dict[str, float], float]:
    # y = b*exp(-a*(x+c)^2)+c from the lab sheet. It is nonlinear in a and c;
    # for each pair (a,c), b is found by least squares.
    best_s = float("inf")
    best_a = 0.0
    best_b = 0.0
    best_c = 0.0
    for a in np.linspace(0.0, 5.0, 181):
        for c in np.linspace(-10.0, 10.0, 181):
            g = np.exp(-float(a) * (xs + float(c)) ** 2)
            denom = float(np.dot(g, g))
            if denom < 1e-14:
                continue
            b = float(np.dot(g, ys - float(c)) / denom)
            predicted = b * g + float(c)
            s = float(np.sum((predicted - ys) ** 2))
            if s < best_s:
                best_s = s
                best_a = float(a)
                best_b = b
                best_c = float(c)
    return (
        9,
        "экспоненциальный пик",
        "y = b·exp(−a(x+c)²)+c",
        {"a": best_a, "b": best_b, "c": best_c},
        best_s,
    )


def fit_linear_candidate(
    law_number: int,
    name: str,
    formula: str,
    basis: Sequence[np.ndarray],
    names: Sequence[str],
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[int, str, str, dict[str, float], float]:
    matrix_data = np.column_stack(basis)
    coeffs, *_ = np.linalg.lstsq(matrix_data, ys, rcond=None)
    predicted = matrix_data @ coeffs
    params = {name: float(value) for name, value in zip(names, coeffs)}
    return law_number, name, formula, params, float(np.sum((predicted - ys) ** 2))


def candidate_summary(xs: np.ndarray, ys: np.ndarray) -> list[tuple[int, str, str, dict[str, float], float]]:
    candidates = [
        fit_linear_candidate(
            1,
            "квадратичный закон",
            "y = ax² + bx + c",
            [xs**2, xs, np.ones_like(xs)],
            ["a", "b", "c"],
            xs,
            ys,
        ),
        fit_linear_candidate(
            2,
            "обратные степени",
            "y = a/x² + b/x + c",
            [1.0 / xs**2, 1.0 / xs, np.ones_like(xs)],
            ["a", "b", "c"],
            xs,
            ys,
        ),
        fit_linear_candidate(
            6,
            "экспонента с убыванием",
            "y = ax + b·e⁻ˣ + c",
            [xs, np.exp(-xs), np.ones_like(xs)],
            ["a", "b", "c"],
            xs,
            ys,
        ),
        fit_linear_candidate(
            7,
            "обратная степень и экспонента",
            "y = a/x + b·eˣ + c",
            [1.0 / xs, np.exp(xs), np.ones_like(xs)],
            ["a", "b", "c"],
            xs,
            ys,
        ),
        fit_linear_candidate(
            8,
            "логарифмически-экспоненциальный закон",
            "y = ax·ln(x) + b·eˣ + c",
            [xs * np.log(xs), np.exp(xs), np.ones_like(xs)],
            ["a", "b", "c"],
            xs,
            ys,
        ),
        fit_linear_candidate(
            10,
            "корень и синус",
            "y = a√x + b·sin(x) + c",
            [np.sqrt(xs), np.sin(xs), np.ones_like(xs)],
            ["a", "b", "c"],
            xs,
            ys,
        ),
    ]
    rational = fit_rational_shift(xs, ys)
    candidates.append(
        (
            rational.law_number,
            rational.name,
            rational.formula,
            rational.parameters,
            rational.residual_sum,
        )
    )
    candidates.append(fit_power_shift_candidate(xs, ys))
    candidates.append(fit_exp_shift_candidate(xs, ys))
    candidates.append(fit_gaussian_like_candidate(xs, ys))
    return sorted(candidates, key=lambda item: item[4])


def approximation_table(approx: Approximation, xs: np.ndarray, ys: np.ndarray) -> str:
    lines = [
        "| i | xᵢ | yᵢ | ŷᵢ | (ŷᵢ − yᵢ)² |",
        "|---:|---:|---:|---:|---:|",
    ]
    for i, (x, y, y_hat, square) in enumerate(
        zip(xs, ys, approx.predicted, approx.residual_squares)
    ):
        lines.append(
            f"| {i} | {_fmt_ru(float(x), 3)} | {_fmt_ru(float(y), 3)} | "
            f"{_fmt_ru(float(y_hat), 6)} | {_fmt_ru(float(square), 9)} |"
        )
    lines.append(
        f"|  |  |  | **δ** | **{_fmt_ru(float(approx.residual_sum), 9)}** |"
    )
    return "\n".join(lines)


def summary_table(candidates: Sequence[tuple[int, str, str, dict[str, float], float]]) -> str:
    lines = [
        "| закон | вид | невязка δ |",
        "|---:|:---|---:|",
    ]
    for law_number, _, formula, _, residual in candidates:
        lines.append(f"| {law_number} | {formula} | {_fmt_ru(residual, 9)} |")
    return "\n".join(lines)


def candidate_value(law_number: int, params: dict[str, float], x: float) -> float:
    a = params.get("a", 0.0)
    b = params.get("b", 0.0)
    c = params.get("c", 0.0)
    if law_number == 1:
        return a * x * x + b * x + c
    if law_number == 2:
        return a / (x * x) + b / x + c
    if law_number == 3:
        return b * (x**a) + c
    if law_number == 4:
        return b * math.exp(a * x) + c
    if law_number == 5:
        return b / (x + a) + c
    if law_number == 6:
        return a * x + b * math.exp(-x) + c
    if law_number == 7:
        return a / x + b * math.exp(x) + c
    if law_number == 8:
        return a * x * math.log(x) + b * math.exp(x) + c
    if law_number == 9:
        return b * math.exp(-a * (x + c) ** 2) + c
    if law_number == 10:
        return a * math.sqrt(x) + b * math.sin(x) + c
    raise ValueError(f"Unknown law number: {law_number}")


def build_candidate_curves(
    candidates: Sequence[tuple[int, str, str, dict[str, float], float]],
    left: float,
    right: float,
    count: int = 220,
) -> list[CandidateCurve]:
    curves: list[CandidateCurve] = []
    for law_number, _, formula, params, residual_sum in sorted(candidates, key=lambda item: item[0]):
        points: list[Point] = []
        for i in range(count):
            x = left + (right - left) * i / (count - 1)
            try:
                y = candidate_value(law_number, params, x)
            except (ValueError, OverflowError):
                y = float("nan")
            if math.isfinite(y):
                points.append(Point(x=x, y=y))
        curves.append(
            CandidateCurve(
                law_number=law_number,
                formula=formula,
                params=params,
                residual_sum=residual_sum,
                points=points,
            )
        )
    return curves


def formula_quadratic(approx: Approximation) -> str:
    a = approx.parameters["a"]
    b = approx.parameters["b"]
    c = approx.parameters["c"]
    return (
        f"y = {_fmt_ru(a, 6)}·x²"
        f"{_fmt_ru_sign(b, 6)}·x"
        f"{_fmt_ru_sign(c, 6)}"
    )


def formula_rational(approx: Approximation) -> str:
    a = approx.parameters["a"]
    b = approx.parameters["b"]
    c = approx.parameters["c"]
    return (
        f"y = {_fmt_ru(b, 6)} / (x + {_fmt_ru(a, 6)})"
        f"{_fmt_ru_sign(c, 6)}"
    )


def formula_power(approx: Approximation) -> str:
    a = approx.parameters["a"]
    b = approx.parameters["b"]
    c = approx.parameters["c"]
    return (
        f"y = {_fmt_ru(b, 6)}·x<sup>{_fmt_ru(a, 6)}</sup>"
        f"{_fmt_ru_sign(c, 6)}"
    )


def sample_points(func: Callable[[float], float], left: float, right: float, count: int = 500) -> list[Point]:
    points: list[Point] = []
    for i in range(count):
        x = left + (right - left) * i / (count - 1)
        points.append(Point(x=x, y=func(x)))
    return points


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


def _build_ticks(v_min: float, v_max: float, target_ticks: int = 8) -> list[float]:
    step = _tick_step(v_min, v_max, target_ticks)
    start = math.floor(v_min / step) * step
    end = math.ceil(v_max / step) * step
    ticks: list[float] = []
    value = start
    while value <= end + 1e-12:
        ticks.append(round(value, 10))
        value += step
    return ticks


def write_svg(
    path: Path,
    title: str,
    data_points: Sequence[Point],
    series: Sequence[GraphSeries],
) -> None:
    width, height = 980, 640
    margin_left, margin_right = 86, 36
    margin_top, margin_bottom = 68, 76

    all_points = list(data_points)
    for item in series:
        all_points.extend(item.points)
    x_min = min(point.x for point in all_points)
    x_max = max(point.x for point in all_points)
    y_min = min(point.y for point in all_points)
    y_max = max(point.y for point in all_points)
    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.12 * (y_max - y_min if y_max > y_min else 1.0)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    plot_left, plot_right = margin_left, width - margin_right
    plot_top, plot_bottom = margin_top, height - margin_bottom

    def sx(x: float) -> float:
        return plot_left + (x - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def sy(y: float) -> float:
        return plot_bottom - (y - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    x_axis = sy(0.0) if y_min <= 0.0 <= y_max else plot_bottom
    y_axis = sx(0.0) if x_min <= 0.0 <= x_max else plot_left
    x_ticks = _build_ticks(x_min, x_max)
    y_ticks = _build_ticks(y_min, y_max)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'  <text x="{width / 2:.0f}" y="34" text-anchor="middle" font-size="24" font-family="Arial" fill="#1f2933">{title}</text>',
    ]

    for xt in x_ticks:
        px = sx(xt)
        if plot_left <= px <= plot_right:
            parts.append(
                f'  <line x1="{px:.2f}" y1="{plot_top:.2f}" x2="{px:.2f}" y2="{plot_bottom:.2f}" stroke="#eceff3" stroke-width="1"/>'
            )
            parts.append(
                f'  <text x="{px:.2f}" y="{plot_bottom + 22:.2f}" text-anchor="middle" font-size="12" font-family="Arial" fill="#4b5563">{_fmt_ru(xt, 1)}</text>'
            )
    for yt in y_ticks:
        py = sy(yt)
        if plot_top <= py <= plot_bottom:
            parts.append(
                f'  <line x1="{plot_left:.2f}" y1="{py:.2f}" x2="{plot_right:.2f}" y2="{py:.2f}" stroke="#eceff3" stroke-width="1"/>'
            )
            parts.append(
                f'  <text x="{plot_left - 12:.2f}" y="{py + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#4b5563">{_fmt_ru(yt, 1)}</text>'
            )
    parts.append(
        f'  <line x1="{plot_left:.2f}" y1="{x_axis:.2f}" x2="{plot_right:.2f}" y2="{x_axis:.2f}" stroke="#5c6670" stroke-width="1.4"/>'
    )
    parts.append(
        f'  <line x1="{y_axis:.2f}" y1="{plot_top:.2f}" x2="{y_axis:.2f}" y2="{plot_bottom:.2f}" stroke="#5c6670" stroke-width="1.4"/>'
    )

    for item in series:
        polyline = " ".join(f"{sx(point.x):.2f},{sy(point.y):.2f}" for point in item.points)
        dash = f' stroke-dasharray="{item.dasharray}"' if item.dasharray else ""
        parts.append(
            f'  <polyline points="{polyline}" fill="none" stroke="{item.color}" stroke-width="{item.width}"{dash}/>'
        )

    for i, point in enumerate(data_points):
        parts.append(
            f'  <circle cx="{sx(point.x):.2f}" cy="{sy(point.y):.2f}" r="4.5" fill="#111827"/>'
        )
        parts.append(
            f'  <text x="{sx(point.x) + 7:.2f}" y="{sy(point.y) - 7:.2f}" font-size="11" font-family="Arial" fill="#111827">P{i}</text>'
        )

    legend_x, legend_y = plot_left + 10, plot_top + 14
    for i, item in enumerate(series):
        y = legend_y + 22 * i
        dash = f' stroke-dasharray="{item.dasharray}"' if item.dasharray else ""
        parts.append(
            f'  <line x1="{legend_x:.2f}" y1="{y:.2f}" x2="{legend_x + 28:.2f}" y2="{y:.2f}" stroke="{item.color}" stroke-width="{item.width}"{dash}/>'
        )
        parts.append(
            f'  <text x="{legend_x + 36:.2f}" y="{y + 4:.2f}" font-size="12" font-family="Arial" fill="#1f2933">{item.name}</text>'
        )

    parts.append(
        f'  <text x="{plot_right + 10:.2f}" y="{x_axis + 5:.2f}" font-size="16" font-family="Arial" fill="#1f2933">x</text>'
    )
    parts.append(
        f'  <text x="{y_axis + 8:.2f}" y="{plot_top - 14:.2f}" font-size="16" font-family="Arial" fill="#1f2933">y</text>'
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_model_grid_svg(
    path: Path,
    title: str,
    data_points: Sequence[Point],
    curves: Sequence[CandidateCurve],
    selected_laws: set[int],
) -> None:
    panel_w, panel_h = 420, 250
    gap_x, gap_y = 34, 44
    margin_left, margin_top = 64, 82
    cols = 2
    rows = math.ceil(len(curves) / cols)
    width = margin_left * 2 + cols * panel_w + (cols - 1) * gap_x
    height = margin_top + rows * panel_h + (rows - 1) * gap_y + 56

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'  <text x="{width / 2:.0f}" y="38" text-anchor="middle" font-size="26" font-family="Arial" fill="#1f2933">{title}</text>',
    ]

    data_x_min = min(point.x for point in data_points)
    data_x_max = max(point.x for point in data_points)

    for index, curve in enumerate(curves):
        col = index % cols
        row = index // cols
        left = margin_left + col * (panel_w + gap_x)
        top = margin_top + row * (panel_h + gap_y)
        plot_left = left + 46
        plot_right = left + panel_w - 18
        plot_top = top + 34
        plot_bottom = top + panel_h - 38

        panel_points = list(data_points) + curve.points
        y_min = min(point.y for point in panel_points)
        y_max = max(point.y for point in panel_points)
        y_pad = 0.12 * (y_max - y_min if y_max > y_min else 1.0)
        y_min -= y_pad
        y_max += y_pad
        x_min = data_x_min - 0.04 * (data_x_max - data_x_min)
        x_max = data_x_max + 0.04 * (data_x_max - data_x_min)

        def sx(value: float) -> float:
            return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

        def sy(value: float) -> float:
            return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

        selected = curve.law_number in selected_laws
        stroke = "#0b7285" if selected else "#7b8794"
        title_fill = "#0b7285" if selected else "#1f2933"
        border = "#0b7285" if selected else "#d7dde3"
        parts.append(
            f'  <rect x="{left:.2f}" y="{top:.2f}" width="{panel_w}" height="{panel_h}" rx="0" fill="#ffffff" stroke="{border}" stroke-width="{2 if selected else 1}"/>'
        )
        label = f"Закон {curve.law_number}: {curve.formula}"
        parts.append(
            f'  <text x="{left + 14:.2f}" y="{top + 22:.2f}" font-size="14" font-family="Arial" fill="{title_fill}">{label}</text>'
        )
        if selected:
            parts.append(
                f'  <text x="{left + panel_w - 74:.2f}" y="{top + 22:.2f}" font-size="12" font-family="Arial" fill="#0b7285">выбран</text>'
            )

        for xt in _build_ticks(x_min, x_max, 4):
            px = sx(xt)
            if plot_left <= px <= plot_right:
                parts.append(
                    f'  <line x1="{px:.2f}" y1="{plot_top:.2f}" x2="{px:.2f}" y2="{plot_bottom:.2f}" stroke="#eef2f5" stroke-width="1"/>'
                )
                parts.append(
                    f'  <text x="{px:.2f}" y="{plot_bottom + 18:.2f}" text-anchor="middle" font-size="10" font-family="Arial" fill="#59636e">{_fmt_ru(xt, 1)}</text>'
                )
        for yt in _build_ticks(y_min, y_max, 4):
            py = sy(yt)
            if plot_top <= py <= plot_bottom:
                parts.append(
                    f'  <line x1="{plot_left:.2f}" y1="{py:.2f}" x2="{plot_right:.2f}" y2="{py:.2f}" stroke="#eef2f5" stroke-width="1"/>'
                )
                parts.append(
                    f'  <text x="{plot_left - 8:.2f}" y="{py + 3:.2f}" text-anchor="end" font-size="10" font-family="Arial" fill="#59636e">{_fmt_ru(yt, 1)}</text>'
                )

        if curve.points:
            polyline = " ".join(f"{sx(point.x):.2f},{sy(point.y):.2f}" for point in curve.points)
            parts.append(
                f'  <polyline points="{polyline}" fill="none" stroke="{stroke}" stroke-width="{2.4 if selected else 1.8}"/>'
            )

        for point in data_points:
            parts.append(
                f'  <circle cx="{sx(point.x):.2f}" cy="{sy(point.y):.2f}" r="3.1" fill="#111827"/>'
            )

    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_csv(path: Path, xs: np.ndarray, ys: np.ndarray, approximations: Sequence[Approximation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        headers = ["x", "y"] + [f"law_{item.law_number}" for item in approximations]
        file.write(",".join(headers) + "\n")
        for row_idx in range(len(xs)):
            row = [xs[row_idx], ys[row_idx]] + [item.predicted[row_idx] for item in approximations]
            file.write(",".join(f"{float(value):.12f}" for value in row) + "\n")


def build_report(
    path: Path,
    candidates: Sequence[tuple[int, str, str, dict[str, float], float]],
    quadratic: Approximation,
    power: Approximation,
) -> None:
    lines: list[str] = []
    lines.append(f"# Лабораторная работа №6, вариант {VARIANT}")
    lines.append("")
    lines.append("## Постановка задачи")
    lines.append("")
    lines.append("Функция задана таблицей из 9 значений. Нужно выбрать два аппроксимирующих закона из предложенного списка и построить их по методу наименьших квадратов.")
    lines.append("")
    lines.append("Исходные данные:")
    lines.append("")
    lines.append(points_markdown(X_VALUES, Y_VALUES))
    lines.append("")
    lines.append("График исходных точек:")
    lines.append("")
    lines.append("![points](points.svg)")
    lines.append("")
    lines.append("## 1) Выбор аппроксимирующих законов")
    lines.append("")
    lines.append("Для выбора законов были промоделированы графики всех 10 функций из задания. На каждом мини-графике черными точками показаны исходные данные, а линией — соответствующий аппроксимирующий закон с подобранными на компьютере коэффициентами.")
    lines.append("")
    lines.append("![model grid](model_grid.svg)")
    lines.append("")
    lines.append("По виду графиков наиболее естественно подходят:")
    lines.append("")
    lines.append("1. `y = ax² + bx + c` — хорошо повторяет общий плавный изгиб точек.")
    lines.append("2. `y = b·xᵃ + c` — визуально почти совпадает с точками и также хорошо описывает монотонный рост.")
    lines.append("")
    lines.append("Именно эти два закона далее строятся подробно: для них составляются нормальные системы, находятся параметры и рассчитываются итоговые невязки.")
    lines.append("")

    lines.append("## 2) Закон 1: квадратичная аппроксимация")
    lines.append("")
    lines.append("Ищем функцию вида:")
    lines.append("")
    lines.append("**y = ax² + bx + c**")
    lines.append("")
    lines.append("Для МНК нормальная система имеет вид:")
    lines.append("")
    lines.append(
        inline_equations_markdown(
            [
                "a·Σxᵢ⁴ + b·Σxᵢ³ + c·Σxᵢ² = Σxᵢ²yᵢ",
                "a·Σxᵢ³ + b·Σxᵢ² + c·Σxᵢ = Σxᵢyᵢ",
                "a·Σxᵢ² + b·Σxᵢ + c·n = Σyᵢ",
            ]
        )
    )
    lines.append("")
    lines.append("После подстановки сумм:")
    lines.append("")
    lines.append(
        inline_equations_markdown(
            substituted_equations_markdown(quadratic.normal_matrix, quadratic.normal_rhs, ["a", "b", "c"]).splitlines()
        )
    )
    lines.append("")
    lines.append("Решение системы:")
    lines.append("")
    lines.append(f"`a = {_fmt_ru(quadratic.parameters['a'], 9)}`")
    lines.append(f"`b = {_fmt_ru(quadratic.parameters['b'], 9)}`")
    lines.append(f"`c = {_fmt_ru(quadratic.parameters['c'], 9)}`")
    lines.append("")
    lines.append("Полученный закон:")
    lines.append("")
    lines.append(f"**{formula_quadratic(quadratic)}**")
    lines.append("")
    lines.append("Таблица ошибок:")
    lines.append("")
    lines.append(approximation_table(quadratic, X_VALUES, Y_VALUES))
    lines.append("")
    lines.append(f"Невязка: **δ = {_fmt_ru(quadratic.residual_sum, 9)}**.")
    lines.append("")
    lines.append("![quadratic](quadratic.svg)")
    lines.append("")

    lines.append("## 3) Закон 3: степенная аппроксимация со сдвигом")
    lines.append("")
    lines.append("Выбран закон:")
    lines.append("")
    lines.append("**y = b·xᵃ + c**")
    lines.append("")
    lines.append("Параметры `a` и `c` входят нелинейно, поэтому нормальную систему сразу для всех трех параметров составить нельзя.")
    lines.append("")
    lines.append("Поэтому используется вложенный подбор:")
    lines.append("")
    lines.append("1. Берем пробное значение `c` так, чтобы все `yᵢ − c > 0`.")
    lines.append("2. Выполняем замену `tᵢ = ln(xᵢ)`, `zᵢ = ln(yᵢ − c)`.")
    lines.append("3. Получаем линейную зависимость `z ≈ a·t + ln b`.")
    lines.append("4. Для этого `c` находим лучшие `a` и `ln b` через нормальную систему МНК.")
    lines.append("5. Считаем невязку `δ(c)` уже в исходных координатах.")
    lines.append("6. Меняем `c` и повторяем расчет, пока невязка не станет минимальной.")
    lines.append("")
    lines.append("В результате такого моделирования минимум невязки найден при:")
    lines.append("")
    lines.append(f"`c = {_fmt_ru(power.parameters['c'], 9)}`")
    lines.append("")
    lines.append("Ниже показан последний шаг этого процесса: для найденного `c` вводим замену")
    lines.append("")
    lines.append("```text")
    lines.append("tᵢ = ln(xᵢ)")
    lines.append("zᵢ = ln(yᵢ − c)")
    lines.append("z ≈ a·t + ln b")
    lines.append("```")
    lines.append("")
    lines.append("и составляем нормальную систему для окончательных `a` и `ln b`:")
    lines.append("")
    lines.append(
        inline_equations_markdown(
            [
                "a·Σtᵢ² + ln(b)·Σtᵢ = Σtᵢzᵢ",
                "a·Σtᵢ + ln(b)·n = Σzᵢ",
            ]
        )
    )
    lines.append("")
    lines.append("После подстановки сумм:")
    lines.append("")
    lines.append(
        inline_equations_markdown(
            substituted_equations_markdown(power.normal_matrix, power.normal_rhs, ["a", "ln(b)"]).splitlines()
        )
    )
    lines.append("")
    lines.append("Решение системы:")
    lines.append("")
    lines.append(f"`a = {_fmt_ru(power.parameters['a'], 9)}`")
    lines.append(f"`ln b = {_fmt_ru(power.parameters['ln_b'], 9)}`")
    lines.append(f"`b = e^({_fmt_ru(power.parameters['ln_b'], 9)}) = {_fmt_ru(power.parameters['b'], 9)}`")
    lines.append(f"`c = {_fmt_ru(power.parameters['c'], 9)}`")
    lines.append("")
    lines.append("Полученный закон:")
    lines.append("")
    lines.append(f"**{formula_power(power)}**")
    lines.append("")
    lines.append("Таблица ошибок:")
    lines.append("")
    lines.append(approximation_table(power, X_VALUES, Y_VALUES))
    lines.append("")
    lines.append(f"Невязка: **δ = {_fmt_ru(power.residual_sum, 9)}**.")
    lines.append("")
    lines.append("![power](power.svg)")
    lines.append("")

    lines.append("## 4) Сравнение")
    lines.append("")
    lines.append("| закон | функция | невязка δ |")
    lines.append("|---:|:---|---:|")
    lines.append(f"| 1 | {formula_quadratic(quadratic)} | {_fmt_ru(quadratic.residual_sum, 9)} |")
    lines.append(f"| 3 | {formula_power(power)} | {_fmt_ru(power.residual_sum, 9)} |")
    lines.append("")
    lines.append("Лучший из двух выбранных законов по невязке — квадратичная аппроксимация, но степенной закон визуально также хорошо описывает исходные точки.")
    lines.append("")
    lines.append("Общий график:")
    lines.append("")
    lines.append("![combined](combined.svg)")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, candidates: Sequence[tuple[int, str, str, dict[str, float], float]], approximations: Sequence[Approximation]) -> None:
    payload = {
        "variant": VARIANT,
        "points": [{"x": float(x), "y": float(y)} for x, y in zip(X_VALUES, Y_VALUES)],
        "candidates": [
            {
                "law_number": number,
                "name": name,
                "formula": formula,
                "parameters": params,
                "residual_sum": residual,
            }
            for number, name, formula, params, residual in candidates
        ],
        "selected": [asdict(item) for item in approximations],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    quadratic = fit_quadratic(X_VALUES, Y_VALUES)
    power = fit_power_shift(X_VALUES, Y_VALUES)
    candidates = candidate_summary(X_VALUES, Y_VALUES)
    approximations = [quadratic, power]

    data_points = [Point(float(x), float(y)) for x, y in zip(X_VALUES, Y_VALUES)]
    left, right = float(min(X_VALUES)), float(max(X_VALUES))
    candidate_curves = build_candidate_curves(candidates, left, right)

    qa, qb, qc = quadratic.parameters["a"], quadratic.parameters["b"], quadratic.parameters["c"]
    pa, pb, pc = power.parameters["a"], power.parameters["b"], power.parameters["c"]

    quadratic_series = GraphSeries(
        name="закон 1",
        color="#0b7285",
        points=sample_points(lambda value: qa * value**2 + qb * value + qc, left, right),
    )
    power_series = GraphSeries(
        name="закон 3",
        color="#d9480f",
        points=sample_points(lambda value: pb * value**pa + pc, left, right),
        dasharray="8,5",
    )

    write_svg(results_dir / "points.svg", "Исходные точки варианта 17", data_points, [])
    write_model_grid_svg(
        results_dir / "model_grid.svg",
        "Моделирование 10 аппроксимирующих законов",
        data_points,
        candidate_curves,
        selected_laws={1, 3},
    )
    write_svg(results_dir / "quadratic.svg", "Аппроксимация законом 1", data_points, [quadratic_series])
    write_svg(results_dir / "power.svg", "Аппроксимация законом 3", data_points, [power_series])
    write_svg(results_dir / "combined.svg", "Сравнение двух аппроксимаций", data_points, [quadratic_series, power_series])

    write_csv(results_dir / "approximation_points.csv", X_VALUES, Y_VALUES, approximations)
    build_report(results_dir / "report.md", candidates, quadratic, power)
    write_json(results_dir / "results.json", candidates, approximations)

    print(f"Lab 6 variant {VARIANT} solved.")
    print(f"Results directory: {results_dir}")
    print(
        "Law 1:",
        f"a={quadratic.parameters['a']:.9f}",
        f"b={quadratic.parameters['b']:.9f}",
        f"c={quadratic.parameters['c']:.9f}",
        f"delta={quadratic.residual_sum:.12f}",
    )
    print(
        "Law 3:",
        f"a={power.parameters['a']:.9f}",
        f"b={power.parameters['b']:.9f}",
        f"c={power.parameters['c']:.9f}",
        f"delta={power.residual_sum:.12f}",
    )


if __name__ == "__main__":
    main()
