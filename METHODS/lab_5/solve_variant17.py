from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence


VARIANT = 17

# Variant 17 from the lab table.
X_VALUES = [2.119, 3.618, 5.342, 7.859, 8.934]
Y_VALUES = [0.605, 0.718, 0.105, 2.157, 3.431]

# The lab asks to evaluate L4(x1 + x2) and N4(x1 + x2).
QUERY_X = X_VALUES[1] + X_VALUES[2]


@dataclass
class Point:
    x: float
    y: float


@dataclass
class LinearSegment:
    index: int
    x_left: float
    x_right: float
    a: float
    b: float


@dataclass
class QuadraticSegment:
    index: int
    x_left: float
    x_right: float
    node_indices: list[int]
    a: float
    b: float
    c: float


@dataclass
class GraphSeries:
    name: str
    color: str
    points: list[Point]
    width: float = 2.6
    dasharray: str | None = None


def _fmt(value: float, digits: int = 6) -> str:
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}"


def _fmt_ru(value: float, digits: int = 6) -> str:
    return _fmt(value, digits).replace(".", ",").replace("-", "−")


def _fmt_compact(value: float, digits: int = 6) -> str:
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _fmt_ru_compact(value: float, digits: int = 6) -> str:
    return _fmt_compact(value, digits).replace(".", ",").replace("-", "−")


def _fmt_sign(value: float, digits: int = 6) -> str:
    sign = "+" if value >= 0 else "-"
    return f" {sign} {_fmt(abs(value), digits)}"


def _fmt_ru_sign(value: float, digits: int = 6) -> str:
    sign = "+" if value >= 0 else "−"
    return f" {sign} {_fmt_ru(abs(value), digits)}"


SUBSCRIPT = str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋")
SUPERSCRIPT = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


def subscript(value: int | str) -> str:
    return str(value).translate(SUBSCRIPT)


def superscript(value: int | str) -> str:
    return str(value).translate(SUPERSCRIPT)


def interval_text(left: float, right: float) -> str:
    return f"[{_fmt_ru(left, 3)}; {_fmt_ru(right, 3)}]"


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


def poly_eval(coefficients: Sequence[float], x: float) -> float:
    result = 0.0
    for coeff in reversed(coefficients):
        result = result * x + coeff
    return result


def poly_add(left: Sequence[float], right: Sequence[float]) -> list[float]:
    size = max(len(left), len(right))
    result = [0.0] * size
    for i in range(size):
        result[i] = (left[i] if i < len(left) else 0.0) + (
            right[i] if i < len(right) else 0.0
        )
    return result


def poly_scale(poly: Sequence[float], factor: float) -> list[float]:
    return [factor * coeff for coeff in poly]


def poly_mul_linear(poly: Sequence[float], root: float) -> list[float]:
    # Multiplies a polynomial by (x - root). Coefficients are stored ascending.
    result = [0.0] * (len(poly) + 1)
    for i, coeff in enumerate(poly):
        result[i] -= root * coeff
        result[i + 1] += coeff
    return result


def polynomial_from_lagrange_nodes(
    xs: Sequence[float], ys: Sequence[float], indices: Sequence[int]
) -> list[float]:
    poly = [0.0]
    for i in indices:
        term = [ys[i]]
        denom = 1.0
        for j in indices:
            if i == j:
                continue
            term = poly_mul_linear(term, xs[j])
            denom *= xs[i] - xs[j]
        poly = poly_add(poly, poly_scale(term, 1.0 / denom))
    return poly


def lagrange_basis_multipliers(
    xs: Sequence[float], ys: Sequence[float]
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for i, xi in enumerate(xs):
        denom = 1.0
        for j, xj in enumerate(xs):
            if i != j:
                denom *= xi - xj
        rows.append(
            {
                "i": i,
                "x_i": xi,
                "y_i": ys[i],
                "denominator": denom,
                "multiplier": ys[i] / denom,
            }
        )
    return rows


def lagrange_value(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    value = 0.0
    for i, xi in enumerate(xs):
        term = ys[i]
        for j, xj in enumerate(xs):
            if i != j:
                term *= (x - xj) / (xi - xj)
        value += term
    return value


def finite_differences(ys: Sequence[float]) -> list[list[float]]:
    columns = [list(ys)]
    while len(columns[-1]) > 1:
        prev = columns[-1]
        columns.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])
    return columns


def divided_differences(xs: Sequence[float], ys: Sequence[float]) -> list[list[float]]:
    n = len(xs)
    table = [[0.0] * n for _ in range(n)]
    for i, value in enumerate(ys):
        table[i][0] = value
    for order in range(1, n):
        for i in range(n - order):
            table[i][order] = (table[i + 1][order - 1] - table[i][order - 1]) / (
                xs[i + order] - xs[i]
            )
    return table


def newton_power_coefficients(xs: Sequence[float], coefficients: Sequence[float]) -> list[float]:
    poly = [0.0]
    term = [1.0]
    for order, coeff in enumerate(coefficients):
        poly = poly_add(poly, poly_scale(term, coeff))
        if order < len(xs) - 1:
            term = poly_mul_linear(term, xs[order])
    return poly


def newton_value(xs: Sequence[float], coefficients: Sequence[float], x: float) -> float:
    value = coefficients[-1]
    for i in range(len(coefficients) - 2, -1, -1):
        value = value * (x - xs[i]) + coefficients[i]
    return value


def build_linear_segments(xs: Sequence[float], ys: Sequence[float]) -> list[LinearSegment]:
    segments: list[LinearSegment] = []
    for i in range(len(xs) - 1):
        a = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
        b = ys[i] - a * xs[i]
        segments.append(
            LinearSegment(
                index=i + 1,
                x_left=xs[i],
                x_right=xs[i + 1],
                a=a,
                b=b,
            )
        )
    return segments


def build_quadratic_segments(
    xs: Sequence[float], ys: Sequence[float]
) -> list[QuadraticSegment]:
    groups = [(0, 1, 2), (2, 3, 4)]
    segments: list[QuadraticSegment] = []
    for number, group in enumerate(groups, start=1):
        coeffs = polynomial_from_lagrange_nodes(xs, ys, group)
        coeffs = coeffs + [0.0] * (3 - len(coeffs))
        c, b, a = coeffs[:3]
        segments.append(
            QuadraticSegment(
                index=number,
                x_left=xs[group[0]],
                x_right=xs[group[-1]],
                node_indices=[i for i in group],
                a=a,
                b=b,
                c=c,
            )
        )
    return segments


def evaluate_linear_spline(segments: Sequence[LinearSegment], x: float) -> float:
    for segment in segments:
        if segment.x_left - 1e-12 <= x <= segment.x_right + 1e-12:
            return segment.a * x + segment.b
    segment = segments[0] if x < segments[0].x_left else segments[-1]
    return segment.a * x + segment.b


def evaluate_quadratic_spline(segments: Sequence[QuadraticSegment], x: float) -> float:
    for segment in segments:
        if segment.x_left - 1e-12 <= x <= segment.x_right + 1e-12:
            return segment.a * x * x + segment.b * x + segment.c
    segment = segments[0] if x < segments[0].x_left else segments[-1]
    return segment.a * x * x + segment.b * x + segment.c


def sample_function(
    func: Callable[[float], float], x_min: float, x_max: float, count: int = 500
) -> list[Point]:
    return [
        Point(
            x=x_min + (x_max - x_min) * i / (count - 1),
            y=func(x_min + (x_max - x_min) * i / (count - 1)),
        )
        for i in range(count)
    ]


def points_table(points: Sequence[Point]) -> str:
    lines = ["| i | xᵢ | yᵢ |", "|---:|---:|---:|"]
    for i, point in enumerate(points):
        lines.append(f"| {i} | {_fmt_ru(point.x, 3)} | {_fmt_ru(point.y, 3)} |")
    return "\n".join(lines)


def finite_difference_table_markdown(xs: Sequence[float], columns: Sequence[Sequence[float]]) -> str:
    headers = ["xₖ", "yₖ"] + [
        f"Δ{superscript(order) if order > 1 else ''}yₖ"
        for order in range(1, len(columns))
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---:"] * len(headers)) + " |",
    ]
    for i, x in enumerate(xs):
        cells = [_fmt_ru(x, 3)]
        for order, column in enumerate(columns):
            if i < len(column):
                digits = 3 if order == 0 else 6
                cells.append(_fmt_ru(column[i], digits))
            else:
                cells.append("")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def divided_header(order: int) -> str:
    if order == 0:
        return "yₖ"
    return f"{order}-го порядка"


def divided_difference_table_markdown(xs: Sequence[float], table: Sequence[Sequence[float]]) -> str:
    headers = ["xₖ"] + [divided_header(order) for order in range(len(xs))]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---:"] * len(headers)) + " |",
    ]
    n = len(xs)
    for i, x in enumerate(xs):
        cells = [_fmt_ru(x, 3)]
        for order in range(n):
            if i <= n - order - 1:
                digits = 3 if order == 0 else 9
                cells.append(_fmt_ru(table[i][order], digits))
            else:
                cells.append("")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def basis_multiplier_table(rows: Sequence[dict[str, float | int]]) -> str:
    lines = [
        "| i | yᵢ | ∏(xᵢ − xⱼ), j ≠ i | 1 / ∏(xᵢ − xⱼ) | yᵢ / ∏(xᵢ − xⱼ) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        denom = float(row["denominator"])
        lines.append(
            f"| {row['i']} | {_fmt_ru(float(row['y_i']), 3)} | {_fmt_ru(denom, 9)} | {_fmt_ru(1.0 / denom, 9)} | {_fmt_ru(float(row['multiplier']), 9)} |"
        )
    return "\n".join(lines)


def lagrange_denominator_calculation_lines(
    xs: Sequence[float], basis_rows: Sequence[dict[str, float | int]]
) -> str:
    lines: list[str] = []
    for row in basis_rows:
        i = int(row["i"])
        factors = "·".join(
            f"({_fmt_ru_compact(xs[i], 3)} − {_fmt_ru_compact(xj, 3)})"
            for j, xj in enumerate(xs)
            if j != i
        )
        lines.append(
            f"D{subscript(i)} = {factors} = {_fmt_ru(float(row['denominator']), 9)}"
        )
    return "\n".join(lines)


def lagrange_basis_formula_lines(
    xs: Sequence[float], basis_rows: Sequence[dict[str, float | int]]
) -> str:
    lines: list[str] = []
    for row in basis_rows:
        i = int(row["i"])
        numerator = "·".join(
            f"(x − {_fmt_ru_compact(xj, 3)})" for j, xj in enumerate(xs) if j != i
        )
        denom = _fmt_ru(float(row["denominator"]), 9)
        reciprocal = _fmt_ru(1.0 / float(row["denominator"]), 9)
        lines.append(
            f"p{subscript(i)}(x) = [{numerator}] / {denom} = {reciprocal}·{numerator}"
        )
    return "\n".join(lines)


def lagrange_linear_combination(ys: Sequence[float]) -> str:
    parts: list[str] = []
    for i, y in enumerate(ys):
        term = f"{_fmt_ru(abs(y), 3)}·p{subscript(i)}(x)"
        if not parts:
            parts.append(term if y >= 0 else f"−{term}")
        else:
            parts.append((" + " if y >= 0 else " − ") + term)
    return "".join(parts)


def linear_spline_table(segments: Sequence[LinearSegment]) -> str:
    lines = [
        "| № | интервал | aᵢ | bᵢ | Sᵢ(x) = aᵢx + bᵢ |",
        "|---:|:---|---:|---:|:---|",
    ]
    for segment in segments:
        sign = "+" if segment.b >= 0 else "−"
        lines.append(
            f"| {segment.index} | {interval_text(segment.x_left, segment.x_right)} | "
            f"{_fmt_ru(segment.a, 9)} | {_fmt_ru(segment.b, 9)} | "
            f"{_fmt_ru(segment.a, 6)}·x {sign} {_fmt_ru(abs(segment.b), 6)} |"
        )
    return "\n".join(lines)


def linear_piecewise_template(segments: Sequence[LinearSegment]) -> str:
    lines = ["φ(x) = {"]
    for segment in segments:
        comma = "," if segment.index < len(segments) else "."
        lines.append(
            f"  a{subscript(segment.index)}·x + b{subscript(segment.index)},  "
            f"{_fmt_ru(segment.x_left, 3)} ≤ x ≤ {_fmt_ru(segment.x_right, 3)}{comma}"
        )
    lines.append("}")
    return "\n".join(lines)


def linear_piecewise_solution(segments: Sequence[LinearSegment]) -> str:
    lines = ["φ(x) = {"]
    for segment in segments:
        sign = "+" if segment.b >= 0 else "−"
        comma = "," if segment.index < len(segments) else "."
        lines.append(
            f"  {_fmt_ru(segment.a, 6)}·x {sign} {_fmt_ru(abs(segment.b), 6)},  "
            f"{_fmt_ru(segment.x_left, 3)} ≤ x ≤ {_fmt_ru(segment.x_right, 3)}{comma}"
        )
    lines.append("}")
    return "\n".join(lines)


def linear_systems_text(segments: Sequence[LinearSegment], ys: Sequence[float]) -> str:
    blocks: list[str] = []
    for segment in segments:
        i = segment.index
        blocks.append(
            "\n".join(
                [
                    f"S{subscript(i)}:",
                    f"  {_fmt_ru(segment.x_left, 3)}·a{subscript(i)} + b{subscript(i)} = {_fmt_ru(ys[i - 1], 3)}",
                    f"  {_fmt_ru(segment.x_right, 3)}·a{subscript(i)} + b{subscript(i)} = {_fmt_ru(ys[i], 3)}",
                ]
            )
        )
    return "\n\n".join(blocks)


def quadratic_spline_table(segments: Sequence[QuadraticSegment]) -> str:
    lines = [
        "| № | интервал | узлы | aᵢ | bᵢ | cᵢ | Qᵢ(x) = aᵢx² + bᵢx + cᵢ |",
        "|---:|:---|:---|---:|---:|---:|:---|",
    ]
    for segment in segments:
        formula = (
            f"{_fmt_ru(segment.a, 6)}·x²"
            f"{_fmt_ru_sign(segment.b, 6)}·x"
            f"{_fmt_ru_sign(segment.c, 6)}"
        )
        node_text = ", ".join(f"x{subscript(i)}" for i in segment.node_indices)
        lines.append(
            f"| {segment.index} | {interval_text(segment.x_left, segment.x_right)} | "
            f"{node_text} | {_fmt_ru(segment.a, 9)} | {_fmt_ru(segment.b, 9)} | {_fmt_ru(segment.c, 9)} | {formula} |"
        )
    return "\n".join(lines)


def quadratic_piecewise_template(segments: Sequence[QuadraticSegment]) -> str:
    lines = ["φ(x) = {"]
    for segment in segments:
        comma = "," if segment.index < len(segments) else "."
        lines.append(
            f"  a{subscript(segment.index)}·x² + b{subscript(segment.index)}·x + c{subscript(segment.index)},  "
            f"x ∈ {interval_text(segment.x_left, segment.x_right)}{comma}"
        )
    lines.append("}")
    return "\n".join(lines)


def quadratic_piecewise_solution(segments: Sequence[QuadraticSegment]) -> str:
    lines = ["φ(x) = {"]
    for segment in segments:
        comma = "," if segment.index < len(segments) else "."
        formula = (
            f"{_fmt_ru(segment.a, 6)}·x²"
            f"{_fmt_ru_sign(segment.b, 6)}·x"
            f"{_fmt_ru_sign(segment.c, 6)}"
        )
        lines.append(
            f"  {formula},  x ∈ {interval_text(segment.x_left, segment.x_right)}{comma}"
        )
    lines.append("}")
    return "\n".join(lines)


def quadratic_systems_text(
    segments: Sequence[QuadraticSegment], xs: Sequence[float], ys: Sequence[float]
) -> str:
    blocks: list[str] = []
    for segment in segments:
        i = segment.index
        equations = [f"Q{subscript(i)}:"]
        for node_index in segment.node_indices:
            x = xs[node_index]
            y = ys[node_index]
            equations.append(
                f"  {_fmt_ru(x * x, 6)}·a{subscript(i)} + "
                f"{_fmt_ru(x, 3)}·b{subscript(i)} + c{subscript(i)} = {_fmt_ru(y, 3)}"
            )
        blocks.append("\n".join(equations))
    return "\n\n".join(blocks)


def polynomial_power_formula(coefficients: Sequence[float], variable: str = "x") -> str:
    parts: list[str] = []
    for degree, coeff in enumerate(coefficients):
        if abs(coeff) < 1e-12:
            continue
        if degree == 0:
            term = _fmt_ru(abs(coeff), 6)
        elif degree == 1:
            term = f"{_fmt_ru(abs(coeff), 6)}·{variable}"
        else:
            term = f"{_fmt_ru(abs(coeff), 6)}·{variable}{superscript(degree)}"

        if not parts:
            parts.append(term if coeff >= 0 else f"−{term}")
        else:
            parts.append((" + " if coeff >= 0 else " − ") + term)
    return "".join(parts) if parts else "0"


def lagrange_product_formula(
    xs: Sequence[float], basis_rows: Sequence[dict[str, float | int]]
) -> str:
    lines: list[str] = []
    for row in basis_rows:
        i = int(row["i"])
        multiplier = float(row["multiplier"])
        factors = "·".join(
            f"(x − {_fmt_ru_compact(xj, 3)})" for j, xj in enumerate(xs) if j != i
        )
        if i == 0:
            prefix = "−" if multiplier < 0 else ""
        else:
            prefix = "<br>− " if multiplier < 0 else "<br>+ "
        lines.append(f"{prefix}{_fmt_ru(abs(multiplier), 9)}·{factors}")
    return "\n".join(lines)


def newton_product_formula(xs: Sequence[float], coefficients: Sequence[float]) -> str:
    parts = [_fmt_ru(coefficients[0], 9)]
    for order in range(1, len(coefficients)):
        factors = "·".join(
            f"(x − {_fmt_ru_compact(xs[i], 3)})" for i in range(order)
        )
        coeff = coefficients[order]
        parts.append(
            ("<br>− " if coeff < 0 else "<br>+ ")
            + f"{_fmt_ru(abs(coeff), 9)}·{factors}"
        )
    return "\n".join(parts)


def write_csv(
    path: Path,
    xs: Sequence[float],
    lagrange_y: Sequence[float],
    newton_y: Sequence[float],
    linear_y: Sequence[float],
    quadratic_y: Sequence[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write("x,lagrange,newton,linear_spline,quadratic_spline\n")
        for row in zip(xs, lagrange_y, newton_y, linear_y, quadratic_y):
            file.write(",".join(f"{value:.12f}" for value in row) + "\n")


def write_svg(path: Path, title: str, series: Sequence[GraphSeries], nodes: Sequence[Point]) -> None:
    width, height = 980, 640
    margin_left, margin_right = 86, 36
    margin_top, margin_bottom = 68, 76

    all_points: list[Point] = []
    for item in series:
        all_points.extend(item.points)
    all_points.extend(nodes)

    x_min = min(point.x for point in all_points)
    x_max = max(point.x for point in all_points)
    y_min = min(point.y for point in all_points)
    y_max = max(point.y for point in all_points)

    x_pad = 0.03 * (x_max - x_min)
    y_pad = 0.12 * (y_max - y_min if y_max > y_min else 1.0)
    x_min -= x_pad
    x_max += x_pad
    y_min = min(y_min - y_pad, -0.25)
    y_max = max(y_max + y_pad, 0.25)

    plot_left, plot_right = margin_left, width - margin_right
    plot_top, plot_bottom = margin_top, height - margin_bottom

    def sx(x: float) -> float:
        return plot_left + (x - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def sy(y: float) -> float:
        return plot_bottom - (y - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    x_axis = sy(0.0) if y_min <= 0.0 <= y_max else plot_bottom
    y_axis = sx(0.0) if x_min <= 0.0 <= x_max else plot_left

    x_ticks = _build_ticks(x_min, x_max, target_ticks=8)
    y_ticks = _build_ticks(y_min, y_max, target_ticks=8)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '  <rect x="0" y="0" width="980" height="640" fill="white"/>',
        f'  <text x="{width / 2:.0f}" y="34" text-anchor="middle" font-size="24" font-family="Arial" fill="#1f2933">{title}</text>',
    ]

    for xt in x_ticks:
        px = sx(xt)
        if plot_left <= px <= plot_right:
            parts.append(
                f'  <line x1="{px:.2f}" y1="{plot_top:.2f}" x2="{px:.2f}" y2="{plot_bottom:.2f}" stroke="#eceff3" stroke-width="1"/>'
            )
            parts.append(
                f'  <text x="{px:.2f}" y="{plot_bottom + 22:.2f}" text-anchor="middle" font-size="12" font-family="Arial" fill="#4b5563">{_fmt_compact(xt, 2)}</text>'
            )

    for yt in y_ticks:
        py = sy(yt)
        if plot_top <= py <= plot_bottom:
            parts.append(
                f'  <line x1="{plot_left:.2f}" y1="{py:.2f}" x2="{plot_right:.2f}" y2="{py:.2f}" stroke="#eceff3" stroke-width="1"/>'
            )
            parts.append(
                f'  <text x="{plot_left - 12:.2f}" y="{py + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#4b5563">{_fmt_compact(yt, 2)}</text>'
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

    for i, point in enumerate(nodes):
        parts.append(
            f'  <circle cx="{sx(point.x):.2f}" cy="{sy(point.y):.2f}" r="4.4" fill="#111827"/>'
        )
        parts.append(
            f'  <text x="{sx(point.x) + 7:.2f}" y="{sy(point.y) - 7:.2f}" font-size="11" font-family="Arial" fill="#111827">P{i}</text>'
        )

    legend_x = plot_left + 10
    legend_y = plot_top + 14
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


def build_report(
    path: Path,
    nodes: list[Point],
    lagrange_coefficients: list[float],
    lagrange_basis_rows: list[dict[str, float | int]],
    finite_table: list[list[float]],
    divided_table: list[list[float]],
    newton_coefficients: list[float],
    newton_power_coeffs: list[float],
    linear_segments: list[LinearSegment],
    quadratic_segments: list[QuadraticSegment],
    lagrange_at_query: float,
    newton_at_query: float,
) -> None:
    lines: list[str] = []
    lines.append(f"# Лабораторная работа №5, вариант {VARIANT}")
    lines.append("")
    lines.append("## Постановка задачи")
    lines.append("")
    lines.append("Функция задана таблично в пяти узлах. Требуется:")
    lines.append("")
    lines.append("1. Построить интерполяционный многочлен Лагранжа и вычислить `L₄(x₁ + x₂)`.")
    lines.append("2. Построить таблицы конечных и разделенных разностей.")
    lines.append("3. Построить полином Ньютона и вычислить `N₄(x₁ + x₂)`.")
    lines.append("4. Построить линейный и квадратичный интерполяционные сплайны.")
    lines.append("5. Построить графики многочленов и сплайнов.")
    lines.append("")
    lines.append("Исходные узлы варианта 17:")
    lines.append("")
    lines.append(points_table(nodes))
    lines.append("")
    lines.append(
        f"Точка вычисления: `x₁ + x₂ = {_fmt_ru(X_VALUES[1], 3)} + {_fmt_ru(X_VALUES[2], 3)} = {_fmt_ru(QUERY_X, 3)}`."
    )
    if QUERY_X > X_VALUES[-1]:
        lines.append(
            f"Значение немного выходит за последний узел `x₄ = {_fmt_ru(X_VALUES[-1], 3)}`, поэтому вычисление является малой экстраполяцией."
        )
    lines.append("")

    lines.append("## 1) Многочлен Лагранжа")
    lines.append("")
    lines.append("Сначала строится сам интерполяционный многочлен `L₄(x)`, а затем в него подставляется `x = x₁ + x₂`.")
    lines.append("")
    lines.append("Числа в знаменателях получаются как произведения разностей выбранного узла со всеми остальными узлами:")
    lines.append("")
    lines.append("```text")
    lines.append(lagrange_denominator_calculation_lines(X_VALUES, lagrange_basis_rows))
    lines.append("```")
    lines.append("")
    lines.append("Базисные многочлены Лагранжа имеют вид:")
    lines.append("")
    lines.append("```text")
    lines.append(lagrange_basis_formula_lines(X_VALUES, lagrange_basis_rows))
    lines.append("```")
    lines.append("")
    lines.append("Здесь `Dᵢ = ∏(xᵢ − xⱼ)`, `j ≠ i`. Например, `D₀ = (2,119 − 3,618)(2,119 − 5,342)(2,119 − 7,859)(2,119 − 8,934) = 188,990376814`.")
    lines.append("")
    lines.append("Числовые коэффициенты:")
    lines.append("")
    lines.append(basis_multiplier_table(lagrange_basis_rows))
    lines.append("")
    lines.append("Линейная комбинация базисных многочленов:")
    lines.append("")
    lines.append("**L₄(x)** = " + lagrange_linear_combination(Y_VALUES))
    lines.append("")
    lines.append("После подстановки всех базисных множителей:")
    lines.append("")
    lines.append("**L₄(x)** = " + lagrange_product_formula(X_VALUES, lagrange_basis_rows))
    lines.append("")
    lines.append("В степенной форме:")
    lines.append("")
    lines.append("**L₄(x)** = " + polynomial_power_formula(lagrange_coefficients))
    lines.append("")
    lines.append(
        f"**L₄(x₁ + x₂) = L₄({_fmt_ru(QUERY_X, 3)}) = {_fmt_ru(lagrange_at_query, 12)}**"
    )
    lines.append("")
    lines.append("График:")
    lines.append("")
    lines.append("![lagrange](lagrange.svg)")
    lines.append("")

    lines.append("## 2) Таблицы разностей")
    lines.append("")
    lines.append("Таблица конечных разностей:")
    lines.append("")
    lines.append(finite_difference_table_markdown(X_VALUES, finite_table))
    lines.append("")
    lines.append("Таблица разделенных разностей:")
    lines.append("")
    lines.append(divided_difference_table_markdown(X_VALUES, divided_table))
    lines.append("")
    lines.append(
        "В этой таблице столбец `1-го порядка` соответствует разделенным разностям вида `f[xₖ; xₖ₊₁]`, столбец `2-го порядка` — `f[xₖ; xₖ₊₁; xₖ₊₂]` и так далее."
    )
    lines.append("")

    lines.append("## 3) Полином Ньютона")
    lines.append("")
    lines.append("Коэффициенты первой строки таблицы разделенных разностей:")
    lines.append("")
    lines.append(
        "`"
        + ", ".join(_fmt_ru(value, 9) for value in newton_coefficients)
        + "`"
    )
    lines.append("")
    lines.append("Общая формула полинома Ньютона для пяти узлов:")
    lines.append("")
    lines.append(
        "**N₄(x)** = f[x₀] + f[x₀; x₁]·(x − x₀) + f[x₀; x₁; x₂]·(x − x₀)·(x − x₁)"
    )
    lines.append(
        "<br>+ f[x₀; x₁; x₂; x₃]·(x − x₀)·(x − x₁)·(x − x₂)"
    )
    lines.append(
        "<br>+ f[x₀; x₁; x₂; x₃; x₄]·(x − x₀)·(x − x₁)·(x − x₂)·(x − x₃)"
    )
    lines.append("")
    lines.append("После подстановки разделенных разностей:")
    lines.append("")
    lines.append("**N₄(x)** = " + newton_product_formula(X_VALUES, newton_coefficients))
    lines.append("")
    lines.append("В степенной форме:")
    lines.append("")
    lines.append("**N₄(x)** = " + polynomial_power_formula(newton_power_coeffs))
    lines.append("")
    lines.append(
        f"**N₄(x₁ + x₂) = N₄({_fmt_ru(QUERY_X, 3)}) = {_fmt_ru(newton_at_query, 12)}**"
    )
    lines.append("")
    lines.append(
        f"Контроль: `|L₄ − N₄| = {_fmt_ru(abs(lagrange_at_query - newton_at_query), 12)}`."
    )
    lines.append("")
    lines.append("График:")
    lines.append("")
    lines.append("![newton](newton.svg)")
    lines.append("")

    lines.append("## 4) Интерполяционные сплайны")
    lines.append("")
    lines.append("### Кусочно-линейная аппроксимация")
    lines.append("")
    lines.append("На каждом интервале `[xᵢ; xᵢ₊₁]` линейное звено имеет вид `Sᵢ(x) = aᵢx + bᵢ`.")
    lines.append("")
    lines.append("Общий вид кусочно-линейной функции:")
    lines.append("")
    lines.append("```text")
    lines.append(linear_piecewise_template(linear_segments))
    lines.append("```")
    lines.append("")
    lines.append("Для нахождения коэффициентов составляются системы:")
    lines.append("")
    lines.append("```text")
    lines.append(linear_systems_text(linear_segments, Y_VALUES))
    lines.append("```")
    lines.append("")
    lines.append("Решая эти системы, получаем:")
    lines.append("")
    lines.append(linear_spline_table(linear_segments))
    lines.append("")
    lines.append("Тогда линейный сплайн имеет вид:")
    lines.append("")
    lines.append("```text")
    lines.append(linear_piecewise_solution(linear_segments))
    lines.append("```")
    lines.append("")
    lines.append("### Кусочно-квадратичная аппроксимация")
    lines.append("")
    lines.append("Квадратичный сплайн строится двумя звеньями: по узлам `(x₀, x₁, x₂)` и `(x₂, x₃, x₄)`.")
    lines.append("Каждое звено имеет вид `Qᵢ(x) = aᵢx² + bᵢx + cᵢ`.")
    lines.append("")
    lines.append("Общий вид кусочно-квадратичной функции:")
    lines.append("")
    lines.append("```text")
    lines.append(quadratic_piecewise_template(quadratic_segments))
    lines.append("```")
    lines.append("")
    lines.append("Системы для коэффициентов:")
    lines.append("")
    lines.append("```text")
    lines.append(quadratic_systems_text(quadratic_segments, X_VALUES, Y_VALUES))
    lines.append("```")
    lines.append("")
    lines.append("Решения систем:")
    lines.append("")
    lines.append(quadratic_spline_table(quadratic_segments))
    lines.append("")
    lines.append("Тогда квадратичный сплайн имеет вид:")
    lines.append("")
    lines.append("```text")
    lines.append(quadratic_piecewise_solution(quadratic_segments))
    lines.append("```")
    lines.append("")
    lines.append("Графики сплайнов:")
    lines.append("")
    lines.append("![linear](linear_spline.svg)")
    lines.append("")
    lines.append("![quadratic](quadratic_spline.svg)")
    lines.append("")

    lines.append("## 5) Общий график и вывод")
    lines.append("")
    lines.append("На одном чертеже показаны оба многочлена и оба сплайна:")
    lines.append("")
    lines.append("![combined](combined.svg)")
    lines.append("")
    lines.append("| Метод | Значение в x₁ + x₂ |")
    lines.append("|:---|---:|")
    lines.append(f"| Лагранж | {_fmt_ru(lagrange_at_query, 12)} |")
    lines.append(f"| Ньютон | {_fmt_ru(newton_at_query, 12)} |")
    lines.append("")
    lines.append(
        "Многочлены Лагранжа и Ньютона совпали с точностью вычислений, потому что являются разными формами одного интерполяционного полинома 4-й степени."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(
    path: Path,
    nodes: list[Point],
    lagrange_coefficients: list[float],
    lagrange_basis_rows: list[dict[str, float | int]],
    finite_table: list[list[float]],
    divided_table: list[list[float]],
    newton_coefficients: list[float],
    newton_power_coeffs: list[float],
    linear_segments: list[LinearSegment],
    quadratic_segments: list[QuadraticSegment],
    lagrange_at_query: float,
    newton_at_query: float,
) -> None:
    payload = {
        "variant": VARIANT,
        "nodes": [asdict(node) for node in nodes],
        "query": {
            "expression": "x1 + x2",
            "x": QUERY_X,
            "is_extrapolation": QUERY_X > X_VALUES[-1],
        },
        "lagrange": {
            "power_coefficients_ascending": lagrange_coefficients,
            "basis": lagrange_basis_rows,
            "value_at_query": lagrange_at_query,
        },
        "finite_differences": finite_table,
        "divided_differences": divided_table,
        "newton": {
            "coefficients": newton_coefficients,
            "power_coefficients_ascending": newton_power_coeffs,
            "value_at_query": newton_at_query,
        },
        "splines": {
            "linear": [asdict(segment) for segment in linear_segments],
            "quadratic": [asdict(segment) for segment in quadratic_segments],
        },
        "graphs": [
            "lagrange.svg",
            "newton.svg",
            "linear_spline.svg",
            "quadratic_spline.svg",
            "combined.svg",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    nodes = [Point(x=x, y=y) for x, y in zip(X_VALUES, Y_VALUES)]

    lagrange_coefficients = polynomial_from_lagrange_nodes(
        X_VALUES, Y_VALUES, range(len(X_VALUES))
    )
    lagrange_basis_rows = lagrange_basis_multipliers(X_VALUES, Y_VALUES)
    lagrange_at_query = lagrange_value(X_VALUES, Y_VALUES, QUERY_X)

    finite_table = finite_differences(Y_VALUES)
    divided_table = divided_differences(X_VALUES, Y_VALUES)
    newton_coefficients = divided_table[0][:]
    newton_power_coeffs = newton_power_coefficients(X_VALUES, newton_coefficients)
    newton_at_query = newton_value(X_VALUES, newton_coefficients, QUERY_X)

    linear_segments = build_linear_segments(X_VALUES, Y_VALUES)
    quadratic_segments = build_quadratic_segments(X_VALUES, Y_VALUES)

    x_min, x_max = min(X_VALUES), max(X_VALUES)
    graph_xs = [x_min + (x_max - x_min) * i / 499 for i in range(500)]

    lagrange_ys = [poly_eval(lagrange_coefficients, x) for x in graph_xs]
    newton_ys = [poly_eval(newton_power_coeffs, x) for x in graph_xs]
    linear_ys = [evaluate_linear_spline(linear_segments, x) for x in graph_xs]
    quadratic_ys = [
        evaluate_quadratic_spline(quadratic_segments, x) for x in graph_xs
    ]

    write_csv(
        results_dir / "graph_points.csv",
        graph_xs,
        lagrange_ys,
        newton_ys,
        linear_ys,
        quadratic_ys,
    )

    lagrange_series = GraphSeries(
        name="Лагранж",
        color="#0b7285",
        points=[Point(x=x, y=y) for x, y in zip(graph_xs, lagrange_ys)],
    )
    newton_series = GraphSeries(
        name="Ньютон",
        color="#d9480f",
        points=[Point(x=x, y=y) for x, y in zip(graph_xs, newton_ys)],
        dasharray="8,5",
    )
    linear_series = GraphSeries(
        name="Линейный сплайн",
        color="#2f9e44",
        points=[Point(x=x, y=y) for x, y in zip(graph_xs, linear_ys)],
    )
    quadratic_series = GraphSeries(
        name="Квадратичный сплайн",
        color="#5f3dc4",
        points=[Point(x=x, y=y) for x, y in zip(graph_xs, quadratic_ys)],
        dasharray="5,4",
    )

    write_svg(results_dir / "lagrange.svg", "Многочлен Лагранжа", [lagrange_series], nodes)
    write_svg(results_dir / "newton.svg", "Полином Ньютона", [newton_series], nodes)
    write_svg(
        results_dir / "linear_spline.svg",
        "Линейный интерполяционный сплайн",
        [linear_series],
        nodes,
    )
    write_svg(
        results_dir / "quadratic_spline.svg",
        "Квадратичный интерполяционный сплайн",
        [quadratic_series],
        nodes,
    )
    write_svg(
        results_dir / "combined.svg",
        "Интерполяционные многочлены и сплайны",
        [lagrange_series, newton_series, linear_series, quadratic_series],
        nodes,
    )

    build_report(
        path=results_dir / "report.md",
        nodes=nodes,
        lagrange_coefficients=lagrange_coefficients,
        lagrange_basis_rows=lagrange_basis_rows,
        finite_table=finite_table,
        divided_table=divided_table,
        newton_coefficients=newton_coefficients,
        newton_power_coeffs=newton_power_coeffs,
        linear_segments=linear_segments,
        quadratic_segments=quadratic_segments,
        lagrange_at_query=lagrange_at_query,
        newton_at_query=newton_at_query,
    )
    write_json(
        path=results_dir / "results.json",
        nodes=nodes,
        lagrange_coefficients=lagrange_coefficients,
        lagrange_basis_rows=lagrange_basis_rows,
        finite_table=finite_table,
        divided_table=divided_table,
        newton_coefficients=newton_coefficients,
        newton_power_coeffs=newton_power_coeffs,
        linear_segments=linear_segments,
        quadratic_segments=quadratic_segments,
        lagrange_at_query=lagrange_at_query,
        newton_at_query=newton_at_query,
    )

    print(f"Lab 5 variant {VARIANT} solved.")
    print(f"Results directory: {results_dir}")
    print(f"x1 + x2 = {QUERY_X:.6f}")
    print(f"L4(x1+x2) = {lagrange_at_query:.12f}")
    print(f"N4(x1+x2) = {newton_at_query:.12f}")
    print(f"|L4 - N4| = {abs(lagrange_at_query - newton_at_query):.12e}")


if __name__ == "__main__":
    main()
