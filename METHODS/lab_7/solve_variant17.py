from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable, Sequence
from xml.sax.saxutils import escape


VARIANT = 17
A = 1.0
B = 4.0
N = 6


@dataclass
class Point:
    x: float
    y: float


@dataclass
class MethodResult:
    name: str
    short_name: str
    coefficients: list[float]
    weighted_sum: float
    value: float
    absolute_error: float
    relative_error_percent: float


@dataclass
class GraphSeries:
    name: str
    color: str
    points: list[Point]
    width: float = 2.6
    dasharray: str | None = None


def f(x: float) -> float:
    return 7.0 * math.sqrt(x) + 2.0 * x**2


def antiderivative(x: float) -> float:
    return (14.0 / 3.0) * x ** 1.5 + (2.0 / 3.0) * x**3


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


def step() -> float:
    return (B - A) / N


def nodes() -> list[float]:
    h = step()
    return [A + i * h for i in range(N + 1)]


def function_values(xs: Sequence[float]) -> list[float]:
    return [f(x) for x in xs]


def exact_integral() -> float:
    return antiderivative(B) - antiderivative(A)


def left_rectangles_coefficients() -> list[float]:
    return [1.0] * N + [0.0]


def right_rectangles_coefficients() -> list[float]:
    return [0.0] + [1.0] * N


def trapezoid_coefficients() -> list[float]:
    return [0.5] + [1.0] * (N - 1) + [0.5]


def simpson_coefficients() -> list[float]:
    if N % 2 != 0:
        raise ValueError("The Simpson formula requires an even number of intervals.")
    coeffs = [1.0 / 3.0]
    for i in range(1, N):
        coeffs.append(4.0 / 3.0 if i % 2 == 1 else 2.0 / 3.0)
    coeffs.append(1.0 / 3.0)
    return coeffs


def method_result(name: str, short_name: str, coeffs: list[float], ys: Sequence[float], exact: float) -> MethodResult:
    weighted_sum = sum(c * y for c, y in zip(coeffs, ys))
    value = step() * weighted_sum
    absolute_error = abs(value - exact)
    relative_error_percent = absolute_error / abs(exact) * 100.0
    return MethodResult(
        name=name,
        short_name=short_name,
        coefficients=coeffs,
        weighted_sum=weighted_sum,
        value=value,
        absolute_error=absolute_error,
        relative_error_percent=relative_error_percent,
    )


def calculate_methods() -> list[MethodResult]:
    xs = nodes()
    ys = function_values(xs)
    exact = exact_integral()
    return [
        method_result("Левые прямоугольники", "Левые", left_rectangles_coefficients(), ys, exact),
        method_result("Правые прямоугольники", "Правые", right_rectangles_coefficients(), ys, exact),
        method_result("Трапеции", "Трапеции", trapezoid_coefficients(), ys, exact),
        method_result("Параболы Симпсона", "Симпсон", simpson_coefficients(), ys, exact),
    ]


def coefficient_text(value: float) -> str:
    if abs(value - 1.0 / 3.0) < 1e-12:
        return "1/3"
    if abs(value - 2.0 / 3.0) < 1e-12:
        return "2/3"
    if abs(value - 4.0 / 3.0) < 1e-12:
        return "4/3"
    return _fmt_ru_compact(value, 6)


def nodes_table_markdown(xs: Sequence[float], ys: Sequence[float], methods: Sequence[MethodResult]) -> str:
    lines = [
        "| i | xᵢ | f(xᵢ) | cᵢ лев. | cᵢ прав. | cᵢ трап. | cᵢ Симп. |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, (x, y) in enumerate(zip(xs, ys)):
        coeffs = [method.coefficients[i] for method in methods]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    _fmt_ru(x, 3),
                    _fmt_ru(y, 6),
                    *(coefficient_text(c) for c in coeffs),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def comparison_table_markdown(methods: Sequence[MethodResult], exact: float) -> str:
    lines = [
        "| способ | значение интеграла | абсолютная ошибка | относительная ошибка |",
        "|:---|---:|---:|---:|",
    ]
    for method in methods:
        lines.append(
            f"| {method.name} | {_fmt_ru(method.value, 9)} | "
            f"{_fmt_ru(method.absolute_error, 9)} | {_fmt_ru(method.relative_error_percent, 6)}% |"
        )
    lines.append(f"| Ньютон–Лейбниц | {_fmt_ru(exact, 9)} | 0,000000000 | 0,000000% |")
    return "\n".join(lines)


def method_formula_block(method: MethodResult) -> list[str]:
    h_text = _fmt_ru(step(), 3)
    weighted_text = _fmt_ru(method.weighted_sum, 9)
    value_text = _fmt_ru(method.value, 9)
    if method.short_name == "Левые":
        formula = "Iₗ = h·(f₀ + f₁ + ... + fₙ₋₁)"
    elif method.short_name == "Правые":
        formula = "Iᵣ = h·(f₁ + f₂ + ... + fₙ)"
    elif method.short_name == "Трапеции":
        formula = "Iₜ = h·(0,5·f₀ + f₁ + ... + fₙ₋₁ + 0,5·fₙ)"
    else:
        formula = "Iₛ = h/3·(f₀ + 4f₁ + 2f₂ + 4f₃ + ... + fₙ)"
    return [
        f"`{formula}`",
        "",
        f"`I = {h_text}·{weighted_text} = {value_text}`",
    ]


def build_report(methods: Sequence[MethodResult]) -> str:
    xs = nodes()
    ys = function_values(xs)
    h = step()
    exact = exact_integral()
    best_method = min(methods, key=lambda method: method.absolute_error)

    lines: list[str] = []
    lines.append(f"# Лабораторная работа №7, вариант {VARIANT}")
    lines.append("")
    lines.append("## Задание")
    lines.append("")
    lines.append("Вычислить интеграл квадратурными формулами прямоугольников, трапеций и парабол Симпсона при заданном числе интервалов. Дополнительно вычислить точное значение по формуле Ньютона–Лейбница и сравнить результаты.")
    lines.append("")
    lines.append("Для варианта 17:")
    lines.append("")
    lines.append("`I = ∫₁⁴ (7√x + 2x²) dx`, `n = 6`.")
    lines.append("")
    lines.append("Шаг разбиения:")
    lines.append("")
    lines.append(f"`h = (b − a) / n = (4 − 1) / 6 = {_fmt_ru(h, 3)}`")
    lines.append("")
    lines.append("## 1) Значения функции в узлах")
    lines.append("")
    lines.append("Узлы берутся по формуле `xᵢ = a + ih`. Для каждого узла вычисляем `f(xᵢ) = 7√xᵢ + 2xᵢ²`.")
    lines.append("")
    lines.append(nodes_table_markdown(xs, ys, methods))
    lines.append("")
    lines.append("Удобно использовать общую квадратурную запись:")
    lines.append("")
    lines.append("`I ≈ h·Σcᵢf(xᵢ)`")
    lines.append("")
    lines.append("где `cᵢ` — коэффициенты выбранной формулы. Они уже внесены в таблицу выше.")
    lines.append("")
    lines.append("![function](function.svg)")
    lines.append("")

    lines.append("## 2) Метод прямоугольников")
    lines.append("")
    lines.append("Для прямоугольников считаем две стандартные формулы: по левым и по правым концам отрезков.")
    lines.append("")
    for method in methods[:2]:
        lines.append(f"**{method.name}:**")
        lines.append("")
        lines.extend(method_formula_block(method))
        lines.append("")
    lines.append("На рисунках видно, что функция на каждом малом отрезке заменяется прямоугольником. У левых прямоугольников высота берется в левом конце отрезка, у правых — в правом.")
    lines.append("")
    lines.append("![left rectangles](rectangles_left.svg)")
    lines.append("")
    lines.append("![right rectangles](rectangles_right.svg)")
    lines.append("")

    lines.append("## 3) Метод трапеций")
    lines.append("")
    lines.extend(method_formula_block(methods[2]))
    lines.append("")
    lines.append("Здесь на каждом отрезке кривая заменяется прямой между соседними узлами, поэтому площадь считается как сумма трапеций.")
    lines.append("")
    lines.append("![trapezoids](trapezoids.svg)")
    lines.append("")

    lines.append("## 4) Метод парабол Симпсона")
    lines.append("")
    lines.append("Так как `n = 6` — четное число, формулу Симпсона можно применять.")
    lines.append("")
    lines.extend(method_formula_block(methods[3]))
    lines.append("")
    lines.append("В методе Симпсона каждые два соседних интервала объединяются, а функция на них заменяется параболой, проходящей через три узла.")
    lines.append("")
    lines.append("![simpson](simpson.svg)")
    lines.append("")

    lines.append("## 5) Формула Ньютона–Лейбница")
    lines.append("")
    lines.append("Найдем первообразную:")
    lines.append("")
    lines.append("`F(x) = ∫(7√x + 2x²) dx = 14/3·x^(3/2) + 2/3·x³`")
    lines.append("")
    lines.append("Тогда:")
    lines.append("")
    lines.append("`I = F(4) − F(1)`")
    lines.append("")
    lines.append("`F(4) = 14/3·4^(3/2) + 2/3·4³ = 80`")
    lines.append("")
    lines.append("`F(1) = 14/3·1^(3/2) + 2/3·1³ = 16/3`")
    lines.append("")
    lines.append(f"`I = 80 − 16/3 = 224/3 = {_fmt_ru(exact, 9)}`")
    lines.append("")

    lines.append("## 6) Сравнение результатов")
    lines.append("")
    lines.append(comparison_table_markdown(methods, exact))
    lines.append("")
    lines.append("![comparison](comparison.svg)")
    lines.append("")
    lines.append(f"Наиболее точный результат при данном числе интервалов дал метод **{best_method.name}**: его абсолютная ошибка равна `{_fmt_ru(best_method.absolute_error, 9)}`.")
    return "\n".join(lines) + "\n"


def write_nodes_csv(path: Path, xs: Sequence[float], ys: Sequence[float], methods: Sequence[MethodResult]) -> None:
    header = [
        "i",
        "x",
        "f_x",
        "left_c",
        "right_c",
        "trapezoid_c",
        "simpson_c",
        "left_c_f",
        "right_c_f",
        "trapezoid_c_f",
        "simpson_c_f",
    ]
    rows = [",".join(header)]
    for i, (x, y) in enumerate(zip(xs, ys)):
        coeffs = [method.coefficients[i] for method in methods]
        products = [c * y for c in coeffs]
        values = [i, x, y, *coeffs, *products]
        rows.append(",".join(str(value) for value in values))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_results_json(path: Path, methods: Sequence[MethodResult], exact: float) -> None:
    data = {
        "variant": VARIANT,
        "integrand": "7*sqrt(x) + 2*x^2",
        "a": A,
        "b": B,
        "n": N,
        "h": step(),
        "exact_newton_leibniz": exact,
        "methods": [asdict(method) for method in methods],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _tick_step(v_min: float, v_max: float, target_ticks: int) -> float:
    span = v_max - v_min
    if span <= 0.0:
        return 1.0
    raw_step = span / max(target_ticks, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        step_value = factor * magnitude
        if step_value >= raw_step:
            return step_value
    return 10.0 * magnitude


def _build_ticks(v_min: float, v_max: float, target_ticks: int = 8) -> list[float]:
    step_value = _tick_step(v_min, v_max, target_ticks)
    start = math.floor(v_min / step_value) * step_value
    end = math.ceil(v_max / step_value) * step_value
    ticks: list[float] = []
    value = start
    while value <= end + 1e-12:
        ticks.append(round(value, 10))
        value += step_value
    return ticks


def sample_points(func: Callable[[float], float], left: float, right: float, count: int = 450) -> list[Point]:
    return [Point(left + (right - left) * i / (count - 1), func(left + (right - left) * i / (count - 1))) for i in range(count)]


def write_function_svg(path: Path, title: str, series: Sequence[GraphSeries], nodes_to_draw: Sequence[Point]) -> None:
    width = 980
    height = 640
    margin_left = 86
    margin_right = 36
    margin_top = 70
    margin_bottom = 74
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_points = [point for item in series for point in item.points] + list(nodes_to_draw)
    x_min = min(point.x for point in all_points)
    x_max = max(point.x for point in all_points)
    y_min = min(point.y for point in all_points)
    y_max = max(point.y for point in all_points)
    y_padding = (y_max - y_min) * 0.1
    y_min -= y_padding
    y_max += y_padding

    def sx(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_width

    def sy(y: float) -> float:
        return margin_top + (y_max - y) / (y_max - y_min) * plot_height

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="24" font-family="Arial" fill="#1f2933">{escape(title)}</text>',
    ]

    for tick in _build_ticks(x_min, x_max, 7):
        x = sx(tick)
        svg.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="#e6edf3"/>')
        svg.append(f'<text x="{x:.2f}" y="{height - margin_bottom + 22}" text-anchor="middle" font-size="12" font-family="Arial" fill="#415166">{_fmt_ru_compact(tick, 2)}</text>')
    for tick in _build_ticks(y_min, y_max, 8):
        y = sy(tick)
        svg.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e6edf3"/>')
        svg.append(f'<text x="{margin_left - 14}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#415166">{_fmt_ru_compact(tick, 2)}</text>')

    svg.append(f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#5c6b7a" stroke-width="1.4"/>')
    svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#5c6b7a" stroke-width="1.4"/>')
    svg.append(f'<text x="{width - margin_right + 10}" y="{height - margin_bottom + 5}" font-size="16" font-family="Arial" fill="#1f2933">x</text>')
    svg.append(f'<text x="{margin_left + 8}" y="{margin_top - 16}" font-size="16" font-family="Arial" fill="#1f2933">y</text>')

    legend_y = margin_top + 16
    for idx, item in enumerate(series):
        legend_x = margin_left + 12 + idx * 150
        svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{item.color}" stroke-width="{item.width}" stroke-linecap="round"/>')
        svg.append(f'<text x="{legend_x + 38}" y="{legend_y + 4}" font-size="12" font-family="Arial" fill="#1f2933">{escape(item.name)}</text>')

    for item in series:
        points = " ".join(f"{sx(point.x):.2f},{sy(point.y):.2f}" for point in item.points)
        dash = f' stroke-dasharray="{item.dasharray}"' if item.dasharray else ""
        svg.append(f'<polyline points="{points}" fill="none" stroke="{item.color}" stroke-width="{item.width}" stroke-linecap="round" stroke-linejoin="round"{dash}/>')

    for index, point in enumerate(nodes_to_draw):
        x = sx(point.x)
        y = sy(point.y)
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#111827"/>')
        svg.append(f'<text x="{x + 7:.2f}" y="{y - 7:.2f}" font-size="12" font-family="Arial" fill="#111827">x{index}</text>')

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def simpson_quadratic_value(x: float, x0: float, y0: float, x1: float, y1: float, x2: float, y2: float) -> float:
    l0 = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))
    l1 = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))
    l2 = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))
    return y0 * l0 + y1 * l1 + y2 * l2


def write_quadrature_svg(path: Path, title: str, kind: str, method: MethodResult) -> None:
    width = 980
    height = 600
    margin_left = 86
    margin_right = 38
    margin_top = 74
    margin_bottom = 82
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    xs = nodes()
    ys = function_values(xs)
    x_min = A
    x_max = B
    y_min = 0.0
    y_max = max(ys) * 1.12

    def sx(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_width

    def sy(y: float) -> float:
        return margin_top + (y_max - y) / (y_max - y_min) * plot_height

    style = {
        "left": ("#dbeafe", "#2563eb"),
        "right": ("#e0f2fe", "#0284c7"),
        "trapezoid": ("#dcfce7", "#16a34a"),
        "simpson": ("#ede9fe", "#7c3aed"),
    }[kind]
    fill_color, stroke_color = style
    baseline = sy(0.0)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="24" font-family="Arial" fill="#1f2933">{escape(title)}</text>',
        f'<text x="{width / 2}" y="58" text-anchor="middle" font-size="14" font-family="Arial" fill="#415166">I ≈ {_fmt_ru(method.value, 9)}</text>',
    ]

    for tick in _build_ticks(x_min, x_max, 7):
        x = sx(tick)
        svg.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="#e6edf3"/>')
        svg.append(f'<text x="{x:.2f}" y="{height - margin_bottom + 22}" text-anchor="middle" font-size="12" font-family="Arial" fill="#415166">{_fmt_ru_compact(tick, 2)}</text>')
    for tick in _build_ticks(y_min, y_max, 8):
        y = sy(tick)
        svg.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e6edf3"/>')
        svg.append(f'<text x="{margin_left - 14}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#415166">{_fmt_ru_compact(tick, 2)}</text>')

    if kind in {"left", "right"}:
        for i in range(N):
            x_left = xs[i]
            x_right = xs[i + 1]
            height_value = ys[i] if kind == "left" else ys[i + 1]
            rect_x = sx(x_left)
            rect_y = sy(height_value)
            rect_width = sx(x_right) - sx(x_left)
            rect_height = baseline - rect_y
            svg.append(
                f'<rect x="{rect_x:.2f}" y="{rect_y:.2f}" width="{rect_width:.2f}" height="{rect_height:.2f}" '
                f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="1.4" fill-opacity="0.72"/>'
            )
    elif kind == "trapezoid":
        for i in range(N):
            x_left = xs[i]
            x_right = xs[i + 1]
            y_left = ys[i]
            y_right = ys[i + 1]
            points = [
                (sx(x_left), baseline),
                (sx(x_left), sy(y_left)),
                (sx(x_right), sy(y_right)),
                (sx(x_right), baseline),
            ]
            point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            svg.append(f'<polygon points="{point_text}" fill="{fill_color}" stroke="{stroke_color}" stroke-width="1.4" fill-opacity="0.72"/>')
    elif kind == "simpson":
        for i in range(0, N, 2):
            x0, x1, x2 = xs[i], xs[i + 1], xs[i + 2]
            y0, y1, y2 = ys[i], ys[i + 1], ys[i + 2]
            curve_points: list[tuple[float, float]] = []
            for sample_index in range(45):
                x = x0 + (x2 - x0) * sample_index / 44
                y = simpson_quadratic_value(x, x0, y0, x1, y1, x2, y2)
                curve_points.append((sx(x), sy(y)))
            polygon_points = [(sx(x0), baseline), *curve_points, (sx(x2), baseline)]
            polygon_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in polygon_points)
            curve_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in curve_points)
            svg.append(f'<polygon points="{polygon_text}" fill="{fill_color}" stroke="{stroke_color}" stroke-width="1.3" fill-opacity="0.68"/>')
            svg.append(f'<polyline points="{curve_text}" fill="none" stroke="{stroke_color}" stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round"/>')

    function_points = " ".join(f"{sx(point.x):.2f},{sy(point.y):.2f}" for point in sample_points(f, A, B, 500))
    svg.append(f'<polyline points="{function_points}" fill="none" stroke="#0f766e" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/>')

    for x in xs:
        svg.append(f'<line x1="{sx(x):.2f}" y1="{baseline:.2f}" x2="{sx(x):.2f}" y2="{baseline + 6:.2f}" stroke="#5c6b7a" stroke-width="1.1"/>')
    for index, (x, y) in enumerate(zip(xs, ys)):
        svg.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4.2" fill="#111827"/>')
        svg.append(f'<text x="{sx(x) + 6:.2f}" y="{sy(y) - 7:.2f}" font-size="11" font-family="Arial" fill="#111827">x{index}</text>')

    svg.append(f'<line x1="{margin_left}" y1="{baseline:.2f}" x2="{width - margin_right}" y2="{baseline:.2f}" stroke="#5c6b7a" stroke-width="1.4"/>')
    svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#5c6b7a" stroke-width="1.4"/>')
    svg.append(f'<text x="{width - margin_right + 10}" y="{baseline + 5:.2f}" font-size="16" font-family="Arial" fill="#1f2933">x</text>')
    svg.append(f'<text x="{margin_left + 8}" y="{margin_top - 16}" font-size="16" font-family="Arial" fill="#1f2933">y</text>')

    svg.append(f'<rect x="{margin_left + 16}" y="{margin_top + 14}" width="18" height="12" fill="{fill_color}" stroke="{stroke_color}" fill-opacity="0.72"/>')
    svg.append(f'<text x="{margin_left + 42}" y="{margin_top + 25}" font-size="12" font-family="Arial" fill="#1f2933">площадь, которую суммирует метод</text>')
    svg.append(f'<line x1="{margin_left + 16}" y1="{margin_top + 43}" x2="{margin_left + 34}" y2="{margin_top + 43}" stroke="#0f766e" stroke-width="2.8" stroke-linecap="round"/>')
    svg.append(f'<text x="{margin_left + 42}" y="{margin_top + 47}" font-size="12" font-family="Arial" fill="#1f2933">исходная функция</text>')

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def write_comparison_svg(path: Path, methods: Sequence[MethodResult], exact: float) -> None:
    width = 980
    height = 520
    margin_left = 86
    margin_right = 38
    margin_top = 70
    margin_bottom = 98
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    values = [method.value for method in methods] + [exact]
    labels = [method.short_name for method in methods] + ["Н-Л"]
    colors = ["#db6b4d", "#4d7cdb", "#30a46c", "#7b61ff", "#111827"]
    y_min = min(values) - 3.0
    y_max = max(values) + 3.0

    def sy(value: float) -> float:
        return margin_top + (y_max - value) / (y_max - y_min) * plot_height

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="24" font-family="Arial" fill="#1f2933">Сравнение результатов интегрирования</text>',
    ]

    for tick in _build_ticks(y_min, y_max, 7):
        y = sy(tick)
        svg.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e6edf3"/>')
        svg.append(f'<text x="{margin_left - 14}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#415166">{_fmt_ru_compact(tick, 2)}</text>')

    svg.append(f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#5c6b7a" stroke-width="1.4"/>')
    svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#5c6b7a" stroke-width="1.4"/>')

    group_width = plot_width / len(values)
    bar_width = min(92.0, group_width * 0.56)
    exact_y = sy(exact)
    svg.append(f'<line x1="{margin_left}" y1="{exact_y:.2f}" x2="{width - margin_right}" y2="{exact_y:.2f}" stroke="#111827" stroke-width="1.6" stroke-dasharray="7 5"/>')
    svg.append(f'<text x="{width - margin_right - 6}" y="{exact_y - 8:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#111827">точное</text>')

    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        center = margin_left + group_width * (i + 0.5)
        y = sy(value)
        bar_height = height - margin_bottom - y
        svg.append(f'<rect x="{center - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}" rx="4"/>')
        svg.append(f'<text x="{center:.2f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-size="13" font-family="Arial" fill="#1f2933">{escape(label)}</text>')
        svg.append(f'<text x="{center:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-size="12" font-family="Arial" fill="#1f2933">{_fmt_ru(value, 3)}</text>')

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    current_dir = Path(__file__).resolve().parent
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)

    xs = nodes()
    ys = function_values(xs)
    exact = exact_integral()
    methods = calculate_methods()

    (results_dir / "report.md").write_text(build_report(methods), encoding="utf-8")
    write_nodes_csv(results_dir / "nodes.csv", xs, ys, methods)
    write_results_json(results_dir / "results.json", methods, exact)

    function_series = [
        GraphSeries(
            name="f(x)=7√x+2x²",
            color="#087f95",
            points=sample_points(f, A, B),
        )
    ]
    node_points = [Point(x, y) for x, y in zip(xs, ys)]
    write_function_svg(results_dir / "function.svg", "График функции и узлы интегрирования", function_series, node_points)
    write_quadrature_svg(results_dir / "rectangles_left.svg", "Метод левых прямоугольников", "left", methods[0])
    write_quadrature_svg(results_dir / "rectangles_right.svg", "Метод правых прямоугольников", "right", methods[1])
    write_quadrature_svg(results_dir / "trapezoids.svg", "Метод трапеций", "trapezoid", methods[2])
    write_quadrature_svg(results_dir / "simpson.svg", "Метод Симпсона", "simpson", methods[3])
    write_comparison_svg(results_dir / "comparison.svg", methods, exact)

    print(f"Lab 7 variant {VARIANT} solved.")
    print(f"Results directory: {results_dir}")
    print(f"Exact integral: {exact:.12f}")
    for method in methods:
        print(f"{method.name}: value={method.value:.12f}, abs_error={method.absolute_error:.12f}")


if __name__ == "__main__":
    main()
