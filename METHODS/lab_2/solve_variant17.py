from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Iterable


VARIANT = 17
EQUATION_TEXT = "x^3 + 5x - 1 = 0"
EPS_MAIN = 1e-2
EPS_EXTRA = 1e-6


def f(x: float) -> float:
    return x**3 + 5.0 * x - 1.0


def df(x: float) -> float:
    return 3.0 * x**2 + 5.0


@dataclass
class IterationRow:
    n: int
    x: float
    fx: float
    delta: float


def isolate_root_intervals(
    func: Callable[[float], float],
    x_min: float,
    x_max: float,
    step: float,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    x_left = x_min
    f_left = func(x_left)

    while x_left + step <= x_max + 1e-12:
        x_right = round(x_left + step, 10)
        f_right = func(x_right)

        if f_left == 0.0:
            intervals.append((x_left, x_left))
        elif f_left * f_right < 0.0:
            intervals.append((x_left, x_right))

        x_left = x_right
        f_left = f_right

    return intervals


def newton_method(
    func: Callable[[float], float],
    dfunc: Callable[[float], float],
    x0: float,
    eps: float,
    max_iter: int = 100,
) -> tuple[float, list[IterationRow]]:
    rows: list[IterationRow] = []
    x_curr = x0

    for n in range(1, max_iter + 1):
        derivative = dfunc(x_curr)
        if derivative == 0.0:
            raise ZeroDivisionError("Newton method failed: derivative is zero.")

        x_next = x_curr - func(x_curr) / derivative
        delta = abs(x_next - x_curr)
        fx_next = func(x_next)
        rows.append(IterationRow(n=n, x=x_next, fx=fx_next, delta=delta))

        if delta < eps:
            return x_next, rows

        x_curr = x_next

    raise RuntimeError("Newton method did not converge within max_iter.")


def secant_method(
    func: Callable[[float], float],
    x0: float,
    x1: float,
    eps: float,
    max_iter: int = 100,
) -> tuple[float, list[IterationRow]]:
    rows: list[IterationRow] = []
    x_prev = x0
    x_curr = x1

    for n in range(1, max_iter + 1):
        f_prev = func(x_prev)
        f_curr = func(x_curr)
        denom = f_curr - f_prev
        if denom == 0.0:
            raise ZeroDivisionError("Secant method failed: zero denominator.")

        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        delta = abs(x_next - x_curr)
        fx_next = func(x_next)
        rows.append(IterationRow(n=n, x=x_next, fx=fx_next, delta=delta))

        if delta < eps:
            return x_next, rows

        x_prev, x_curr = x_curr, x_next

    raise RuntimeError("Secant method did not converge within max_iter.")


def steffensen_method(
    func: Callable[[float], float],
    x0: float,
    eps: float,
    max_iter: int = 100,
) -> tuple[float, list[IterationRow]]:
    rows: list[IterationRow] = []
    x_curr = x0

    for n in range(1, max_iter + 1):
        fx = func(x_curr)
        denom = func(x_curr + fx) - fx
        if denom == 0.0:
            raise ZeroDivisionError("Steffensen method failed: zero denominator.")

        x_next = x_curr - (fx * fx) / denom
        delta = abs(x_next - x_curr)
        fx_next = func(x_next)
        rows.append(IterationRow(n=n, x=x_next, fx=fx_next, delta=delta))

        if delta < eps:
            return x_next, rows

        x_curr = x_next

    raise RuntimeError("Steffensen method did not converge within max_iter.")


def sample_points(
    func: Callable[[float], float],
    x_min: float,
    x_max: float,
    step: float,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    x = x_min
    while x <= x_max + 1e-12:
        xr = round(x, 10)
        points.append((xr, func(xr)))
        x += step
    return points


def write_csv(path: Path, rows: Iterable[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write("x,f(x)\n")
        for x, fx in rows:
            file.write(f"{x:.6f},{fx:.12f}\n")


def _tick_step(v_min: float, v_max: float, target_ticks: int) -> float:
    span = v_max - v_min
    if span <= 0:
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


def _tick_label(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.6g}"


def write_svg(
    path: Path,
    points: list[tuple[float, float]],
    root_x: float | None = None,
    newton_xs: list[float] | None = None,
) -> None:
    width, height = 980, 640
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 70, 80
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    y_pad = 0.12 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    # Keep zero visible on y-axis labels even if data range is shifted.
    y_min = min(y_min, -0.5)
    y_max = max(y_max, 0.5)

    def sx(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * (
            width - margin_left - margin_right
        )

    def sy(y: float) -> float:
        return height - margin_bottom - (y - y_min) / (y_max - y_min) * (
            height - margin_top - margin_bottom
        )

    plot_left, plot_right = margin_left, width - margin_right
    plot_top, plot_bottom = margin_top, height - margin_bottom

    polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
    x_axis = sy(0.0)
    y_axis = sx(0.0)
    x_ticks = _build_ticks(x_min, x_max, target_ticks=8)
    y_ticks = _build_ticks(y_min, y_max, target_ticks=8)

    x_label_y = x_axis + 18 if plot_top <= x_axis <= plot_bottom else plot_bottom + 22
    y_label_x = y_axis - 10 if plot_left <= y_axis <= plot_right else plot_left - 10

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
    ]

    # Grid + ticks.
    for xt in x_ticks:
        px = sx(xt)
        if px < plot_left - 1e-9 or px > plot_right + 1e-9:
            continue
        parts.append(
            f'  <line x1="{px:.2f}" y1="{plot_top:.2f}" x2="{px:.2f}" y2="{plot_bottom:.2f}" stroke="#ececec" stroke-width="1"/>'
        )
        parts.append(
            f'  <line x1="{px:.2f}" y1="{x_axis - 5:.2f}" x2="{px:.2f}" y2="{x_axis + 5:.2f}" stroke="#666" stroke-width="1"/>'
        )
        parts.append(
            f'  <text x="{px:.2f}" y="{x_label_y:.2f}" text-anchor="middle" font-size="12" font-family="Arial" fill="#444">{_tick_label(xt)}</text>'
        )

    for yt in y_ticks:
        py = sy(yt)
        if py < plot_top - 1e-9 or py > plot_bottom + 1e-9:
            continue
        parts.append(
            f'  <line x1="{plot_left:.2f}" y1="{py:.2f}" x2="{plot_right:.2f}" y2="{py:.2f}" stroke="#ececec" stroke-width="1"/>'
        )
        parts.append(
            f'  <line x1="{y_axis - 5:.2f}" y1="{py:.2f}" x2="{y_axis + 5:.2f}" y2="{py:.2f}" stroke="#666" stroke-width="1"/>'
        )
        parts.append(
            f'  <text x="{y_label_x:.2f}" y="{py + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#444">{_tick_label(yt)}</text>'
        )

    # Axes.
    parts.append(
        f'  <line x1="{plot_left:.2f}" y1="{x_axis:.2f}" x2="{plot_right:.2f}" y2="{x_axis:.2f}" stroke="#666" stroke-width="1.4"/>'
    )
    parts.append(
        f'  <line x1="{y_axis:.2f}" y1="{plot_top:.2f}" x2="{y_axis:.2f}" y2="{plot_bottom:.2f}" stroke="#666" stroke-width="1.4"/>'
    )

    # Newton tangents.
    if newton_xs and len(newton_xs) > 1:
        tangent_colors = ["#d9480f", "#2b8a3e", "#5f3dc4", "#0b7285"]
        for i, (xn, x_next) in enumerate(zip(newton_xs[:-1], newton_xs[1:])):
            color = tangent_colors[i % len(tangent_colors)]
            y_at_xn = f(xn)
            slope = df(xn)
            y_left = y_at_xn + slope * (x_min - xn)
            y_right = y_at_xn + slope * (x_max - xn)
            parts.append(
                f'  <line x1="{sx(x_min):.2f}" y1="{sy(y_left):.2f}" x2="{sx(x_max):.2f}" y2="{sy(y_right):.2f}" stroke="{color}" stroke-width="1.8" stroke-dasharray="7,5" opacity="0.9"/>'
            )
            parts.append(
                f'  <circle cx="{sx(xn):.2f}" cy="{sy(y_at_xn):.2f}" r="4" fill="{color}"/>'
            )
            parts.append(
                f'  <circle cx="{sx(x_next):.2f}" cy="{sy(0.0):.2f}" r="3.5" fill="{color}"/>'
            )
            parts.append(
                f'  <line x1="{sx(xn):.2f}" y1="{sy(0.0):.2f}" x2="{sx(xn):.2f}" y2="{sy(y_at_xn):.2f}" stroke="{color}" stroke-width="1" stroke-dasharray="3,3" opacity="0.8"/>'
            )
            parts.append(
                f'  <text x="{sx(x_next):.2f}" y="{x_axis + 34:.2f}" text-anchor="middle" font-size="12" font-family="Arial" fill="{color}">x{i+1}</text>'
            )
        # Mark initial point on x-axis for readability.
        parts.append(
            f'  <circle cx="{sx(newton_xs[0]):.2f}" cy="{sy(0.0):.2f}" r="3.5" fill="#d9480f"/>'
        )
        parts.append(
            f'  <text x="{sx(newton_xs[0]):.2f}" y="{x_axis + 34:.2f}" text-anchor="middle" font-size="12" font-family="Arial" fill="#d9480f">x0</text>'
        )

    # Function graph + root marker.
    parts.append(
        f'  <polyline points="{polyline}" fill="none" stroke="#005bbb" stroke-width="2.8"/>'
    )
    if root_x is not None:
        parts.append(
            f'  <circle cx="{sx(root_x):.2f}" cy="{sy(0.0):.2f}" r="5.2" fill="#c92a2a"/>'
        )
        parts.append(
            f'  <text x="{sx(root_x) + 8:.2f}" y="{sy(0.0) - 10:.2f}" font-size="12" font-family="Arial" fill="#c92a2a">x*≈{root_x:.4f}</text>'
        )

    # Titles and legend.
    parts.append(
        f'  <text x="{width / 2:.0f}" y="34" text-anchor="middle" font-size="32" font-family="Arial">f(x) = x^3 + 5x - 1</text>'
    )
    parts.append(
        f'  <text x="{plot_right + 16:.2f}" y="{x_axis + 5:.2f}" font-size="18" font-family="Arial">x</text>'
    )
    parts.append(
        f'  <text x="{y_axis + 10:.2f}" y="{plot_top - 16:.2f}" font-size="18" font-family="Arial">f(x)</text>'
    )
    legend_x = plot_left + 8
    legend_y = plot_top + 12
    parts.append(
        f'  <line x1="{legend_x:.2f}" y1="{legend_y:.2f}" x2="{legend_x + 26:.2f}" y2="{legend_y:.2f}" stroke="#005bbb" stroke-width="2.8"/>'
    )
    parts.append(
        f'  <text x="{legend_x + 34:.2f}" y="{legend_y + 4:.2f}" font-size="12" font-family="Arial" fill="#222">график f(x)</text>'
    )
    parts.append(
        f'  <line x1="{legend_x:.2f}" y1="{legend_y + 20:.2f}" x2="{legend_x + 26:.2f}" y2="{legend_y + 20:.2f}" stroke="#d9480f" stroke-width="1.8" stroke-dasharray="7,5"/>'
    )
    parts.append(
        f'  <text x="{legend_x + 34:.2f}" y="{legend_y + 24:.2f}" font-size="12" font-family="Arial" fill="#222">касательные Ньютона</text>'
    )
    parts.append(
        f'  <circle cx="{legend_x + 13:.2f}" cy="{legend_y + 40:.2f}" r="4.5" fill="#c92a2a"/>'
    )
    parts.append(
        f'  <text x="{legend_x + 34:.2f}" y="{legend_y + 44:.2f}" font-size="12" font-family="Arial" fill="#222">приближенный корень</text>'
    )
    parts.append("</svg>")

    svg = "\n".join(parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def _fmt(v: float) -> str:
    return f"{v:.10f}"


def rows_to_markdown(rows: list[IterationRow]) -> str:
    lines = [
        "| n | x_n | f(x_n) | abs(x_n - x_(n-1)) |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.n} | {_fmt(row.x)} | {_fmt(row.fx)} | {_fmt(row.delta)} |"
        )
    return "\n".join(lines)


def build_report(
    path: Path,
    isolation_rows: list[tuple[float, float]],
    intervals: list[tuple[float, float]],
    newton_root: float,
    newton_rows: list[IterationRow],
    secant_root: float,
    secant_rows: list[IterationRow],
    steff_root: float,
    steff_rows: list[IterationRow],
    newton_x0: float,
    secant_x0: float,
    secant_x1: float,
    steff_x0: float,
) -> None:
    graph_table = [
        "| x | f(x) |",
        "|---:|---:|",
    ]
    for x, fx in isolation_rows:
        graph_table.append(f"| {_fmt(x)} | {_fmt(fx)} |")

    interval_text = ", ".join(f"[{a:.1f}; {b:.1f}]" for a, b in intervals) or "none"

    report = f"""# Лабораторная работа №2, вариант {VARIANT}

Уравнение: `{EQUATION_TEXT}`

## Что сделано

1. Выполнена графическая изоляция корня через табулирование функции и поиск смены знака.
2. Для основной части использован один метод (метод Ньютона) при точности `eps=0.01`.
3. Для дополнительной части использованы два метода (секущих и Стеффенсена) при точности `eps=1e-6`.
4. Построен график с подписанными осями, делениями и касательными Ньютона.

## 1) Графическая изоляция корня

Интервалы смены знака функции при табулировании с шагом 0.1:

`{interval_text}`

График функции (с осями, делениями и касательными Ньютона):

![graph](graph.svg)

Таблица значений в окрестности пересечения оси `Ox`:

{chr(10).join(graph_table)}

Так как `f'(x)=3x^2+5>0` для всех `x`, функция строго возрастает, значит действительный корень единственный.

## 2) Основное задание (один метод, eps = {EPS_MAIN})

Выбранный метод: метод Ньютона.

Начальное приближение: `x0 = {newton_x0}`

Приближенный корень:

`x ≈ {newton_root:.10f}` (`|Δx| < 0.01`)

Итерации:

{rows_to_markdown(newton_rows)}

## 3) Дополнительное задание (eps = {EPS_EXTRA})

Выбраны два метода:

1. Метод секущих
2. Метод Стеффенсена

Метод секущих: `x0 = {secant_x0}`, `x1 = {secant_x1}`

Результат: `x ≈ {secant_root:.10f}`

{rows_to_markdown(secant_rows)}

Метод Стеффенсена: `x0 = {steff_x0}`

Результат: `x ≈ {steff_root:.10f}`

{rows_to_markdown(steff_rows)}

## 4) Как работают использованные методы

### Метод Ньютона (касательных)

Итерационная формула:

`x_(n+1) = x_n - f(x_n)/f'(x_n)`

Идея: в точке `x_n` строится касательная к графику `y=f(x)`, а ее пересечение с осью `Ox` берется как следующее приближение.

Почему метод сошелся в этой задаче:

- `f'(x)=3x^2+5` не обращается в ноль;
- стартовое приближение близко к корню (`x0=0.1`, корень в `[0.1; 0.2]`);
- функция гладкая и монотонная.

### Метод секущих

Итерационная формула:

`x_(n+1) = x_n - f(x_n)*(x_n - x_(n-1)) / (f(x_n)-f(x_(n-1)))`

Идея: вместо производной используется наклон секущей через две последние точки графика.
Это удобно, когда не хочется считать производную явно.

### Метод Стеффенсена

Итерационная формула:

`x_(n+1) = x_n - f(x_n)^2 / ( f(x_n + f(x_n)) - f(x_n) )`

Идея: ускорение итерационного процесса без прямого вычисления производной. На практике часто дает быструю сходимость рядом с корнем.

## 5) Вывод

Все использованные методы сошлись к одному корню:

`x* ≈ 0.1984372145`

Основное требование (`eps=0.01`) выполнено методом Ньютона, дополнительное (`eps=1e-6`) выполнено методами секущих и Стеффенсена.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    intervals = isolate_root_intervals(f, -3.0, 3.0, 0.1)
    if not intervals:
        raise RuntimeError("No sign-change intervals found.")
    a, b = intervals[0]

    newton_x0 = a
    secant_x0, secant_x1 = a, b
    steff_x0 = a

    # Main task: one method only (Newton), eps=0.01.
    newton_root, newton_rows = newton_method(f, df, x0=newton_x0, eps=EPS_MAIN)

    # Additional task: two methods with eps=1e-6.
    secant_root, secant_rows = secant_method(
        f, x0=secant_x0, x1=secant_x1, eps=EPS_EXTRA
    )
    steff_root, steff_rows = steffensen_method(f, x0=steff_x0, eps=EPS_EXTRA)

    graph_points = sample_points(f, -2.0, 2.0, 0.05)
    isolation_points = sample_points(f, 0.0, 0.3, 0.05)
    write_csv(results_dir / "graph_points.csv", graph_points)
    newton_sequence = [newton_x0] + [row.x for row in newton_rows]
    write_svg(
        results_dir / "graph.svg",
        graph_points,
        root_x=newton_root,
        newton_xs=newton_sequence,
    )

    build_report(
        path=results_dir / "report.md",
        isolation_rows=isolation_points,
        intervals=intervals,
        newton_root=newton_root,
        newton_rows=newton_rows,
        secant_root=secant_root,
        secant_rows=secant_rows,
        steff_root=steff_root,
        steff_rows=steff_rows,
        newton_x0=newton_x0,
        secant_x0=secant_x0,
        secant_x1=secant_x1,
        steff_x0=steff_x0,
    )

    print(f"Variant {VARIANT} solved.")
    print(f"Main (Newton, eps={EPS_MAIN}): x = {newton_root:.10f}")
    print(f"Extra (Secant, eps={EPS_EXTRA}): x = {secant_root:.10f}")
    print(f"Extra (Steffensen, eps={EPS_EXTRA}): x = {steff_root:.10f}")
    print(f"Results directory: {results_dir}")


if __name__ == "__main__":
    main()
