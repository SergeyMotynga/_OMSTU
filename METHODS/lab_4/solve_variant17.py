from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable


VARIANT = 17
EPS = 1e-3
MAX_ITER = 200

# Variant 17 (from lab sheet):
# sin(x + 2) - y = 1.5
# x + cos(y - 2) = -0.5


def f1(x: float, y: float) -> float:
    return math.sin(x + 2.0) - y - 1.5


def f2(x: float, y: float) -> float:
    return x + math.cos(y - 2.0) + 0.5


def phi_x(_: float, y: float) -> float:
    return -0.5 - math.cos(y - 2.0)


def phi_y(x: float, _: float) -> float:
    return math.sin(x + 2.0) - 1.5


def jacobian(x: float, y: float) -> tuple[float, float, float, float]:
    # [ df1/dx  df1/dy ]
    # [ df2/dx  df2/dy ]
    return math.cos(x + 2.0), -1.0, 1.0, -math.sin(y - 2.0)


def residual_inf(x: float, y: float) -> float:
    return max(abs(f1(x, y)), abs(f2(x, y)))


def objective_phi(x: float, y: float) -> float:
    r1 = f1(x, y)
    r2 = f2(x, y)
    return r1 * r1 + r2 * r2


def grad_phi(x: float, y: float) -> tuple[float, float]:
    r1 = f1(x, y)
    r2 = f2(x, y)
    gx = 2.0 * (r1 * math.cos(x + 2.0) + r2)
    gy = 2.0 * (-r1 - r2 * math.sin(y - 2.0))
    return gx, gy


def solve_2x2(
    a11: float, a12: float, a21: float, a22: float, b1: float, b2: float
) -> tuple[float, float]:
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-14:
        raise ZeroDivisionError("Degenerate 2x2 system (determinant is near zero).")
    x = (b1 * a22 - a12 * b2) / det
    y = (a11 * b2 - b1 * a21) / det
    return x, y


@dataclass
class NewtonRow:
    k: int
    x: float
    y: float
    dx: float
    dy: float
    delta_inf: float
    residual_inf: float
    f1: float
    f2: float


@dataclass
class IterRow:
    k: int
    x: float
    y: float
    delta_inf: float
    residual_inf: float
    f1: float
    f2: float


@dataclass
class DescentRow:
    k: int
    x: float
    y: float
    alpha: float
    grad_inf: float
    delta_inf: float
    residual_inf: float
    phi: float


def newton_method(
    x0: float, y0: float, eps: float, max_iter: int = MAX_ITER
) -> tuple[float, float, list[NewtonRow]]:
    x, y = x0, y0
    rows: list[NewtonRow] = []

    for k in range(1, max_iter + 1):
        r1 = f1(x, y)
        r2 = f2(x, y)
        a11, a12, a21, a22 = jacobian(x, y)
        dx, dy = solve_2x2(a11, a12, a21, a22, -r1, -r2)
        xn = x + dx
        yn = y + dy
        delta = max(abs(dx), abs(dy))
        res = residual_inf(xn, yn)
        rows.append(
            NewtonRow(
                k=k,
                x=xn,
                y=yn,
                dx=dx,
                dy=dy,
                delta_inf=delta,
                residual_inf=res,
                f1=f1(xn, yn),
                f2=f2(xn, yn),
            )
        )
        if delta < eps and res < eps:
            return xn, yn, rows
        x, y = xn, yn

    raise RuntimeError("Newton method did not converge.")


def simple_iteration_method(
    x0: float, y0: float, eps: float, max_iter: int = MAX_ITER
) -> tuple[float, float, list[IterRow]]:
    x, y = x0, y0
    rows: list[IterRow] = []

    for k in range(1, max_iter + 1):
        xn = phi_x(x, y)
        yn = phi_y(x, y)
        delta = max(abs(xn - x), abs(yn - y))
        res = residual_inf(xn, yn)
        rows.append(
            IterRow(
                k=k,
                x=xn,
                y=yn,
                delta_inf=delta,
                residual_inf=res,
                f1=f1(xn, yn),
                f2=f2(xn, yn),
            )
        )
        if delta < eps and res < eps:
            return xn, yn, rows
        x, y = xn, yn

    raise RuntimeError("Simple iteration method did not converge.")


def seidel_method(
    x0: float, y0: float, eps: float, max_iter: int = MAX_ITER
) -> tuple[float, float, list[IterRow]]:
    x, y = x0, y0
    rows: list[IterRow] = []

    for k in range(1, max_iter + 1):
        xn = phi_x(x, y)
        yn = phi_y(xn, y)
        delta = max(abs(xn - x), abs(yn - y))
        res = residual_inf(xn, yn)
        rows.append(
            IterRow(
                k=k,
                x=xn,
                y=yn,
                delta_inf=delta,
                residual_inf=res,
                f1=f1(xn, yn),
                f2=f2(xn, yn),
            )
        )
        if delta < eps and res < eps:
            return xn, yn, rows
        x, y = xn, yn

    raise RuntimeError("Seidel method did not converge.")


def line_search_alpha(x: float, y: float, gx: float, gy: float) -> float:
    def psi(alpha: float) -> float:
        return objective_phi(x - alpha * gx, y - alpha * gy)

    f0 = psi(0.0)
    mid = 1.0
    f_mid = psi(mid)

    while f_mid >= f0 and mid > 1e-8:
        mid *= 0.5
        f_mid = psi(mid)
    if mid <= 1e-8:
        return mid

    left = 0.0
    right = min(2.0 * mid, 64.0)
    f_right = psi(right)
    while f_right < f_mid and right < 64.0:
        left = mid
        mid = right
        f_mid = f_right
        right = min(2.0 * right, 64.0)
        if right == mid:
            break
        f_right = psi(right)

    if right - left < 1e-8:
        return mid

    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = left, right
    x1 = b - invphi * (b - a)
    x2 = a + invphi * (b - a)
    f1v = psi(x1)
    f2v = psi(x2)

    for _ in range(80):
        if b - a < 1e-8:
            break
        if f1v > f2v:
            a = x1
            x1 = x2
            f1v = f2v
            x2 = a + invphi * (b - a)
            f2v = psi(x2)
        else:
            b = x2
            x2 = x1
            f2v = f1v
            x1 = b - invphi * (b - a)
            f1v = psi(x1)

    return 0.5 * (a + b)


def steepest_descent_method(
    x0: float, y0: float, eps: float, max_iter: int = MAX_ITER
) -> tuple[float, float, list[DescentRow]]:
    x, y = x0, y0
    rows: list[DescentRow] = []

    for k in range(1, max_iter + 1):
        gx, gy = grad_phi(x, y)
        g_inf = max(abs(gx), abs(gy))
        if g_inf < 1e-14 and residual_inf(x, y) < eps:
            return x, y, rows

        alpha = line_search_alpha(x, y, gx, gy)
        xn = x - alpha * gx
        yn = y - alpha * gy
        delta = max(abs(xn - x), abs(yn - y))
        res = residual_inf(xn, yn)
        rows.append(
            DescentRow(
                k=k,
                x=xn,
                y=yn,
                alpha=alpha,
                grad_inf=g_inf,
                delta_inf=delta,
                residual_inf=res,
                phi=objective_phi(xn, yn),
            )
        )
        if delta < eps and res < eps:
            return xn, yn, rows
        x, y = xn, yn

    raise RuntimeError("Steepest descent did not converge.")


def _fmt(v: float, digits: int = 10) -> str:
    return f"{v:.{digits}f}"


def newton_table(rows: list[NewtonRow]) -> str:
    lines = [
        "| k | x_k | y_k | dx | dy | delta_inf | residual_inf |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.k} | {_fmt(r.x)} | {_fmt(r.y)} | {_fmt(r.dx)} | {_fmt(r.dy)} | {_fmt(r.delta_inf)} | {_fmt(r.residual_inf)} |"
        )
    return "\n".join(lines)


def iter_table(rows: list[IterRow]) -> str:
    lines = [
        "| k | x_k | y_k | delta_inf | residual_inf |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.k} | {_fmt(r.x)} | {_fmt(r.y)} | {_fmt(r.delta_inf)} | {_fmt(r.residual_inf)} |"
        )
    return "\n".join(lines)


def descent_table(rows: list[DescentRow]) -> str:
    lines = [
        "| k | x_k | y_k | alpha_k | grad_inf | delta_inf | residual_inf | Phi(x_k,y_k) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.k} | {_fmt(r.x)} | {_fmt(r.y)} | {_fmt(r.alpha)} | {_fmt(r.grad_inf)} | {_fmt(r.delta_inf)} | {_fmt(r.residual_inf)} | {_fmt(r.phi)} |"
        )
    return "\n".join(lines)


def write_svg(path: Path, root_xy: tuple[float, float]) -> None:
    width, height = 980, 640
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 70, 80
    plot_left, plot_right = margin_left, width - margin_right
    plot_top, plot_bottom = margin_top, height - margin_bottom

    x_min, x_max = -1.5, 0.5
    y_min, y_max = -2.2, 5.2

    def sx(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def sy(y: float) -> float:
        return plot_bottom - (y - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    def points_to_polyline(points: list[tuple[float, float]]) -> str:
        return " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)

    xs: list[float] = []
    x = x_min
    while x <= x_max + 1e-12:
        xs.append(round(x, 10))
        x += 0.01

    curve1 = [(xv, math.sin(xv + 2.0) - 1.5) for xv in xs]
    curve2_down = [
        (xv, 2.0 - math.acos(max(-1.0, min(1.0, -xv - 0.5)))) for xv in xs
    ]
    curve2_up = [
        (xv, 2.0 + math.acos(max(-1.0, min(1.0, -xv - 0.5)))) for xv in xs
    ]

    x_axis_y = sy(0.0)
    y_axis_x = sx(0.0)

    x_ticks = [round(-1.5 + 0.25 * i, 2) for i in range(9)]
    y_ticks = [round(-2.0 + 0.5 * i, 2) for i in range(15)]

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
    ]

    for xt in x_ticks:
        px = sx(xt)
        parts.append(
            f'  <line x1="{px:.2f}" y1="{plot_top:.2f}" x2="{px:.2f}" y2="{plot_bottom:.2f}" stroke="#ededed" stroke-width="1"/>'
        )
        parts.append(
            f'  <text x="{px:.2f}" y="{plot_bottom + 22:.2f}" text-anchor="middle" font-size="12" font-family="Arial" fill="#444">{xt:.2f}</text>'
        )

    for yt in y_ticks:
        py = sy(yt)
        parts.append(
            f'  <line x1="{plot_left:.2f}" y1="{py:.2f}" x2="{plot_right:.2f}" y2="{py:.2f}" stroke="#ededed" stroke-width="1"/>'
        )
        parts.append(
            f'  <text x="{plot_left - 12:.2f}" y="{py + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#444">{yt:.2f}</text>'
        )

    parts.append(
        f'  <line x1="{plot_left:.2f}" y1="{x_axis_y:.2f}" x2="{plot_right:.2f}" y2="{x_axis_y:.2f}" stroke="#666" stroke-width="1.4"/>'
    )
    parts.append(
        f'  <line x1="{y_axis_x:.2f}" y1="{plot_top:.2f}" x2="{y_axis_x:.2f}" y2="{plot_bottom:.2f}" stroke="#666" stroke-width="1.4"/>'
    )

    parts.append(
        f'  <polyline points="{points_to_polyline(curve1)}" fill="none" stroke="#0b7285" stroke-width="2.6"/>'
    )
    parts.append(
        f'  <polyline points="{points_to_polyline(curve2_down)}" fill="none" stroke="#c92a2a" stroke-width="2.6"/>'
    )
    parts.append(
        f'  <polyline points="{points_to_polyline(curve2_up)}" fill="none" stroke="#c92a2a" stroke-width="1.8" stroke-dasharray="6,4" opacity="0.75"/>'
    )

    rx, ry = root_xy
    parts.append(
        f'  <circle cx="{sx(rx):.2f}" cy="{sy(ry):.2f}" r="5.2" fill="#2f9e44"/>'
    )
    parts.append(
        f'  <text x="{sx(rx) + 10:.2f}" y="{sy(ry) - 10:.2f}" font-size="12" font-family="Arial" fill="#2f9e44">root≈({_fmt(rx,4)}, {_fmt(ry,4)})</text>'
    )

    parts.append(
        f'  <text x="{width / 2:.0f}" y="34" text-anchor="middle" font-size="28" font-family="Arial">Система варианта 17</text>'
    )
    parts.append(
        f'  <text x="{plot_right + 12:.2f}" y="{x_axis_y + 5:.2f}" font-size="18" font-family="Arial">x</text>'
    )
    parts.append(
        f'  <text x="{y_axis_x + 8:.2f}" y="{plot_top - 14:.2f}" font-size="18" font-family="Arial">y</text>'
    )

    legend_x = plot_left + 8
    legend_y = plot_top + 12
    parts.append(
        f'  <line x1="{legend_x:.2f}" y1="{legend_y:.2f}" x2="{legend_x + 24:.2f}" y2="{legend_y:.2f}" stroke="#0b7285" stroke-width="2.6"/>'
    )
    parts.append(
        f'  <text x="{legend_x + 30:.2f}" y="{legend_y + 4:.2f}" font-size="12" font-family="Arial">y = sin(x+2) - 1.5</text>'
    )
    parts.append(
        f'  <line x1="{legend_x:.2f}" y1="{legend_y + 20:.2f}" x2="{legend_x + 24:.2f}" y2="{legend_y + 20:.2f}" stroke="#c92a2a" stroke-width="2.6"/>'
    )
    parts.append(
        f'  <text x="{legend_x + 30:.2f}" y="{legend_y + 24:.2f}" font-size="12" font-family="Arial">x = -0.5 - cos(y-2)</text>'
    )
    parts.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def build_report(
    path: Path,
    x0: float,
    y0: float,
    newton_xy: tuple[float, float],
    newton_rows: list[NewtonRow],
    simple_xy: tuple[float, float],
    simple_rows: list[IterRow],
    seidel_xy: tuple[float, float],
    seidel_rows: list[IterRow],
    descent_xy: tuple[float, float],
    descent_rows: list[DescentRow],
) -> None:
    nx, ny = newton_xy
    sx, sy = simple_xy
    zx, zy = seidel_xy
    dx, dy = descent_xy

    lines: list[str] = []
    lines.append(f"# Лабораторная работа №4, вариант {VARIANT}")
    lines.append("")
    lines.append("## Постановка задачи")
    lines.append("")
    lines.append("Требуется решить систему нелинейных уравнений с точностью `eps = 0.001`:")
    lines.append("")
    lines.append("```text")
    lines.append("sin(x + 2) - y = 1.5")
    lines.append("x + cos(y - 2) = -0.5")
    lines.append("```")
    lines.append("")
    lines.append("Использованные методы:")
    lines.append("1. Метод Ньютона")
    lines.append("2. Метод простых итераций (Якоби)")
    lines.append("3. Метод Зейделя")
    lines.append("4. Метод наискорейшего спуска (дополнительное задание)")
    lines.append("")
    lines.append(f"Начальное приближение: `x0 = {_fmt(x0, 4)}`, `y0 = {_fmt(y0, 4)}`.")
    lines.append("")
    lines.append("Графическая интерпретация системы (пересечение кривых):")
    lines.append("")
    lines.append("![graph](graph.svg)")
    lines.append("")
    lines.append("## 1) Метод Ньютона")
    lines.append("")
    lines.append("Итерационная схема:")
    lines.append("")
    lines.append("```text")
    lines.append("J(x_k, y_k) * [dx_k, dy_k]^T = -F(x_k, y_k)")
    lines.append("x_(k+1) = x_k + dx_k")
    lines.append("y_(k+1) = y_k + dy_k")
    lines.append("```")
    lines.append("")
    lines.append(newton_table(newton_rows))
    lines.append("")
    lines.append(
        f"Итог: `x ≈ {_fmt(nx, 10)}`, `y ≈ {_fmt(ny, 10)}`, итераций: `{len(newton_rows)}`."
    )
    lines.append("")
    lines.append("## 2) Метод простых итераций (Якоби)")
    lines.append("")
    lines.append("Выбрано преобразование системы:")
    lines.append("")
    lines.append("```text")
    lines.append("x = -0.5 - cos(y - 2)")
    lines.append("y = sin(x + 2) - 1.5")
    lines.append("```")
    lines.append("")
    lines.append("Схема Якоби:")
    lines.append("")
    lines.append("```text")
    lines.append("x_(k+1) = -0.5 - cos(y_k - 2)")
    lines.append("y_(k+1) = sin(x_k + 2) - 1.5")
    lines.append("```")
    lines.append("")
    lines.append(iter_table(simple_rows))
    lines.append("")
    lines.append(
        f"Итог: `x ≈ {_fmt(sx, 10)}`, `y ≈ {_fmt(sy, 10)}`, итераций: `{len(simple_rows)}`."
    )
    lines.append("")
    lines.append("## 3) Метод Зейделя")
    lines.append("")
    lines.append("Схема Зейделя:")
    lines.append("")
    lines.append("```text")
    lines.append("x_(k+1) = -0.5 - cos(y_k - 2)")
    lines.append("y_(k+1) = sin(x_(k+1) + 2) - 1.5")
    lines.append("```")
    lines.append("")
    lines.append(iter_table(seidel_rows))
    lines.append("")
    lines.append(
        f"Итог: `x ≈ {_fmt(zx, 10)}`, `y ≈ {_fmt(zy, 10)}`, итераций: `{len(seidel_rows)}`."
    )
    lines.append("")
    lines.append("## 4) Дополнительное задание: метод наискорейшего спуска")
    lines.append("")
    lines.append("Функция минимизации:")
    lines.append("")
    lines.append("```text")
    lines.append("Phi(x, y) = f1(x, y)^2 + f2(x, y)^2")
    lines.append("```")
    lines.append("")
    lines.append("Градиентный шаг:")
    lines.append("")
    lines.append("```text")
    lines.append("[x_(k+1), y_(k+1)]^T = [x_k, y_k]^T - alpha_k * grad(Phi)(x_k, y_k)")
    lines.append("```")
    lines.append("")
    lines.append(
        "Шаг `alpha_k` выбирается одномерной минимизацией `Phi(x_k - alpha*Phi_x', y_k - alpha*Phi_y')`."
    )
    lines.append("")
    lines.append(descent_table(descent_rows))
    lines.append("")
    lines.append(
        f"Итог: `x ≈ {_fmt(dx, 10)}`, `y ≈ {_fmt(dy, 10)}`, итераций: `{len(descent_rows)}`."
    )
    lines.append("")
    lines.append("## 5) Сводный результат")
    lines.append("")
    lines.append("| Метод | x | y | iterations | residual_inf |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| Ньютон | {_fmt(nx, 10)} | {_fmt(ny, 10)} | {len(newton_rows)} | {_fmt(residual_inf(nx, ny), 12)} |"
    )
    lines.append(
        f"| Простые итерации | {_fmt(sx, 10)} | {_fmt(sy, 10)} | {len(simple_rows)} | {_fmt(residual_inf(sx, sy), 12)} |"
    )
    lines.append(
        f"| Зейдель | {_fmt(zx, 10)} | {_fmt(zy, 10)} | {len(seidel_rows)} | {_fmt(residual_inf(zx, zy), 12)} |"
    )
    lines.append(
        f"| Наискорейший спуск | {_fmt(dx, 10)} | {_fmt(dy, 10)} | {len(descent_rows)} | {_fmt(residual_inf(dx, dy), 12)} |"
    )
    lines.append("")
    lines.append(
        "Все методы сошлись к одному и тому же решению системы в пределах заданной точности."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(
    path: Path,
    x0: float,
    y0: float,
    newton_xy: tuple[float, float],
    newton_rows: list[NewtonRow],
    simple_xy: tuple[float, float],
    simple_rows: list[IterRow],
    seidel_xy: tuple[float, float],
    seidel_rows: list[IterRow],
    descent_xy: tuple[float, float],
    descent_rows: list[DescentRow],
) -> None:
    payload = {
        "variant": VARIANT,
        "eps": EPS,
        "initial_guess": {"x0": x0, "y0": y0},
        "system": {
            "f1": "sin(x + 2) - y - 1.5 = 0",
            "f2": "x + cos(y - 2) + 0.5 = 0",
        },
        "methods": {
            "newton": {
                "solution": {"x": newton_xy[0], "y": newton_xy[1]},
                "iterations": len(newton_rows),
                "residual_inf": residual_inf(*newton_xy),
                "history": [asdict(r) for r in newton_rows],
            },
            "simple_iteration": {
                "solution": {"x": simple_xy[0], "y": simple_xy[1]},
                "iterations": len(simple_rows),
                "residual_inf": residual_inf(*simple_xy),
                "history": [asdict(r) for r in simple_rows],
            },
            "seidel": {
                "solution": {"x": seidel_xy[0], "y": seidel_xy[1]},
                "iterations": len(seidel_rows),
                "residual_inf": residual_inf(*seidel_xy),
                "history": [asdict(r) for r in seidel_rows],
            },
            "steepest_descent": {
                "solution": {"x": descent_xy[0], "y": descent_xy[1]},
                "iterations": len(descent_rows),
                "residual_inf": residual_inf(*descent_xy),
                "history": [asdict(r) for r in descent_rows],
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Start close to the intersection found from the graph.
    x0 = 0.5
    y0 = -0.8

    newton_x, newton_y, newton_rows = newton_method(x0, y0, EPS)
    simple_x, simple_y, simple_rows = simple_iteration_method(x0, y0, EPS)
    seidel_x, seidel_y, seidel_rows = seidel_method(x0, y0, EPS)
    descent_x, descent_y, descent_rows = steepest_descent_method(x0, y0, EPS)

    write_svg(results_dir / "graph.svg", (newton_x, newton_y))
    build_report(
        path=results_dir / "report.md",
        x0=x0,
        y0=y0,
        newton_xy=(newton_x, newton_y),
        newton_rows=newton_rows,
        simple_xy=(simple_x, simple_y),
        simple_rows=simple_rows,
        seidel_xy=(seidel_x, seidel_y),
        seidel_rows=seidel_rows,
        descent_xy=(descent_x, descent_y),
        descent_rows=descent_rows,
    )
    write_json(
        path=results_dir / "results.json",
        x0=x0,
        y0=y0,
        newton_xy=(newton_x, newton_y),
        newton_rows=newton_rows,
        simple_xy=(simple_x, simple_y),
        simple_rows=simple_rows,
        seidel_xy=(seidel_x, seidel_y),
        seidel_rows=seidel_rows,
        descent_xy=(descent_x, descent_y),
        descent_rows=descent_rows,
    )

    print(f"Lab 4 variant {VARIANT} solved.")
    print(f"Results directory: {results_dir}")
    print(
        "Newton:",
        f"x={newton_x:.10f}",
        f"y={newton_y:.10f}",
        f"it={len(newton_rows)}",
    )
    print(
        "Simple iteration:",
        f"x={simple_x:.10f}",
        f"y={simple_y:.10f}",
        f"it={len(simple_rows)}",
    )
    print(
        "Seidel:",
        f"x={seidel_x:.10f}",
        f"y={seidel_y:.10f}",
        f"it={len(seidel_rows)}",
    )
    print(
        "Steepest descent:",
        f"x={descent_x:.10f}",
        f"y={descent_y:.10f}",
        f"it={len(descent_rows)}",
    )


if __name__ == "__main__":
    main()
