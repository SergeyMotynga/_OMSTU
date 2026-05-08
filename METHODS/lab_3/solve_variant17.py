from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


VARIANT = 17
GROUP_NUMBER = None

EPS_A = 1e-3
EPS_B = 1e-2
EPS_C_ITER = 1e-2


# a) Variant 17 from "ЛР № 3_2026 (1).pdf", page with variants for Gauss/inverse.
A_A = [
    [3.32, 1.49, -2.74, 0.68],
    [1.73, -0.85, 3.91, -1.82],
    [2.58, 4.12, -1.27, 2.93],
    [-1.87, 2.65, 0.94, -3.48],
]
b_A = [4.86, -2.67, 6.84, 1.79]


# б) Iterative methods with parameter m (variant).
m = VARIANT
A_B = [
    [3.0, 1.0, -1.0, 1.0],
    [1.0, -4.0, 1.0, -1.0],
    [-1.0, 1.0, 4.0, 1.0],
    [1.0, 2.0, 1.0, -5.0],
]
b_B = [3.0 * m, m - 6.0, 15.0 - m, m + 2.0]
x0_B = [0.7 * m, 1.0, 2.0, 0.5]


# в) Variant 17 from tridiagonal systems table.
A_C = [
    [3.0, 2.0, 0.0, 0.0],
    [-1.0, -5.0, 3.0, 0.0],
    [0.0, -2.0, 7.0, 4.0],
    [0.0, 0.0, 3.0, 5.0],
]
b_C = [9.0, -18.0, -6.0, -6.0]


# Additional assignment: eigenvalues/eigenvectors, variant 17 matrix.
A_EXTRA = np.array(
    [
        [9.49, -2.21, 0.77, 5.34],
        [2.78, 5.41, 9.28, -0.73],
        [6.28, 0.73, 10.52, 3.48],
        [0.15, 2.74, 0.21, -5.38],
    ],
    dtype=float,
)


@dataclass
class IterationRecord:
    k: int
    x: list[float]
    delta_inf: float
    residual_inf: float


@dataclass
class GaussForwardStep:
    k: int
    pivot_row: int
    pivot_value: float
    swapped: bool
    eliminated_rows: list[int]
    factors: list[float]
    matrix_after: list[list[float]]
    rhs_after: list[float]


@dataclass
class GaussBackwardStep:
    i: int
    sum_known: float
    x_i: float


@dataclass
class ThomasForwardStep:
    i: int
    denom: float
    c_prime_i: float
    d_prime_i: float


@dataclass
class ThomasBackwardStep:
    i: int
    x_i: float


@dataclass
class EigenIterRecord:
    k: int
    lambda_est: complex
    residual_inf: float


def _fmt(x: float, digits: int = 6) -> str:
    return f"{x:.{digits}f}"


def _fmt_complex(z: complex, digits: int = 6) -> str:
    re = round(z.real, digits)
    im = round(z.imag, digits)
    if abs(im) < 10 ** (-digits):
        return f"{re:.{digits}f}"
    sign = "+" if im >= 0 else "-"
    return f"{re:.{digits}f} {sign} {abs(im):.{digits}f}i"


def mat_vec(A: list[list[float]], x: list[float]) -> list[float]:
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in A]


def residual(A: list[list[float]], x: list[float], b: list[float]) -> list[float]:
    Ax = mat_vec(A, x)
    return [ax - bi for ax, bi in zip(Ax, b)]


def norm_inf(v: Iterable[float]) -> float:
    return max(abs(vi) for vi in v)


def gauss_solve_with_trace(
    A: list[list[float]], b: list[float]
) -> tuple[list[float], list[GaussForwardStep], list[GaussBackwardStep]]:
    n = len(A)
    M = [row[:] for row in A]
    y = b[:]
    forward_steps: list[GaussForwardStep] = []

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[pivot_row][k]) < 1e-14:
            raise ValueError("Singular matrix in Gaussian elimination.")
        swapped = pivot_row != k
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
            y[k], y[pivot_row] = y[pivot_row], y[k]

        pivot = M[k][k]
        eliminated_rows: list[int] = []
        factors: list[float] = []
        for i in range(k + 1, n):
            factor = M[i][k] / pivot
            eliminated_rows.append(i)
            factors.append(factor)
            M[i][k] = 0.0
            for j in range(k + 1, n):
                M[i][j] -= factor * M[k][j]
            y[i] -= factor * y[k]

        forward_steps.append(
            GaussForwardStep(
                k=k,
                pivot_row=pivot_row,
                pivot_value=pivot,
                swapped=swapped,
                eliminated_rows=eliminated_rows,
                factors=factors,
                matrix_after=[row[:] for row in M],
                rhs_after=y[:],
            )
        )

    x = [0.0] * n
    backward_steps: list[GaussBackwardStep] = []
    for i in range(n - 1, -1, -1):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / M[i][i]
        backward_steps.append(GaussBackwardStep(i=i, sum_known=s, x_i=x[i]))
    backward_steps.reverse()
    return x, forward_steps, backward_steps


def inverse_matrix(A: list[list[float]]) -> list[list[float]]:
    n = len(A)
    aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(aug[i][k]))
        if abs(aug[pivot_row][k]) < 1e-14:
            raise ValueError("Singular matrix in inversion.")
        if pivot_row != k:
            aug[k], aug[pivot_row] = aug[pivot_row], aug[k]

        pivot = aug[k][k]
        for j in range(2 * n):
            aug[k][j] /= pivot

        for i in range(n):
            if i == k:
                continue
            factor = aug[i][k]
            if factor == 0.0:
                continue
            for j in range(2 * n):
                aug[i][j] -= factor * aug[k][j]

    inv = [row[n:] for row in aug]
    return inv


def multiply_matrix_vector(A: list[list[float]], b: list[float]) -> list[float]:
    return [sum(aij * bj for aij, bj in zip(row, b)) for row in A]


def jacobi_method(
    A: list[list[float]],
    b: list[float],
    x0: list[float],
    eps: float,
    max_iter: int = 200,
) -> tuple[list[float], list[IterationRecord]]:
    n = len(A)
    x_prev = x0[:]
    history: list[IterationRecord] = []

    for k in range(1, max_iter + 1):
        x_next = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * x_prev[j] for j in range(n) if j != i)
            x_next[i] = (b[i] - s) / A[i][i]

        delta = norm_inf([x_next[i] - x_prev[i] for i in range(n)])
        res = norm_inf(residual(A, x_next, b))
        history.append(IterationRecord(k=k, x=x_next[:], delta_inf=delta, residual_inf=res))

        if delta < eps:
            return x_next, history
        x_prev = x_next

    raise RuntimeError("Jacobi method did not converge within max_iter.")


def gauss_seidel_method(
    A: list[list[float]],
    b: list[float],
    x0: list[float],
    eps: float,
    max_iter: int = 200,
) -> tuple[list[float], list[IterationRecord]]:
    n = len(A)
    x = x0[:]
    history: list[IterationRecord] = []

    for k in range(1, max_iter + 1):
        x_old = x[:]
        for i in range(n):
            s_left = sum(A[i][j] * x[j] for j in range(i))
            s_right = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s_left - s_right) / A[i][i]

        delta = norm_inf([x[i] - x_old[i] for i in range(n)])
        res = norm_inf(residual(A, x, b))
        history.append(IterationRecord(k=k, x=x[:], delta_inf=delta, residual_inf=res))

        if delta < eps:
            return x, history

    raise RuntimeError("Gauss-Seidel method did not converge within max_iter.")


def solve_tridiagonal_thomas_with_trace(
    A: list[list[float]], d: list[float]
) -> tuple[list[float], list[ThomasForwardStep], list[ThomasBackwardStep]]:
    n = len(A)
    lower = [0.0] * n
    diag = [0.0] * n
    upper = [0.0] * n
    for i in range(n):
        diag[i] = A[i][i]
        if i > 0:
            lower[i] = A[i][i - 1]
        if i < n - 1:
            upper[i] = A[i][i + 1]

    c_prime = [0.0] * n
    d_prime = [0.0] * n
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = d[0] / diag[0]
    forward_steps: list[ThomasForwardStep] = [
        ThomasForwardStep(
            i=0,
            denom=diag[0],
            c_prime_i=c_prime[0],
            d_prime_i=d_prime[0],
        )
    ]

    for i in range(1, n):
        denom = diag[i] - lower[i] * c_prime[i - 1]
        if abs(denom) < 1e-14:
            raise ValueError("Zero pivot in Thomas algorithm.")
        c_prime[i] = upper[i] / denom if i < n - 1 else 0.0
        d_prime[i] = (d[i] - lower[i] * d_prime[i - 1]) / denom
        forward_steps.append(
            ThomasForwardStep(
                i=i,
                denom=denom,
                c_prime_i=c_prime[i],
                d_prime_i=d_prime[i],
            )
        )

    x = [0.0] * n
    x[-1] = d_prime[-1]
    backward_steps: list[ThomasBackwardStep] = [ThomasBackwardStep(i=n - 1, x_i=x[-1])]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
        backward_steps.append(ThomasBackwardStep(i=i, x_i=x[i]))
    backward_steps.reverse()
    return x, forward_steps, backward_steps


def eigenpairs_numpy(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    vals, vecs = np.linalg.eig(A)

    order = sorted(
        range(len(vals)),
        key=lambda i: (round(vals[i].real, 12), round(vals[i].imag, 12)),
        reverse=True,
    )
    vals = vals[order]
    vecs = vecs[:, order]

    residual_norms: list[float] = []
    for i in range(len(vals)):
        lam = vals[i]
        v = vecs[:, i]
        r = A @ v - lam * v
        residual_norms.append(float(np.max(np.abs(r))))
    return vals, vecs, residual_norms


def inverse_iteration_with_shift(
    A: np.ndarray,
    shift: complex,
    x0: np.ndarray,
    max_iter: int = 7,
) -> tuple[np.ndarray, list[EigenIterRecord]]:
    x = x0.astype(complex).copy()
    x = x / np.linalg.norm(x)
    I = np.eye(A.shape[0], dtype=complex)
    records: list[EigenIterRecord] = []

    for k in range(1, max_iter + 1):
        y = np.linalg.solve(A - shift * I, x)
        x_next = y / np.linalg.norm(y)
        lam_est = np.vdot(x_next, A @ x_next) / np.vdot(x_next, x_next)
        residual_inf = float(np.max(np.abs(A @ x_next - lam_est * x_next)))
        records.append(
            EigenIterRecord(
                k=k,
                lambda_est=complex(lam_est),
                residual_inf=residual_inf,
            )
        )
        x = x_next

    return x, records


def vector_to_markdown(v: list[float] | np.ndarray, digits: int = 6) -> str:
    return "[" + ", ".join(_fmt(float(x), digits) for x in v) + "]"


def record_table(records: list[IterationRecord], digits: int = 6) -> str:
    lines = [
        "| k | x | delta_inf | residual_inf |",
        "|---:|:---|---:|---:|",
    ]
    for rec in records:
        lines.append(
            f"| {rec.k} | {vector_to_markdown(rec.x, digits)} | {_fmt(rec.delta_inf, digits)} | {_fmt(rec.residual_inf, digits)} |"
        )
    return "\n".join(lines)


def matrix_to_markdown(A: list[list[float]], digits: int = 2) -> str:
    lines = []
    for row in A:
        lines.append("| " + " | ".join(_fmt(v, digits) for v in row) + " |")
    sep = "| " + " | ".join(["---:"] * len(A[0])) + " |"
    header = "| " + " | ".join(f"c{i+1}" for i in range(len(A[0]))) + " |"
    return "\n".join([header, sep] + lines)


def augmented_matrix_to_markdown(
    A: list[list[float]], b: list[float], digits: int = 4
) -> str:
    n_cols = len(A[0])
    header_cells = [f"a{i+1}" for i in range(n_cols)] + ["b"]
    header = "| " + " | ".join(header_cells) + " |"
    sep = "| " + " | ".join(["---:"] * len(header_cells)) + " |"
    rows = []
    for row, bi in zip(A, b):
        cells = [_fmt(v, digits) for v in row] + [_fmt(bi, digits)]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def build_report(
    results_dir: Path,
    x_gauss: list[float],
    gauss_forward_steps: list[GaussForwardStep],
    gauss_backward_steps: list[GaussBackwardStep],
    x_inv: list[float],
    inv_A: list[list[float]],
    res_a_gauss: list[float],
    res_a_inv: list[float],
    x_jacobi: list[float],
    jacobi_hist: list[IterationRecord],
    x_seidel: list[float],
    seidel_hist: list[IterationRecord],
    res_b_jacobi: list[float],
    res_b_seidel: list[float],
    x_thomas: list[float],
    thomas_forward_steps: list[ThomasForwardStep],
    thomas_backward_steps: list[ThomasBackwardStep],
    res_c: list[float],
    x0_c_seidel: list[float],
    x_c_seidel: list[float],
    c_seidel_hist: list[IterationRecord],
    res_c_seidel: list[float],
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    eig_residuals: list[float],
    eig_iter_shifts: list[complex],
    eig_iter_histories: list[list[EigenIterRecord]],
) -> None:
    lines = []
    lines.append("# Лабораторная работа №3, вариант 17")
    lines.append("")
    lines.append("## Что решено")
    lines.append("")
    lines.append("1. Основная часть:")
    lines.append("1. пункт `a` — СЛАУ методами Гаусса и обратной матрицы (точность 0.001);")
    lines.append("2. пункт `б` — СЛАУ методами Якоби и Зейделя (`eps=0.01`, `m=17`);")
    lines.append("3. пункт `в` — СЛАУ методом прогонки (Thomas).")
    lines.append("2. Дополнительная часть: найдены собственные числа и собственные векторы матрицы варианта 17.")
    lines.append("")
    lines.append("## a) Методы Гаусса и обратной матрицы")
    lines.append("")
    lines.append("Исходная расширенная матрица `[A|b]`:")
    lines.append("")
    lines.append(augmented_matrix_to_markdown(A_A, b_A, digits=4))
    lines.append("")
    lines.append("Пошаговый ход метода Гаусса (прямой ход):")
    lines.append("")
    for step in gauss_forward_steps:
        lines.append(
            f"Шаг `k={step.k+1}`: ведущая строка `{step.pivot_row+1}`, ведущий элемент `{_fmt(step.pivot_value, 6)}`."
        )
        if step.swapped:
            lines.append("Выполнена перестановка строк для выбора ведущего элемента.")
        if step.eliminated_rows:
            lines.append(
                "Коэффициенты исключения: "
                + ", ".join(
                    f"m(r{ri+1})={_fmt(fi, 6)}"
                    for ri, fi in zip(step.eliminated_rows, step.factors)
                )
                + "."
            )
        lines.append("")
        lines.append(augmented_matrix_to_markdown(step.matrix_after, step.rhs_after, digits=6))
        lines.append("")

    lines.append("Обратный ход (вычисление неизвестных):")
    lines.append("")
    lines.append("| i | sum_known | x_i |")
    lines.append("|---:|---:|---:|")
    for step in gauss_backward_steps:
        lines.append(
            f"| {step.i+1} | {_fmt(step.sum_known, 9)} | {_fmt(step.x_i, 9)} |"
        )
    lines.append("")
    lines.append(
        "Примечание: пункт `a` решается прямыми методами (Гаусс, обратная матрица), "
        "поэтому здесь нет итераций в смысле последовательных приближений."
    )
    lines.append("")
    lines.append("Матрица `A_a`:")
    lines.append("")
    lines.append(matrix_to_markdown(A_A, digits=2))
    lines.append("")
    lines.append("Обратная матрица `A^-1`:")
    lines.append("")
    lines.append(matrix_to_markdown(inv_A, digits=6))
    lines.append("")
    lines.append(f"Решение методом Гаусса: `x = {vector_to_markdown(x_gauss, digits=6)}`")
    lines.append(f"Решение через `A^-1 * b`: `x = {vector_to_markdown(x_inv, digits=6)}`")
    lines.append("")
    lines.append("Невязка `r = A*x - b`:")
    lines.append("")
    lines.append(f"- для Гаусса: `{vector_to_markdown(res_a_gauss, digits=9)}`")
    lines.append(f"- для обратной матрицы: `{vector_to_markdown(res_a_inv, digits=9)}`")
    lines.append("")
    lines.append("Округление до 0.001:")
    lines.append("")
    lines.append(f"`x ≈ [{', '.join(f'{xi:.3f}' for xi in x_gauss)}]`")
    lines.append("")
    lines.append("## б) Итерационные методы (m=17, eps=0.01)")
    lines.append("")
    lines.append("Система:")
    lines.append("")
    lines.append(matrix_to_markdown(A_B, digits=2))
    lines.append("")
    lines.append(f"`b_b = {vector_to_markdown(b_B, digits=2)}`")
    lines.append(f"`x0 = {vector_to_markdown(x0_B, digits=2)}`")
    lines.append("")
    lines.append("### Метод Якоби")
    lines.append("")
    lines.append(f"Итог: `x = {vector_to_markdown(x_jacobi, digits=6)}`")
    lines.append(f"Итераций: `{len(jacobi_hist)}`")
    lines.append(f"Невязка: `{vector_to_markdown(res_b_jacobi, digits=9)}`")
    lines.append("")
    lines.append(record_table(jacobi_hist, digits=6))
    lines.append("")
    lines.append("### Метод Зейделя")
    lines.append("")
    lines.append(f"Итог: `x = {vector_to_markdown(x_seidel, digits=6)}`")
    lines.append(f"Итераций: `{len(seidel_hist)}`")
    lines.append(f"Невязка: `{vector_to_markdown(res_b_seidel, digits=9)}`")
    lines.append("")
    lines.append(record_table(seidel_hist, digits=6))
    lines.append("")
    lines.append("## в) Метод прогонки (Thomas)")
    lines.append("")
    lines.append("Трёхдиагональная система варианта 17:")
    lines.append("")
    lines.append(matrix_to_markdown(A_C, digits=2))
    lines.append("")
    lines.append(f"`b_c = {vector_to_markdown(b_C, digits=2)}`")
    lines.append("")
    lines.append("Прямой ход прогонки (коэффициенты `c'_i`, `d'_i`):")
    lines.append("")
    lines.append("| i | denom | c'_i | d'_i |")
    lines.append("|---:|---:|---:|---:|")
    for step in thomas_forward_steps:
        lines.append(
            f"| {step.i+1} | {_fmt(step.denom, 9)} | {_fmt(step.c_prime_i, 9)} | {_fmt(step.d_prime_i, 9)} |"
        )
    lines.append("")
    lines.append("Обратный ход прогонки:")
    lines.append("")
    lines.append("| i | x_i |")
    lines.append("|---:|---:|")
    for step in thomas_backward_steps:
        lines.append(f"| {step.i+1} | {_fmt(step.x_i, 9)} |")
    lines.append("")
    lines.append(f"Решение прогонкой: `x = {vector_to_markdown(x_thomas, digits=6)}`")
    lines.append(f"Невязка прогонки: `{vector_to_markdown(res_c, digits=9)}`")
    lines.append("")
    lines.append("Сравнение: решение той же системы методом Зейделя (`eps=0.01`)")
    lines.append("")
    lines.append(f"`x0 = {vector_to_markdown(x0_c_seidel, digits=2)}`")
    lines.append(f"Итераций: `{len(c_seidel_hist)}`")
    lines.append(f"Решение Зейделя: `x = {vector_to_markdown(x_c_seidel, digits=6)}`")
    lines.append(f"Невязка Зейделя: `{vector_to_markdown(res_c_seidel, digits=9)}`")
    lines.append("")
    lines.append(record_table(c_seidel_hist, digits=6))
    lines.append("")
    lines.append("## Дополнительное задание: собственные числа и векторы")
    lines.append("")
    lines.append("Матрица `A_extra` (вариант 17):")
    lines.append("")
    lines.append(matrix_to_markdown(A_EXTRA.tolist(), digits=2))
    lines.append("")
    lines.append("Собственные пары (`lambda_i`, `v_i`):")
    lines.append("")
    lines.append(
        "| i | lambda_i | v_i (нормированный вектор из `numpy.linalg.eig`) | residual_inf = max abs(Av - lambda v) |"
    )
    lines.append("|---:|:---|:---|---:|")
    for i in range(len(eigvals)):
        lam = eigvals[i]
        vec = eigvecs[:, i]
        vec_str = "[" + ", ".join(_fmt_complex(complex(v), 6) for v in vec) + "]"
        lines.append(
            f"| {i+1} | {_fmt_complex(complex(lam), 6)} | {vec_str} | {_fmt(eig_residuals[i], 12)} |"
        )
    lines.append("")
    lines.append("Итерационная визуализация с `lambda` (обратные итерации со сдвигом)")
    lines.append("")
    lines.append(
        "Для каждой собственной пары использован сдвиг `mu = lambda_i + 0.01` "
        "(для комплексных `lambda_i`: `mu = lambda_i + 0.01 + 0.01i`)."
    )
    lines.append("Показываем сходимость `lambda_est`, ошибки по `lambda` и невязки `residual_inf` по шагам.")
    lines.append("")
    for i, (target_lam, mu, hist) in enumerate(
        zip(eigvals, eig_iter_shifts, eig_iter_histories), start=1
    ):
        lines.append(
            f"Пара `{i}`: target `lambda = {_fmt_complex(complex(target_lam), 6)}`, shift `mu = {_fmt_complex(complex(mu), 6)}`"
        )
        lines.append("")
        lines.append("| k | lambda_est | abs(lambda_est - lambda_target) | residual_inf |")
        lines.append("|---:|:---|---:|---:|")
        for rec in hist:
            lines.append(
                f"| {rec.k} | {_fmt_complex(rec.lambda_est, 9)} | {_fmt(abs(rec.lambda_est - target_lam), 12)} | {_fmt(rec.residual_inf, 12)} |"
            )
        lines.append("")
    if GROUP_NUMBER is not None:
        lines.append(f"Примечание: `n` (номер группы) = {GROUP_NUMBER}.")
    else:
        lines.append("Примечание: параметр `n` из таблицы ЛР для варианта 17 в этой работе не используется.")
    lines.append("")

    (results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "variant": VARIANT,
        "main": {
            "a": {
                "x_gauss": x_gauss,
                "gauss_forward_steps": [
                    {
                        "k": step.k + 1,
                        "pivot_row": step.pivot_row + 1,
                        "pivot_value": step.pivot_value,
                        "swapped": step.swapped,
                        "eliminated_rows": [ri + 1 for ri in step.eliminated_rows],
                        "factors": step.factors,
                        "matrix_after": step.matrix_after,
                        "rhs_after": step.rhs_after,
                    }
                    for step in gauss_forward_steps
                ],
                "gauss_backward_steps": [
                    {
                        "i": step.i + 1,
                        "sum_known": step.sum_known,
                        "x_i": step.x_i,
                    }
                    for step in gauss_backward_steps
                ],
                "x_inverse": x_inv,
                "residual_gauss": res_a_gauss,
                "residual_inverse": res_a_inv,
                "inverse_matrix": inv_A,
            },
            "b": {
                "m": m,
                "eps": EPS_B,
                "x0": x0_B,
                "jacobi": {
                    "x": x_jacobi,
                    "iterations": len(jacobi_hist),
                    "history": [
                        {
                            "k": rec.k,
                            "x": rec.x,
                            "delta_inf": rec.delta_inf,
                            "residual_inf": rec.residual_inf,
                        }
                        for rec in jacobi_hist
                    ],
                    "residual": res_b_jacobi,
                },
                "seidel": {
                    "x": x_seidel,
                    "iterations": len(seidel_hist),
                    "history": [
                        {
                            "k": rec.k,
                            "x": rec.x,
                            "delta_inf": rec.delta_inf,
                            "residual_inf": rec.residual_inf,
                        }
                        for rec in seidel_hist
                    ],
                    "residual": res_b_seidel,
                },
            },
            "c": {
                "x_thomas": x_thomas,
                "forward_steps": [
                    {
                        "i": step.i + 1,
                        "denom": step.denom,
                        "c_prime_i": step.c_prime_i,
                        "d_prime_i": step.d_prime_i,
                    }
                    for step in thomas_forward_steps
                ],
                "backward_steps": [
                    {
                        "i": step.i + 1,
                        "x_i": step.x_i,
                    }
                    for step in thomas_backward_steps
                ],
                "residual": res_c,
                "seidel_compare": {
                    "eps": EPS_C_ITER,
                    "x0": x0_c_seidel,
                    "x": x_c_seidel,
                    "iterations": len(c_seidel_hist),
                    "history": [
                        {
                            "k": rec.k,
                            "x": rec.x,
                            "delta_inf": rec.delta_inf,
                            "residual_inf": rec.residual_inf,
                        }
                        for rec in c_seidel_hist
                    ],
                    "residual": res_c_seidel,
                },
            },
        },
        "extra": {
            "eigenpairs": [
                {
                    "index": i + 1,
                    "eigenvalue": {
                        "re": float(np.real(eigvals[i])),
                        "im": float(np.imag(eigvals[i])),
                    },
                    "eigenvector": [
                        {"re": float(np.real(v)), "im": float(np.imag(v))}
                        for v in eigvecs[:, i]
                    ],
                    "residual_inf": eig_residuals[i],
                }
                for i in range(len(eigvals))
            ],
            "inverse_iteration_visualization": [
                {
                    "index": i + 1,
                    "target_lambda": {
                        "re": float(np.real(eigvals[i])),
                        "im": float(np.imag(eigvals[i])),
                    },
                    "shift_mu": {
                        "re": float(np.real(eig_iter_shifts[i])),
                        "im": float(np.imag(eig_iter_shifts[i])),
                    },
                    "history": [
                        {
                            "k": rec.k,
                            "lambda_est": {
                                "re": float(np.real(rec.lambda_est)),
                                "im": float(np.imag(rec.lambda_est)),
                            },
                            "lambda_abs_error": float(abs(rec.lambda_est - eigvals[i])),
                            "residual_inf": rec.residual_inf,
                        }
                        for rec in eig_iter_histories[i]
                    ],
                }
                for i in range(len(eigvals))
            ],
        },
    }
    (results_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    x_gauss, gauss_forward_steps, gauss_backward_steps = gauss_solve_with_trace(A_A, b_A)
    inv_A = inverse_matrix(A_A)
    x_inv = multiply_matrix_vector(inv_A, b_A)
    res_a_gauss = residual(A_A, x_gauss, b_A)
    res_a_inv = residual(A_A, x_inv, b_A)

    x_jacobi, jacobi_hist = jacobi_method(A_B, b_B, x0_B, EPS_B)
    x_seidel, seidel_hist = gauss_seidel_method(A_B, b_B, x0_B, EPS_B)
    res_b_jacobi = residual(A_B, x_jacobi, b_B)
    res_b_seidel = residual(A_B, x_seidel, b_B)

    x_thomas, thomas_forward_steps, thomas_backward_steps = solve_tridiagonal_thomas_with_trace(A_C, b_C)
    res_c = residual(A_C, x_thomas, b_C)
    x0_c_seidel = [0.0, 0.0, 0.0, 0.0]
    x_c_seidel, c_seidel_hist = gauss_seidel_method(
        A_C, b_C, x0_c_seidel, EPS_C_ITER, max_iter=500
    )
    res_c_seidel = residual(A_C, x_c_seidel, b_C)

    eigvals, eigvecs, eig_residuals = eigenpairs_numpy(A_EXTRA)
    eig_iter_shifts: list[complex] = []
    eig_iter_histories: list[list[EigenIterRecord]] = []
    x0_extra = np.ones(A_EXTRA.shape[0], dtype=complex)
    A_extra_complex = A_EXTRA.astype(complex)
    for lam in eigvals:
        if abs(np.imag(lam)) > 1e-12:
            mu = complex(lam) + (1e-2 + 1e-2j)
        else:
            mu = complex(lam) + 1e-2
        eig_iter_shifts.append(mu)
        _, hist = inverse_iteration_with_shift(
            A_extra_complex, mu, x0_extra, max_iter=7
        )
        eig_iter_histories.append(hist)

    build_report(
        results_dir=results_dir,
        x_gauss=x_gauss,
        gauss_forward_steps=gauss_forward_steps,
        gauss_backward_steps=gauss_backward_steps,
        x_inv=x_inv,
        inv_A=inv_A,
        res_a_gauss=res_a_gauss,
        res_a_inv=res_a_inv,
        x_jacobi=x_jacobi,
        jacobi_hist=jacobi_hist,
        x_seidel=x_seidel,
        seidel_hist=seidel_hist,
        res_b_jacobi=res_b_jacobi,
        res_b_seidel=res_b_seidel,
        x_thomas=x_thomas,
        thomas_forward_steps=thomas_forward_steps,
        thomas_backward_steps=thomas_backward_steps,
        res_c=res_c,
        x0_c_seidel=x0_c_seidel,
        x_c_seidel=x_c_seidel,
        c_seidel_hist=c_seidel_hist,
        res_c_seidel=res_c_seidel,
        eigvals=eigvals,
        eigvecs=eigvecs,
        eig_residuals=eig_residuals,
        eig_iter_shifts=eig_iter_shifts,
        eig_iter_histories=eig_iter_histories,
    )

    print(f"Lab 3 variant {VARIANT} solved.")
    print(f"Results directory: {results_dir}")
    print(f"a) Gauss x: {x_gauss}")
    print(f"б) Jacobi x: {x_jacobi}; iterations={len(jacobi_hist)}")
    print(f"б) Seidel x: {x_seidel}; iterations={len(seidel_hist)}")
    print(f"в) Thomas x: {x_thomas}")
    print("Extra eigenvalues:", [complex(v) for v in eigvals])


if __name__ == "__main__":
    main()
