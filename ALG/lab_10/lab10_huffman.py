"""
Лабораторная работа 10.

Тема: код Хаффмана.

Задание:
- на вход программы подается текст;
- выполнить кодировку текста;
- выдать словарь, в котором каждому символу соответствует код,
  удовлетворяющий условию Фано;
- текст должен быть закодирован минимально возможной последовательностью
  нулей и единиц среди префиксных кодов для заданных частот;
- выполнить восстановление закодированного текста;
- оценить сложность работы алгоритма и время выполнения.
"""

from __future__ import annotations

import heapq
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@dataclass(frozen=True)
class HuffmanNode:
    """Узел дерева Хаффмана."""

    frequency: int
    symbol: str | None = None
    left: "HuffmanNode | None" = None
    right: "HuffmanNode | None" = None

    @property
    def is_leaf(self) -> bool:
        """Проверяет, является ли узел листом дерева."""
        return self.symbol is not None


def count_frequencies(text: str) -> dict[str, int]:
    """Подсчитывает частоты символов в тексте."""
    return dict(Counter(text))


def build_huffman_tree(frequencies: Mapping[str, int]) -> HuffmanNode | None:
    """
    Строит дерево Хаффмана по частотам символов.

    На каждом шаге из очереди берутся два узла с минимальными частотами
    и объединяются в один новый узел.
    """
    heap: list[tuple[int, str, int, HuffmanNode]] = []

    for order, symbol in enumerate(sorted(frequencies)):
        frequency = frequencies[symbol]
        if frequency <= 0:
            continue
        node = HuffmanNode(frequency=frequency, symbol=symbol)
        heapq.heappush(heap, (frequency, symbol, order, node))

    if not heap:
        return None

    next_order = len(heap)
    while len(heap) > 1:
        freq_left, min_left, _, left = heapq.heappop(heap)
        freq_right, min_right, _, right = heapq.heappop(heap)

        merged = HuffmanNode(
            frequency=freq_left + freq_right,
            left=left,
            right=right,
        )
        min_symbol = min(min_left, min_right)
        heapq.heappush(heap, (merged.frequency, min_symbol, next_order, merged))
        next_order += 1

    return heap[0][3]


def build_codes(root: HuffmanNode | None) -> dict[str, str]:
    """
    Обходит дерево Хаффмана и строит словарь символ -> двоичный код.

    Переход влево добавляет 0, переход вправо добавляет 1.
    """
    if root is None:
        return {}

    if root.is_leaf:
        return {root.symbol or "": "0"}

    codes: dict[str, str] = {}

    def walk(node: HuffmanNode, prefix: str) -> None:
        if node.is_leaf:
            codes[node.symbol or ""] = prefix
            return
        if node.left is not None:
            walk(node.left, prefix + "0")
        if node.right is not None:
            walk(node.right, prefix + "1")

    walk(root, "")
    return codes


def encode_text(text: str, codes: Mapping[str, str]) -> str:
    """Кодирует текст по готовому словарю кодов."""
    return "".join(codes[symbol] for symbol in text)


def decode_text(encoded_text: str, root: HuffmanNode | None) -> str:
    """Восстанавливает исходный текст по закодированной битовой строке."""
    if root is None:
        if encoded_text:
            raise ValueError("нельзя декодировать непустой текст без дерева Хаффмана")
        return ""

    if root.is_leaf:
        symbol = root.symbol or ""
        if any(bit != "0" for bit in encoded_text):
            raise ValueError("код Хаффмана для одного символа может содержать только биты 0")
        return symbol * len(encoded_text)

    decoded: list[str] = []
    node = root

    for bit in encoded_text:
        if bit == "0":
            next_node = node.left
        elif bit == "1":
            next_node = node.right
        else:
            raise ValueError(f"недопустимый символ в закодированной строке: {bit!r}")

        if next_node is None:
            raise ValueError("закодированная строка не соответствует дереву Хаффмана")

        node = next_node
        if node.is_leaf:
            decoded.append(node.symbol or "")
            node = root

    if node is not root:
        raise ValueError("закодированная строка закончилась в середине кода")

    return "".join(decoded)


def huffman_encode(text: str) -> tuple[dict[str, int], HuffmanNode | None, dict[str, str], str]:
    """Полный цикл кодирования: частоты, дерево, словарь кодов, битовая строка."""
    frequencies = count_frequencies(text)
    root = build_huffman_tree(frequencies)
    codes = build_codes(root)
    encoded_text = encode_text(text, codes)
    return frequencies, root, codes, encoded_text


def is_prefix_free(codes: Mapping[str, str]) -> bool:
    """
    Проверяет условие Фано: ни один код не является префиксом другого кода.
    """
    code_values = list(codes.values())
    for i, code in enumerate(code_values):
        for j, other in enumerate(code_values):
            if i != j and other.startswith(code):
                return False
    return True


def encoded_length_from_codes(text: str, codes: Mapping[str, str]) -> int:
    """Возвращает длину закодированного текста в битах."""
    return sum(len(codes[symbol]) for symbol in text)


def fixed_length_bits_per_symbol(alphabet_size: int) -> int:
    """Число бит на символ при равномерном кодировании алфавита."""
    if alphabet_size <= 1:
        return 1
    return math.ceil(math.log2(alphabet_size))


def format_symbol(symbol: str) -> str:
    """Печатает служебные символы в понятном виде."""
    if symbol == " ":
        return "<пробел>"
    if symbol == "\n":
        return "<перенос строки>"
    if symbol == "\t":
        return "<табуляция>"
    return symbol


def print_code_table(frequencies: Mapping[str, int], codes: Mapping[str, str]) -> None:
    """Печатает таблицу частот и кодов Хаффмана."""
    print("Словарь кодов Хаффмана:")
    rows = sorted(
        frequencies,
        key=lambda symbol: (-frequencies[symbol], format_symbol(symbol)),
    )
    for symbol in rows:
        print(
            f"  {format_symbol(symbol)!r:>10}: "
            f"частота={frequencies[symbol]:>2}, код={codes[symbol]}"
        )


def print_demo() -> None:
    """Демонстрация кодирования и восстановления текста."""
    print("Демонстрационный пример")
    text = "алгоритмы и анализ сложности"
    frequencies, root, codes, encoded_text = huffman_encode(text)
    decoded_text = decode_text(encoded_text, root)

    print(f"Исходный текст: {text!r}")
    print_code_table(frequencies, codes)
    print(f"\nЗакодированный текст:")
    print(f"  {encoded_text}")
    print(f"Длина кода Хаффмана: {len(encoded_text)} бит")

    fixed_bits = fixed_length_bits_per_symbol(len(frequencies))
    fixed_length = len(text) * fixed_bits
    print(f"Равномерный код: {fixed_bits} бит/символ, всего {fixed_length} бит")
    print(f"Условие Фано: {'выполнено' if is_prefix_free(codes) else 'ошибка'}")
    print(f"Восстановленный текст: {decoded_text!r}")
    print(f"Проверка восстановления: {'выполнено' if decoded_text == text else 'ошибка'}")
    print()


def measure_ms(func, *args, repeats: int = 5) -> float:
    """Среднее время выполнения функции в миллисекундах."""
    func(*args)
    total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args)
        total += time.perf_counter() - t0
    return total / repeats * 1000.0


def generate_random_text(length: int) -> str:
    """Генерирует текст с неравномерными частотами символов для замеров."""
    if length <= 0:
        return ""

    alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя .,;:-"
    weights = [
        8 if symbol in "оеаинтсрвлкмдпуяыьгзбчйхжшюцщэфъё" else 2
        for symbol in alphabet
    ]
    return "".join(random.choices(alphabet, weights=weights, k=length))


def encode_decode_cycle(text: str) -> int:
    """Функция для замеров: кодирует и декодирует текст, возвращает длину кода."""
    _, root, codes, encoded_text = huffman_encode(text)
    decoded_text = decode_text(encoded_text, root)
    if decoded_text != text:
        raise RuntimeError("восстановленный текст отличается от исходного")
    return len(encoded_text)


def run_benchmark() -> tuple[list[int], list[float], list[float]]:
    """Замеряет время кодирования/декодирования на растущих размерах текста."""
    sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000]
    encode_decode_timing: list[float] = []
    encoded_bits_per_symbol: list[float] = []

    for size in sizes:
        repeats = 7 if size <= 10_000 else (4 if size <= 100_000 else 2)
        text = generate_random_text(size)

        time_ms = measure_ms(encode_decode_cycle, text, repeats=repeats)
        frequencies, _, codes, encoded_text = huffman_encode(text)
        bits_per_symbol = len(encoded_text) / len(text) if text else 0.0

        encode_decode_timing.append(time_ms)
        encoded_bits_per_symbol.append(bits_per_symbol)
        print(
            f"n={size}: время={time_ms:.4f} мс, "
            f"алфавит={len(frequencies)}, средняя длина={bits_per_symbol:.3f} бит/символ"
        )

    return sizes, encode_decode_timing, encoded_bits_per_symbol


def print_benchmark_table(
    sizes: Sequence[int],
    timing: Sequence[float],
    bits_per_symbol: Sequence[float],
) -> None:
    """Печатает таблицу результатов замеров."""
    print("\nСравнение времени:")
    print(f"{'n':>10}{'кодирование+декодирование':>30}{'бит/символ':>15}")
    for size, time_ms, avg_bits in zip(sizes, timing, bits_per_symbol):
        print(f"{size:>10}{time_ms:>25.4f} мс{avg_bits:>15.3f}")


def plot_benchmark(
    sizes: Sequence[int],
    timing: Sequence[float],
    bits_per_symbol: Sequence[float],
) -> None:
    """Строит и сохраняет график времени и средней длины кода."""
    if plt is None:
        print("\nmatplotlib не установлен, график пропущен.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ЛР10: код Хаффмана", fontsize=13, fontweight="bold")

    axes[0].plot(sizes, timing, marker="o", linewidth=2, color="#1f77b4")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Размер текста n")
    axes[0].set_ylabel("Время (мс), логарифмическая шкала")
    axes[0].set_title("Время кодирования и декодирования")
    axes[0].grid(True, linestyle="--", alpha=0.6, which="both")

    axes[1].plot(sizes, bits_per_symbol, marker="s", linewidth=2, color="#2ca02c")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Размер текста n")
    axes[1].set_ylabel("Средняя длина кода, бит/символ")
    axes[1].set_title("Сжатие относительно длины текста")
    axes[1].grid(True, linestyle="--", alpha=0.6, which="both")

    output_path = Path(__file__).with_name("lab10_huffman_runtime.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик сохранен: {output_path.name}")


def print_complexity_notes() -> None:
    """Печатает теоретическую оценку сложности."""
    print("\nТеоретическая сложность:")
    print("1) Подсчет частот: O(n), где n — длина текста.")
    print("2) Построение дерева Хаффмана: O(k log k), где k — число разных символов.")
    print("3) Построение словаря кодов: O(k).")
    print("4) Кодирование текста: O(n).")
    print("5) Декодирование текста: O(L), где L — длина закодированной битовой строки.")
    print("Итого: O(n + k log k + L), обычно k намного меньше n.")


def main() -> None:
    """Запуск демонстрации, замеров времени и построения графика."""
    random.seed(42)

    print("Лабораторная работа 10: код Хаффмана")
    print("-" * 72)

    print_demo()
    sizes, timing, bits_per_symbol = run_benchmark()
    print_benchmark_table(sizes, timing, bits_per_symbol)
    print_complexity_notes()
    plot_benchmark(sizes, timing, bits_per_symbol)


if __name__ == "__main__":
    main()
