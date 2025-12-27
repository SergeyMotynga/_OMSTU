import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from collections import Counter
import math
import threading
import queue
from typing import Tuple, List, Optional, Dict, Any, Set


class AffineCipherCryptoanalysis:
    """Класс для модульной арифметики, частотного анализа и криптоанализа аффинного шифра."""

    ALPHABET: Dict[str, int] = {
        "а": 0, "б": 1, "в": 2, "г": 3,
        "д": 4, "е": 5, "ё": 5, "ж": 6, "з": 7,
        "и": 8, "й": 9, "к": 10, "л": 11,
        "м": 12, "н": 13, "о": 14, "п": 15,
        "р": 16, "с": 17, "т": 18, "у": 19,
        "ф": 20, "х": 21, "ц": 22, "ч": 23,
        "ш": 24, "щ": 25, "ъ": 26, "ы": 27,
        "ь": 28, "э": 29, "ю": 30, "я": 31,
    }

    ALPHABET_REVERSE: Dict[int, str] = {v: k for k, v in ALPHABET.items() if k != "ё"}

    LETTER_FREQUENCIES: List[Tuple[str, float]] = [
        ("о", 0.090), ("е", 0.072), ("а", 0.062), ("и", 0.062),
        ("т", 0.053), ("н", 0.053), ("с", 0.045), ("р", 0.040),
        ("в", 0.038), ("л", 0.035), ("к", 0.028), ("м", 0.026),
        ("д", 0.025), ("п", 0.023), ("у", 0.021), ("я", 0.018),
        ("ы", 0.016), ("з", 0.016), ("ь", 0.015), ("б", 0.014),
        ("г", 0.013), ("ч", 0.012), ("й", 0.010), ("х", 0.009),
        ("ж", 0.007), ("ю", 0.006), ("ш", 0.006), ("ц", 0.004),
        ("щ", 0.003), ("э", 0.003), ("ф", 0.002),
    ]

    def __init__(self, modulus: int = 32):
        """Инициализация анализатора."""
        self.modulus = modulus
        self.stop_cracking = False
        self.found_solution = False

    def normalize_for_frequency(self, text: str) -> str:
        """Нормализует текст для частотного анализа: ё→е, ъ→ь, нижний регистр."""
        t = text.lower()
        t = t.replace("ё", "е")
        t = t.replace("ъ", "ь")
        return t

    def extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        """Расширенный алгоритм Евклида: возвращает (g (НОД), u (коэф. Безу), v (коэф. Безу)), где u*a + v*b = g = gcd(a,b)."""
        if b == 0:
            return abs(a), 1 if a >= 0 else -1, 0

        old_r, r = a, b
        old_u, u = 1, 0
        old_v, v = 0, 1

        while r != 0:
            q = old_r // r
            old_r, r = r, old_r - q * r
            old_u, u = u, old_u - q * u
            old_v, v = v, old_v - q * v

        g = old_r
        if g < 0:
            g, old_u, old_v = -g, -old_u, -old_v
        return g, old_u, old_v

    def find_inverse(self, a: int, m: int) -> Optional[int]:
        """Находит обратный элемент a^{-1} по модулю m или None, если обратного нет."""
        a = a % m
        g, u, _ = self.extended_gcd(a, m)
        if g != 1:
            return None
        return u % m

    def solve_congruence(self, a: int, b: int, m: int) -> Optional[List[int]]:
        """Решает сравнение a*x ≡ b (mod m). Возвращает список всех решений или None, если решений нет."""
        a = a % m
        b = b % m
        g, u, _ = self.extended_gcd(a, m)
        if b % g != 0:
            return None

        x0 = (u * (b // g)) % m
        step = m // g
        solutions = sorted({(x0 + k * step) % m for k in range(g)})
        return solutions

    def solve_system(self, a1: int, b1: int, a2: int, b2: int, m: int) -> Optional[List[Tuple[int, int]]]:
        """Решает систему a1*x + y ≡ b1 (mod m), a2*x + y ≡ b2 (mod m). Возвращает все решения (x,y) или None."""
        a1 %= m
        a2 %= m
        b1 %= m
        b2 %= m

        diff_a = (a1 - a2) % m
        diff_b = (b1 - b2) % m

        x_solutions = self.solve_congruence(diff_a, diff_b, m)
        if x_solutions is None:
            return None

        solutions = []
        for x in x_solutions:
            y = (b1 - a1 * x) % m
            solutions.append((x, y))
        return solutions

    def encrypt_affine(self, text: str, a: int, b: int) -> str:
        """Шифрует текст аффинным шифром по текущему модулю."""
        res = []
        for ch in text.lower():
            if ch in self.ALPHABET:
                x = self.ALPHABET[ch]
                y = (a * x + b) % self.modulus
                res.append(self.ALPHABET_REVERSE[y])
        return "".join(res)

    def decrypt_affine(self, text: str, a: int, b: int) -> str:
        """Расшифровывает текст аффинным шифром по текущему модулю."""
        a_inv = self.find_inverse(a, self.modulus)
        if a_inv is None:
            return ""

        res = []
        for ch in text.lower():
            if ch in self.ALPHABET:
                y = self.ALPHABET[ch]
                x = (a_inv * (y - b)) % self.modulus
                res.append(self.ALPHABET_REVERSE[x])
        return "".join(res)

    def frequency_groups(self, text: str) -> List[Dict[str, Any]]:
        """Возвращает частоты групп символов с отождествлением ё/е и ъ/ь, сохраняя варианты исходных символов."""
        t = text.lower()
        group_counts: Counter[str] = Counter()
        group_variants: Dict[str, Set[str]] = {}

        for ch in t:
            if ch not in self.ALPHABET:
                continue
            if ch == "ё":
                gch = "е"
            elif ch == "ъ":
                gch = "ь"
            else:
                gch = ch
            group_counts[gch] += 1
            group_variants.setdefault(gch, set()).add(ch)

        total = sum(group_counts.values())
        if total == 0:
            return []

        items: List[Dict[str, Any]] = []
        for gch, cnt in group_counts.items():
            items.append(
                {
                    "symbol": gch,
                    "count": cnt,
                    "freq": cnt / total,
                    "variants": sorted(group_variants.get(gch, {gch})),
                }
            )
        items.sort(key=lambda d: d["freq"], reverse=True)
        return items

    def frequency_analysis(self, text: str) -> List[Tuple[str, float]]:
        """Возвращает список (символ, частота) по результатам группового анализа."""
        groups = self.frequency_groups(text)
        return [(d["symbol"], d["freq"]) for d in groups]

    def get_valid_a_values(self) -> List[int]:
        """Возвращает список допустимых значений a (взаимно простых с модулем)."""
        return [a for a in range(1, self.modulus) if math.gcd(a, self.modulus) == 1]

    def is_russian_text(self, text: str) -> bool:
        """Эвристика: проверяет, похож ли текст на русский."""
        t = self.normalize_for_frequency(text)

        common_bigrams = ["ст", "но", "то", "на", "ен", "ов", "ни", "ра", "во", "ко", "ть", "ет", "ит"]
        common_trigrams = ["ост", "сто", "ени", "ест", "ние", "ать", "про", "при", "пер", "под"]
        common_words = ["это", "что", "как", "для", "при", "она", "они", "все", "если", "когда", "только"]

        bigram_count = sum(1 for bg in common_bigrams if bg in t)
        trigram_count = sum(1 for tg in common_trigrams if tg in t)
        word_count = sum(1 for w in common_words if w in t)

        return bigram_count >= 3 or trigram_count >= 2 or word_count >= 1

    def score_text(self, text: str) -> float:
        """Оценивает качество текста (чем выше, тем вероятнее осмысленная расшифровка)."""
        t = self.normalize_for_frequency(text)
        score = 0.0

        freq_groups = self.frequency_groups(t)
        top_observed = freq_groups[:7]
        expected = dict(self.LETTER_FREQUENCIES)

        for item in top_observed:
            ch = item["symbol"]
            if ch in expected:
                score += 30.0 * (1.0 - abs(item["freq"] - expected[ch]))

        common_bigrams = [
            "ст", "но", "то", "на", "ен", "ов", "ни", "ра", "во", "ко",
            "ть", "ет", "ит", "ат", "ес", "ер", "ар", "ор", "ир", "ол"
        ]
        for bg in common_bigrams:
            score += t.count(bg) * 0.6

        common_trigrams = [
            "ост", "сто", "ени", "ест", "ова", "ние", "ите", "ель", "ать",
            "про", "при", "пер", "под", "над", "раз", "без"
        ]
        for tg in common_trigrams:
            score += t.count(tg) * 1.2

        common_words = [
            "это", "что", "как", "для", "при", "был", "она", "они", "все",
            "если", "после", "только", "также", "который", "когда"
        ]
        for w in common_words:
            score += t.count(w) * 2.5

        return score

    def crack_affine(
        self,
        ciphertext: str,
        progress_callback=None,
        confirmation_callback=None,
        log_callback=None,
        top_keep: int = 100,
        freq_cipher_top_groups: int = 5,
        freq_plain_top: int = 6,
        confirm_score_threshold: float = 55.0,
        auto_stop_score_threshold: float = 180.0,
        do_bruteforce_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        """Выполняет криптоанализ: частотные гипотезы, затем (опционально) перебор ключей. Возвращает список результатов.
        Parameters
        ----------
        ciphertext : str
            Зашифрованный текст для анализа.
        progress_callback : Callable, optional
            Функция для отображения прогресса (сообщение, процент, всего).
        confirmation_callback : Callable, optional
            Функция для подтверждения найденного решения.
        log_callback : Callable, optional
            Функция для логирования попыток дешифровки.
        top_keep : int, default=100
            Максимальное количество сохраняемых результатов.
        freq_cipher_top_groups : int, default=5
            Сколько групп частотности шифртекста использовать.
        freq_plain_top : int, default=6
            Сколько самых частых русских букв использовать.
        confirm_score_threshold : float, default=55.0
            Порог score для запроса подтверждения.
        auto_stop_score_threshold : float, default=180.0
            Порог score для автоостановки.
        do_bruteforce_fallback : bool, default=True
            Делать ли полный перебор при неудаче частотного анализа.

        Returns
        -------
        List[Dict[str, Any]]
            Отсортированный список результатов. Каждый результат содержит:
            - stage: 'frequency' или 'bruteforce'
            - hypothesis: описание гипотезы
            - key: найденный ключ (a, b)
            - decrypted: расшифрованный текст
            - score: оценка качества
            - looks_russian: похоже ли на русский
"""
        self.stop_cracking = False
        self.found_solution = False

        def log_try(payload: Dict[str, Any]) -> None:
            if not log_callback:
                return
            stage = payload.get("stage", "")
            hypothesis = payload.get("hypothesis", "")
            a, b = payload.get("key", (None, None))
            score = payload.get("score", 0.0)
            decrypted = payload.get("decrypted", "")
            log_callback(
                "\n".join(
                    [
                        "=== TRY ===",
                        f"stage: {stage}",
                        f"hypothesis: {hypothesis}",
                        f"a={a}, b={b}",
                        f"score={score:.4f}",
                        "text:",
                        decrypted,
                        "=== END ===",
                        "",
                    ]
                )
            )

        results: List[Dict[str, Any]] = []
        seen_keys: Set[Tuple[int, int]] = set()

        groups = self.frequency_groups(ciphertext)
        if not groups:
            return results

        cipher_group_list = groups[: max(1, freq_cipher_top_groups)]
        cipher_symbol_candidates: List[str] = []
        for d in cipher_group_list:
            cipher_symbol_candidates.extend(d["variants"])
        cipher_symbol_candidates = list(dict.fromkeys(cipher_symbol_candidates))

        most_frequent_russian = [ch for ch, _ in self.LETTER_FREQUENCIES]
        most_frequent_russian = most_frequent_russian[: max(2, freq_plain_top)]

        total_hypotheses = 0
        for i in range(len(cipher_symbol_candidates)):
            for j in range(i + 1, len(cipher_symbol_candidates)):
                total_hypotheses += (len(most_frequent_russian) * (len(most_frequent_russian) - 1))

        hypothesis_idx = 0

        for i in range(len(cipher_symbol_candidates)):
            if self.stop_cracking or self.found_solution:
                break
            for j in range(i + 1, len(cipher_symbol_candidates)):
                if self.stop_cracking or self.found_solution:
                    break

                c1 = cipher_symbol_candidates[i]
                c2 = cipher_symbol_candidates[j]
                y1 = self.ALPHABET[c1]
                y2 = self.ALPHABET[c2]

                for r1 in most_frequent_russian:
                    if self.stop_cracking or self.found_solution:
                        break
                    for r2 in most_frequent_russian:
                        if self.stop_cracking or self.found_solution:
                            break
                        if r1 == r2:
                            continue

                        hypothesis_idx += 1
                        x1 = self.ALPHABET[r1]
                        x2 = self.ALPHABET[r2]

                        key_solutions = self.solve_system(x1, y1, x2, y2, self.modulus)
                        if not key_solutions:
                            if progress_callback:
                                msg = (
                                    f"[Частотный этап] Гипотеза {hypothesis_idx}/{total_hypotheses}: "
                                    f"{c1}→{r1}, {c2}→{r2} | решений нет"
                                )
                                progress_callback(msg, int((hypothesis_idx / max(1, total_hypotheses)) * 50), 100)
                            continue

                        for a, b in key_solutions:
                            if self.stop_cracking or self.found_solution:
                                break
                            if math.gcd(a, self.modulus) != 1:
                                continue
                            if (a, b) in seen_keys:
                                continue
                            seen_keys.add((a, b))

                            decrypted = self.decrypt_affine(ciphertext, a, b)
                            score = self.score_text(decrypted)
                            ok = self.is_russian_text(decrypted)

                            result = {
                                "stage": "frequency",
                                "hypothesis": f"{c1}→{r1}, {c2}→{r2}",
                                "key": (a, b),
                                "decrypted": decrypted,
                                "score": score,
                                "looks_russian": ok,
                            }

                            log_try(result)

                            if progress_callback:
                                msg = (
                                    f"[Частотный этап] Гипотеза {hypothesis_idx}/{total_hypotheses}: "
                                    f"{c1}→{r1}, {c2}→{r2} | ключ a={a}, b={b} | score={score:.2f} "
                                    f"| {decrypted[:100]}{'...' if len(decrypted) > 100 else ''}"
                                )
                                progress_callback(msg, int((hypothesis_idx / max(1, total_hypotheses)) * 50), 100)

                            if ok:
                                results.append(result)
                                results.sort(key=lambda x: x["score"], reverse=True)
                                if len(results) > top_keep:
                                    results = results[:top_keep]

                            if confirmation_callback and ok and score >= confirm_score_threshold:
                                if confirmation_callback(result):
                                    self.found_solution = True
                                    results = [result] + [r for r in results if r != result]
                                    break

                            if (not confirmation_callback) and ok and score >= auto_stop_score_threshold:
                                self.found_solution = True
                                results = [result] + [r for r in results if r != result]
                                break

        if (not self.found_solution) and do_bruteforce_fallback and (not self.stop_cracking):
            valid_a = self.get_valid_a_values()
            total_keys = len(valid_a) * self.modulus
            attempt = 0

            for a in valid_a:
                if self.stop_cracking or self.found_solution:
                    break
                for b in range(self.modulus):
                    if self.stop_cracking or self.found_solution:
                        break

                    attempt += 1
                    if (a, b) in seen_keys:
                        continue
                    seen_keys.add((a, b))

                    decrypted = self.decrypt_affine(ciphertext, a, b)
                    score = self.score_text(decrypted)
                    ok = self.is_russian_text(decrypted)

                    result = {
                        "stage": "bruteforce",
                        "hypothesis": "перебор ключей",
                        "key": (a, b),
                        "decrypted": decrypted,
                        "score": score,
                        "looks_russian": ok,
                    }

                    log_try(result)

                    if progress_callback and attempt % 3 == 0:
                        msg = (
                            f"[Перебор ключей] {attempt}/{total_keys} | ключ a={a}, b={b} | score={score:.2f} "
                            f"| {decrypted[:100]}{'...' if len(decrypted) > 100 else ''}"
                        )
                        pct = 50 + int((attempt / max(1, total_keys)) * 50)
                        progress_callback(msg, min(pct, 100), 100)

                    if ok:
                        results.append(result)
                        results.sort(key=lambda x: x["score"], reverse=True)
                        if len(results) > top_keep:
                            results = results[:top_keep]

                    if confirmation_callback and ok and score >= confirm_score_threshold:
                        if confirmation_callback(result):
                            self.found_solution = True
                            results = [result] + [r for r in results if r != result]
                            break

                    if (not confirmation_callback) and ok and score >= auto_stop_score_threshold:
                        self.found_solution = True
                        results = [result] + [r for r in results if r != result]
                        break

        if self.found_solution and results:
            head = results[0]
            tail = [r for r in results[1:] if r != head]
            tail.sort(key=lambda x: x["score"], reverse=True)
            results = [head] + tail
        else:
            results.sort(key=lambda x: x["score"], reverse=True)

        return results


class AffineCipherGUI:
    """Графический интерфейс для модульной арифметики и криптоанализа аффинного шифра."""

    def __init__(self):
        """Инициализация GUI."""
        self.root = tk.Tk()
        self.root.title("Криптоанализ аффинного шифра")
        self.root.geometry("1000x750")

        self.cipher_analyzer = AffineCipherCryptoanalysis()
        self.crack_thread: Optional[threading.Thread] = None
        self.current_results: List[Dict[str, Any]] = []
        self.event_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.log_path: Optional[str] = None

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.math_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.math_frame, text="Модульная арифметика")
        self.create_math_tab()

        self.crack_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.crack_frame, text="Взлом шифра")
        self.create_crack_tab()

        self.root.after(40, self.process_queue)

    def create_math_tab(self) -> None:
        """Создает вкладку модульной арифметики."""
        gcd_frame = ttk.LabelFrame(self.math_frame, text="НОД и коэффициенты Безу", padding=10)
        gcd_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(gcd_frame, text="Число a:").grid(row=0, column=0, sticky="w")
        self.gcd_a = ttk.Entry(gcd_frame, width=10)
        self.gcd_a.grid(row=0, column=1, padx=5)

        ttk.Label(gcd_frame, text="Число b:").grid(row=0, column=2, sticky="w")
        self.gcd_b = ttk.Entry(gcd_frame, width=10)
        self.gcd_b.grid(row=0, column=3, padx=5)

        ttk.Button(gcd_frame, text="Вычислить", command=self.calculate_gcd).grid(row=0, column=4, padx=10)

        self.gcd_result = ttk.Label(gcd_frame, text="")
        self.gcd_result.grid(row=1, column=0, columnspan=5, pady=5)

        inverse_frame = ttk.LabelFrame(self.math_frame, text="Обратный элемент", padding=10)
        inverse_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(inverse_frame, text="Элемент a:").grid(row=0, column=0, sticky="w")
        self.inv_a = ttk.Entry(inverse_frame, width=10)
        self.inv_a.grid(row=0, column=1, padx=5)

        ttk.Label(inverse_frame, text="Модуль m:").grid(row=0, column=2, sticky="w")
        self.inv_m = ttk.Entry(inverse_frame, width=10)
        self.inv_m.grid(row=0, column=3, padx=5)
        self.inv_m.insert(0, "32")

        ttk.Button(inverse_frame, text="Найти обратный", command=self.find_inverse).grid(row=0, column=4, padx=10)

        self.inv_result = ttk.Label(inverse_frame, text="")
        self.inv_result.grid(row=1, column=0, columnspan=5, pady=5)

        congruence_frame = ttk.LabelFrame(self.math_frame, text="Решение сравнения ax ≡ b (mod m)", padding=10)
        congruence_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(congruence_frame, text="a:").grid(row=0, column=0, sticky="w")
        self.cong_a = ttk.Entry(congruence_frame, width=10)
        self.cong_a.grid(row=0, column=1, padx=5)

        ttk.Label(congruence_frame, text="b:").grid(row=0, column=2, sticky="w")
        self.cong_b = ttk.Entry(congruence_frame, width=10)
        self.cong_b.grid(row=0, column=3, padx=5)

        ttk.Label(congruence_frame, text="m:").grid(row=0, column=4, sticky="w")
        self.cong_m = ttk.Entry(congruence_frame, width=10)
        self.cong_m.grid(row=0, column=5, padx=5)
        self.cong_m.insert(0, "32")

        ttk.Button(congruence_frame, text="Решить", command=self.solve_congruence).grid(row=0, column=6, padx=10)

        self.cong_result = ttk.Label(congruence_frame, text="")
        self.cong_result.grid(row=1, column=0, columnspan=7, pady=5)

        system_frame = ttk.LabelFrame(self.math_frame, text="Решение системы сравнений", padding=10)
        system_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(system_frame, text="a₁x + y ≡ b₁ (mod m)").grid(row=0, column=0, columnspan=6, sticky="w")
        ttk.Label(system_frame, text="a₂x + y ≡ b₂ (mod m)").grid(row=1, column=0, columnspan=6, sticky="w")

        ttk.Label(system_frame, text="a₁:").grid(row=2, column=0, sticky="w")
        self.sys_a1 = ttk.Entry(system_frame, width=8)
        self.sys_a1.grid(row=2, column=1, padx=2)

        ttk.Label(system_frame, text="b₁:").grid(row=2, column=2, sticky="w")
        self.sys_b1 = ttk.Entry(system_frame, width=8)
        self.sys_b1.grid(row=2, column=3, padx=2)

        ttk.Label(system_frame, text="a₂:").grid(row=3, column=0, sticky="w")
        self.sys_a2 = ttk.Entry(system_frame, width=8)
        self.sys_a2.grid(row=3, column=1, padx=2)

        ttk.Label(system_frame, text="b₂:").grid(row=3, column=2, sticky="w")
        self.sys_b2 = ttk.Entry(system_frame, width=8)
        self.sys_b2.grid(row=3, column=3, padx=2)

        ttk.Label(system_frame, text="m:").grid(row=2, column=4, sticky="w")
        self.sys_m = ttk.Entry(system_frame, width=8)
        self.sys_m.grid(row=2, column=5, padx=2)
        self.sys_m.insert(0, "32")

        ttk.Button(system_frame, text="Решить систему", command=self.solve_system).grid(row=3, column=4, columnspan=2, padx=10)

        self.sys_result = ttk.Label(system_frame, text="")
        self.sys_result.grid(row=4, column=0, columnspan=6, pady=5)

    def create_crack_tab(self) -> None:
        """Создает вкладку взлома шифра."""
        input_frame = ttk.LabelFrame(self.crack_frame, text="Шифротекст", padding=10)
        input_frame.pack(fill="both", expand=True, padx=10, pady=5)

        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill="x")

        ttk.Button(button_frame, text="Загрузить из файла", command=self.load_ciphertext).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Вставить вариант 20", command=self.load_variant20).pack(side="left", padx=5)

        self.ciphertext_input = scrolledtext.ScrolledText(input_frame, height=4, wrap=tk.WORD)
        self.ciphertext_input.pack(fill="both", expand=True, pady=5)

        control_frame = ttk.Frame(self.crack_frame)
        control_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(control_frame, text="Частотный анализ", command=self.frequency_analysis).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Взломать (авто)", command=self.crack_cipher_auto).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Взломать (интерактивно)", command=self.crack_cipher_interactive).pack(side="left", padx=5)

        self.stop_button = ttk.Button(control_frame, text="Остановить", command=self.stop_cracking, state="disabled")
        self.stop_button.pack(side="left", padx=5)

        ttk.Button(control_frame, text="Сохранить результаты", command=self.save_results).pack(side="left", padx=5)

        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(self.crack_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", padx=10, pady=5)

        result_frame = ttk.LabelFrame(self.crack_frame, text="Результаты", padding=10)
        result_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.result_text = scrolledtext.ScrolledText(result_frame, height=15, wrap=tk.WORD)
        self.result_text.pack(fill="both", expand=True)

    def process_queue(self) -> None:
        """Обрабатывает события из очереди, обновляя GUI строго из главного потока."""
        try:
            while True:
                kind, payload = self.event_queue.get_nowait()
                if kind == "progress":
                    message, percent = payload
                    self.progress_var.set(int(percent))
                    self.result_text.insert(tk.END, message + "\n")
                    self.result_text.see(tk.END)
                elif kind == "done":
                    self.display_results(payload)
                elif kind == "confirm":
                    result, done_event, out = payload
                    out["confirmed"] = self.confirm_dialog(result)
                    done_event.set()
                elif kind == "error":
                    messagebox.showerror("Ошибка", str(payload))
        except queue.Empty:
            pass
        finally:
            self.root.after(40, self.process_queue)

    def calculate_gcd(self) -> None:
        """Вычисляет НОД и коэффициенты Безу."""
        try:
            a = int(self.gcd_a.get())
            b = int(self.gcd_b.get())

            g, u, v = self.cipher_analyzer.extended_gcd(a, b)

            result = f"НОД({a}, {b}) = {g}\n"
            result += f"Коэффициенты Безу: u = {u}, v = {v}\n"
            result += f"Проверка: {u} × {a} + {v} × {b} = {u * a + v * b} = {g}"
            self.gcd_result.config(text=result)
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные целые числа")

    def find_inverse(self) -> None:
        """Находит обратный элемент по модулю."""
        try:
            a = int(self.inv_a.get())
            m = int(self.inv_m.get())

            inv = self.cipher_analyzer.find_inverse(a, m)
            if inv is None:
                result = f"Обратный элемент не существует.\nНОД({a}, {m}) = {math.gcd(a, m)} ≠ 1"
            else:
                result = f"{a}⁻¹ ≡ {inv} (mod {m})\n"
                result += f"Проверка: {a} × {inv} ≡ {(a * inv) % m} (mod {m})"
            self.inv_result.config(text=result)
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные целые числа")

    def solve_congruence(self) -> None:
        """Решает сравнение a*x ≡ b (mod m) с учетом множественных решений."""
        try:
            a = int(self.cong_a.get())
            b = int(self.cong_b.get())
            m = int(self.cong_m.get())

            sol = self.cipher_analyzer.solve_congruence(a, b, m)
            if sol is None:
                g = math.gcd(a, m)
                result = f"Решений нет.\nНОД({a}, {m}) = {g}"
                if g > 1 and (b % g != 0):
                    result += f"\n{b} не делится на {g}"
            else:
                if len(sol) == 1:
                    result = f"Решение: x ≡ {sol[0]} (mod {m})"
                else:
                    result = f"Решений: {len(sol)}\n"
                    result += "x ≡ " + ", ".join(map(str, sol)) + f" (mod {m})"
            self.cong_result.config(text=result)
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные целые числа")

    def solve_system(self) -> None:
        """Решает систему сравнений и показывает все решения."""
        try:
            a1 = int(self.sys_a1.get())
            b1 = int(self.sys_b1.get())
            a2 = int(self.sys_a2.get())
            b2 = int(self.sys_b2.get())
            m = int(self.sys_m.get())

            sols = self.cipher_analyzer.solve_system(a1, b1, a2, b2, m)
            if sols is None:
                self.sys_result.config(text="Система не имеет решения")
                return

            lines = [f"Найдено решений: {len(sols)}"]
            show = sols[:12]
            for idx, (x, y) in enumerate(show, 1):
                lines.append(f"{idx}. x ≡ {x} (mod {m}), y ≡ {y} (mod {m})")

            if len(sols) > len(show):
                lines.append("...")

            self.sys_result.config(text="\n".join(lines))
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные целые числа")

    def load_ciphertext(self) -> None:
        """Загружает шифротекст из файла."""
        filename = filedialog.askopenfilename(
            title="Выберите файл с шифротекстом",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not filename:
            return
        try:
            with open(filename, "r", encoding="utf-8") as f:
                text = f.read()
            self.ciphertext_input.delete("1.0", tk.END)
            self.ciphertext_input.insert("1.0", text)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {e}")

    def load_variant20(self) -> None:
        """Вставляет шифротекст варианта 20."""
        variant20 = (
            "аоювэеъизпийищжюозйгдфифределеюищдеэлозйгаоювйозфоэыоужю"
            "лейфэисйезйаоюжаестизиъисоуифитлейфэчепийфжэоьюидофвйзиэидис"
            "ьийфжуизиржзжтизиъчхжийфоджзжиъэжх"
        )
        self.ciphertext_input.delete("1.0", tk.END)
        self.ciphertext_input.insert("1.0", variant20)

    def frequency_analysis(self) -> None:
        """Проводит частотный анализ с отождествлением ё/е и ъ/ь и выводит топ частот."""
        ciphertext = self.ciphertext_input.get("1.0", tk.END).strip()
        if not ciphertext:
            messagebox.showwarning("Предупреждение", "Введите шифротекст")
            return

        groups = self.cipher_analyzer.frequency_groups(ciphertext)
        if not groups:
            messagebox.showwarning("Предупреждение", "Нет символов алфавита для анализа")
            return

        total = sum(d["count"] for d in groups)

        result = "=== ЧАСТОТНЫЙ АНАЛИЗ ===\n\n"
        result += f"Длина текста (буквы алфавита): {total} символов\n\n"
        result += "Топ-10 самых частых (группы: ё/е, ъ/ь объединены):\n"
        result += "-" * 45 + "\n"

        for i, d in enumerate(groups[:10], 1):
            sym = d["symbol"]
            freq = d["freq"]
            cnt = d["count"]
            variants = ", ".join(d["variants"])
            if sym == "е":
                note = f"варианты: {variants} (ё/е)"
            elif sym == "ь":
                note = f"варианты: {variants} (ъ/ь)"
            else:
                note = f"варианты: {variants}"
            result += f"{i:2}. '{sym}' - {freq:.4f} ({cnt} раз), {note}\n"

        result += "\nДве самые частые группы:\n"
        result += f"1) '{groups[0]['symbol']}' ({groups[0]['count']} раз)\n"
        if len(groups) > 1:
            result += f"2) '{groups[1]['symbol']}' ({groups[1]['count']} раз)\n"

        result += "\nПодсказка: наиболее частые в русском тексте обычно 'о' и 'е'.\n"

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", result)

    def choose_log_file(self) -> Optional[str]:
        """Запрашивает у пользователя файл для логирования перебора ключей."""
        filename = filedialog.asksaveasfilename(
            title="Выберите файл для логов перебора (ключи + расшифровки)",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        return filename or None

    def crack_cipher_auto(self) -> None:
        """Запускает автоматический криптоанализ с логированием попыток в файл."""
        ciphertext = self.ciphertext_input.get("1.0", tk.END).strip()
        if not ciphertext:
            messagebox.showwarning("Предупреждение", "Введите шифротекст")
            return

        log_path = self.choose_log_file()
        if not log_path:
            messagebox.showwarning("Предупреждение", "Без файла логов запуск не выполнен")
            return
        self.log_path = log_path

        self.stop_button.config(state="normal")
        self.progress_var.set(0)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", "Автоматический взлом начат...\n\n")

        def worker() -> None:
            try:
                with open(self.log_path, "w", encoding="utf-8") as logf:
                    logf.write("=== LOG START ===\n\n")

                    def log_callback(s: str) -> None:
                        logf.write(s + "\n")
                        logf.flush()

                    def progress_callback(message: str, percent: int, _max: int) -> None:
                        self.event_queue.put(("progress", (message, percent)))

                    results = self.cipher_analyzer.crack_affine(
                        ciphertext=ciphertext,
                        progress_callback=progress_callback,
                        confirmation_callback=None,
                        log_callback=log_callback,
                        do_bruteforce_fallback=True,
                    )

                    logf.write("\n=== LOG END ===\n")
                self.event_queue.put(("done", results))
            except Exception as e:
                self.event_queue.put(("error", f"{e}"))
                self.event_queue.put(("done", []))

        self.crack_thread = threading.Thread(target=worker, daemon=True)
        self.crack_thread.start()

    def crack_cipher_interactive(self) -> None:
        """Запускает интерактивный криптоанализ: при сильном кандидате запрашивает подтверждение, ведет лог в файл."""
        ciphertext = self.ciphertext_input.get("1.0", tk.END).strip()
        if not ciphertext:
            messagebox.showwarning("Предупреждение", "Введите шифротекст")
            return

        log_path = self.choose_log_file()
        if not log_path:
            messagebox.showwarning("Предупреждение", "Без файла логов запуск не выполнен")
            return
        self.log_path = log_path

        self.stop_button.config(state="normal")
        self.progress_var.set(0)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", "Интерактивный взлом начат...\n\n")

        def confirm_threadsafe(result: Dict[str, Any]) -> bool:
            done_event = threading.Event()
            out: Dict[str, Any] = {"confirmed": False}
            self.event_queue.put(("confirm", (result, done_event, out)))
            done_event.wait()
            return bool(out["confirmed"])

        def worker() -> None:
            try:
                with open(self.log_path, "w", encoding="utf-8") as logf:
                    logf.write("=== LOG START ===\n\n")

                    def log_callback(s: str) -> None:
                        logf.write(s + "\n")
                        logf.flush()

                    def progress_callback(message: str, percent: int, _max: int) -> None:
                        self.event_queue.put(("progress", (message, percent)))

                    results = self.cipher_analyzer.crack_affine(
                        ciphertext=ciphertext,
                        progress_callback=progress_callback,
                        confirmation_callback=confirm_threadsafe,
                        log_callback=log_callback,
                        do_bruteforce_fallback=True,
                    )

                    logf.write("\n=== LOG END ===\n")
                self.event_queue.put(("done", results))
            except Exception as e:
                self.event_queue.put(("error", f"{e}"))
                self.event_queue.put(("done", []))

        self.crack_thread = threading.Thread(target=worker, daemon=True)
        self.crack_thread.start()

    def confirm_dialog(self, result: Dict[str, Any]) -> bool:
        """Показывает диалог подтверждения кандидата расшифровки. Выполняется только в главном потоке."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Подтверждение результата")
        dialog.geometry("720x440")
        dialog.grab_set()

        stage = result.get("stage", "")
        a, b = result.get("key", (None, None))
        hypothesis = result.get("hypothesis", "")
        score = float(result.get("score", 0.0))
        decrypted = result.get("decrypted", "")

        ttk.Label(dialog, text="Найден возможный вариант расшифровки:").pack(pady=6)
        ttk.Label(dialog, text=f"Этап: {stage}").pack()
        ttk.Label(dialog, text=f"Ключ: a={a}, b={b}").pack()
        ttk.Label(dialog, text=f"Гипотеза: {hypothesis}").pack()
        ttk.Label(dialog, text=f"Оценка: {score:.2f}").pack()

        text_frame = ttk.Frame(dialog)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)

        text_widget = scrolledtext.ScrolledText(text_frame, height=12, wrap=tk.WORD)
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("1.0", decrypted)
        text_widget.config(state="disabled")

        confirmed = {"value": False}

        def on_yes() -> None:
            confirmed["value"] = True
            dialog.destroy()

        def on_no() -> None:
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Это правильный текст!", command=on_yes).pack(side="left", padx=6)
        ttk.Button(button_frame, text="Продолжить поиск", command=on_no).pack(side="left", padx=6)

        dialog.wait_window()
        return bool(confirmed["value"])

    def display_results(self, results: List[Dict[str, Any]]) -> None:
        """Отображает результаты по завершении взлома."""
        self.stop_button.config(state="disabled")
        self.progress_var.set(100)

        if not results:
            self.result_text.insert(tk.END, "\nНе удалось получить кандидаты расшифровки.\n")
            self.current_results = []
            return

        self.result_text.insert(tk.END, "\n" + "=" * 70 + "\n")
        if self.cipher_analyzer.found_solution:
            self.result_text.insert(tk.END, "✓ НАЙДЕНО ПОДТВЕРЖДЕННОЕ РЕШЕНИЕ!\n\n")

        self.result_text.insert(tk.END, "ЛУЧШИЕ РЕЗУЛЬТАТЫ:\n\n")

        for i, r in enumerate(results[:5], 1):
            if i == 1 and self.cipher_analyzer.found_solution:
                self.result_text.insert(tk.END, "*** ПОДТВЕРЖДЕННЫЙ РЕЗУЛЬТАТ ***\n")
            a, b = r["key"]
            self.result_text.insert(tk.END, f"Результат {i}:\n")
            self.result_text.insert(tk.END, f"Этап: {r.get('stage', '')}\n")
            self.result_text.insert(tk.END, f"Гипотеза: {r.get('hypothesis', '')}\n")
            self.result_text.insert(tk.END, f"Ключ: a={a}, b={b}\n")
            self.result_text.insert(tk.END, f"Оценка: {r.get('score', 0.0):.2f}\n")
            text_preview = r.get("decrypted", "")
            self.result_text.insert(tk.END, f"Текст: {text_preview[:180]}{'...' if len(text_preview) > 180 else ''}\n")
            self.result_text.insert(tk.END, "-" * 70 + "\n")

        self.current_results = results

    def stop_cracking(self) -> None:
        """Останавливает процесс взлома."""
        self.cipher_analyzer.stop_cracking = True
        self.stop_button.config(state="disabled")
        self.result_text.insert(tk.END, "\n*** Процесс остановлен пользователем ***\n")

    def save_results(self) -> None:
        """Сохраняет отображаемые результаты и полные тексты лучших кандидатов в файл."""
        filename = filedialog.asksaveasfilename(
            title="Сохранить результаты",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not filename:
            return
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.result_text.get("1.0", tk.END))

                if self.current_results:
                    f.write("\n\n" + "=" * 70 + "\n")
                    f.write("ПОЛНЫЕ ТЕКСТЫ ЛУЧШИХ РЕЗУЛЬТАТОВ:\n\n")

                    for i, r in enumerate(self.current_results[:3], 1):
                        if i == 1 and self.cipher_analyzer.found_solution:
                            f.write("*** ПОДТВЕРЖДЕННОЕ РЕШЕНИЕ ***\n")
                        a, b = r["key"]
                        f.write(f"Результат {i}:\n")
                        f.write(f"Этап: {r.get('stage', '')}\n")
                        f.write(f"Ключ: a={a}, b={b}\n")
                        f.write(f"Гипотеза: {r.get('hypothesis', '')}\n")
                        f.write(f"Оценка: {r.get('score', 0.0):.2f}\n")
                        f.write("Полный текст:\n")
                        f.write(r.get("decrypted", "") + "\n")
                        f.write("\n" + "=" * 70 + "\n")

                if self.log_path:
                    f.write("\n\n" + "=" * 70 + "\n")
                    f.write(f"Файл логов перебора: {self.log_path}\n")

            messagebox.showinfo("Успех", f"Результаты сохранены в {filename}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

    def run(self) -> None:
        """Запускает главный цикл приложения."""
        self.root.mainloop()


if __name__ == "__main__":
    app = AffineCipherGUI()
    app.run()