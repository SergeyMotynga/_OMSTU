import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import random
import json
import math
from typing import List, Tuple, Optional, Dict, Any


class RSACipher:
    """RSA-шифратор: ключи, кодирование текста, блоки, шифрование/расшифрование."""

    ALPHABET: Dict[str, int] = {
        "А": 10, "Б": 11, "В": 12, "Г": 13, "Д": 14, "Е": 15, "Ж": 16, "З": 17, "И": 18, "Й": 19,
        "К": 20, "Л": 21, "М": 22, "Н": 23, "О": 24, "П": 25, "Р": 26, "С": 27, "Т": 28, "У": 29,
        "Ф": 30, "Х": 31, "Ц": 32, "Ч": 33, "Ш": 34, "Щ": 35, "Ъ": 36, "Ы": 37, "Ь": 38, "Э": 39,
        "Ю": 40, "Я": 41, " ": 99,
    }

    REVERSE_ALPHABET: Dict[int, str] = {v: k for k, v in ALPHABET.items()}

    def __init__(self) -> None:
        self.p: Optional[int] = None
        self.q: Optional[int] = None
        self.n: Optional[int] = None
        self.phi: Optional[int] = None
        self.keys: List[Tuple[int, int]] = []
        self.last_plain_digits: str = ""
        self.last_blocks: List[Tuple[int, int]] = []
        self.last_cipher_blocks: List[int] = []

    def is_prime(self, n: int) -> bool:
        """Проверка простоты."""
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False
        r = int(n ** 0.5)
        for i in range(3, r + 1, 2):
            if n % i == 0:
                return False
        return True

    def set_primes(self, p: int, q: int) -> Tuple[int, int]:
        """Устанавливает p, q и вычисляет n и φ(n)."""
        if not self.is_prime(p) or not self.is_prime(q):
            raise ValueError("p и q должны быть простыми числами")
        if p == q:
            raise ValueError("p и q должны быть различными")
        self.p = p
        self.q = q
        self.n = p * q
        self.phi = (p - 1) * (q - 1)
        return self.n, self.phi

    def extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        """Расширенный алгоритм Евклида: возвращает (g, x, y), где ax + by = g = gcd(a, b)."""
        if a == 0:
            return b, 0, 1
        g, x1, y1 = self.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return g, x, y

    def mod_inverse(self, e: int, phi: int) -> int:
        """Возвращает d = e^{-1} mod phi; бросает исключение, если обратного нет."""
        g, x, _ = self.extended_gcd(e, phi)
        if g != 1:
            raise ValueError("e не имеет обратного по модулю φ(n)")
        return x % phi

    def fast_pow(self, base: int, exp: int, mod: int) -> int:
        """Бинарное возведение в степень по модулю."""
        result = 1
        base %= mod
        while exp > 0:
            if exp & 1:
                result = (result * base) % mod
            base = (base * base) % mod
            exp >>= 1
        return result

    def generate_keys_for_primes(self, p: int, q: int, count: int = 3) -> List[Tuple[int, int]]:
        """Генерирует минимум 3 пары (e, d) для заданных p, q."""
        count = max(3, int(count))
        _, phi = self.set_primes(p, q)

        keys: List[Tuple[int, int]] = []
        seen_e = set()

        preferred_e = [3, 5, 17, 257, 65537]
        random_budget = 20000

        def try_add_e(e: int) -> None:
            if e in seen_e:
                return
            if e <= 1 or e >= phi:
                return
            if math.gcd(e, phi) != 1:
                return
            d = self.mod_inverse(e, phi)
            if d == e:
                return
            keys.append((e, d))
            seen_e.add(e)

        for e in preferred_e:
            if len(keys) >= count:
                break
            try_add_e(e)

        attempts = 0
        while len(keys) < count and attempts < random_budget:
            e = random.randint(2, phi - 1)
            if e not in seen_e and math.gcd(e, phi) == 1:
                try:
                    d = self.mod_inverse(e, phi)
                except ValueError:
                    attempts += 1
                    continue
                if d != e:
                    keys.append((e, d))
                    seen_e.add(e)
            attempts += 1

        if len(keys) < count:
            raise ValueError("Не удалось сгенерировать требуемое число пар ключей")

        self.keys = keys
        return keys

    def add_custom_key(self, e: int) -> Tuple[int, int]:
        """Добавляет пользовательское e и вычисляет d; требует предварительно заданные p, q."""
        if self.phi is None or self.n is None:
            raise ValueError("Сначала задайте p и q и сгенерируйте параметры")
        if e <= 1 or e >= self.phi:
            raise ValueError(f"e должно быть в диапазоне [2; {self.phi - 1}]")
        if math.gcd(e, self.phi) != 1:
            raise ValueError("e должно быть взаимно простым с φ(n)")
        d = self.mod_inverse(e, self.phi)
        if any(existing_e == e for existing_e, _ in self.keys):
            raise ValueError("Ключ с таким e уже существует")
        self.keys.append((e, d))
        return e, d

    def normalize_text(self, text: str) -> str:
        """Нормализация входа: верхний регистр; 'Ё' заменяется на 'Е'; табы/переводы строк заменяются на пробел."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\t", " ").replace("\n", " ")
        text = text.upper().replace("Ё", "Е")
        return text

    def text_to_digits(self, text: str) -> str:
        """Кодирует текст в строку цифр по таблице замен; бросает исключение при недопустимых символах."""
        text = self.normalize_text(text)
        bad = sorted({ch for ch in text if ch not in self.ALPHABET})
        if bad:
            raise ValueError(f"Недопустимые символы: {' '.join(bad)}")
        return "".join(f"{self.ALPHABET[ch]:02d}" for ch in text)

    def split_into_blocks(self, digit_string: str, n: int) -> List[Tuple[int, int]]:
        """Разбивает строку цифр на блоки M < n, гарантируя отсутствие ведущего нуля у каждого блока."""
        if n <= 1:
            raise ValueError("n должно быть больше 1")
        if not digit_string:
            return []
        if digit_string[0] == "0":
            raise ValueError("Цифровая строка не должна начинаться с 0")

        segments: List[str] = []
        i = 0
        s = digit_string

        def shift_boundary_left() -> None:
            nonlocal i
            if not segments:
                raise ValueError("Невозможно устранить ведущий ноль у первого блока")
            prev = segments.pop()
            if len(prev) <= 1:
                raise ValueError("Невозможно устранить ведущий ноль: предыдущий блок слишком короткий")
            segments.append(prev[:-1])
            i -= 1

        while i < len(s):
            while i < len(s) and s[i] == "0":
                shift_boundary_left()

            j = i + 1
            last_valid = None
            while j <= len(s):
                cand = s[i:j]
                if int(cand) >= n:
                    break
                last_valid = j
                j += 1

            if last_valid is None:
                raise ValueError("Невозможно сформировать блок < n (слишком маленький модуль n)")

            seg = s[i:last_valid]
            if seg[0] == "0":
                raise ValueError("Внутренняя ошибка: блок с ведущим нулём")
            segments.append(seg)
            i = last_valid

        blocks = [(int(seg), len(seg)) for seg in segments if seg != ""]
        if not blocks:
            raise ValueError("Не удалось сформировать блоки")
        return blocks

    def encrypt_blocks(self, blocks: List[Tuple[int, int]], e: int) -> List[int]:
        """Шифрует блоки M по RSA: C = M^e mod n."""
        if self.n is None:
            raise ValueError("Не задан модуль n")
        cipher = []
        for m, _ in blocks:
            if m < 0 or m >= self.n:
                raise ValueError("Найден блок вне диапазона [0; n-1]")
            cipher.append(self.fast_pow(m, e, self.n))
        self.last_blocks = blocks
        self.last_cipher_blocks = cipher
        return cipher

    def decrypt_blocks(self, cipher_blocks: List[int], d: int) -> List[int]:
        """Расшифровывает блоки C по RSA: M = C^d mod n."""
        if self.n is None:
            raise ValueError("Не задан модуль n")
        return [self.fast_pow(c, d, self.n) for c in cipher_blocks]

    from typing import List

    def digits_to_text(self, decrypted_numbers: List[int], lengths: List[int]) -> str:
        """
        Декодирует список чисел в текст с учётом длин исходных блоков; 
        не падает при неверном ключе.
        """
        if len(decrypted_numbers) != len(lengths):
            raise ValueError("Количество блоков и количество длин не совпадает")

        parts: List[str] = []
        for num, length in zip(decrypted_numbers, lengths):
            s = str(int(num))
            
            if len(s) < length:
                s = s.zfill(length)
            elif len(s) > length:
                s = s[-length:]
                
            parts.append(s)

        digits = "".join(parts)
        trailing_unknown = False

        # Если общее количество цифр нечетное, отбрасываем последнюю
        if len(digits) % 2 == 1:
            digits = digits[:-1]
            trailing_unknown = True

        out: List[str] = []
        for i in range(0, len(digits), 2):
            code = int(digits[i : i + 2])
            # Используем .get(), чтобы подставить "?", если код символа не найден
            out.append(self.REVERSE_ALPHABET.get(code, "?"))

        if trailing_unknown:
            out.append("?")

        return "".join(out)

    def export_keys(self) -> Dict[str, Any]:
        """Экспортирует параметры и ключи в словарь для сохранения в JSON."""
        if self.p is None or self.q is None or self.n is None or self.phi is None:
            raise ValueError("Параметры p, q не заданы")
        return {
            "p": self.p,
            "q": self.q,
            "n": self.n,
            "phi": self.phi,
            "keys": [{"e": e, "d": d} for e, d in self.keys],
        }

    def import_keys(self, data: Dict[str, Any]) -> None:
        """Импортирует параметры и ключи из словаря (JSON)."""
        p = int(data["p"])
        q = int(data["q"])
        self.set_primes(p, q)

        keys_raw = data.get("keys", [])
        keys: List[Tuple[int, int]] = []
        for item in keys_raw:
            e = int(item["e"])
            d = int(item["d"])
            keys.append((e, d))
        self.keys = keys

    def export_package(self) -> Dict[str, Any]:
        """Экспортирует полный пакет: параметры, ключи и последний результат шифрования."""
        base = self.export_keys()
        base.update(
            {
                "last_plain_digits": self.last_plain_digits,
                "blocks": [{"m": m, "len": ln} for m, ln in self.last_blocks],
                "cipher_blocks": list(self.last_cipher_blocks),
            }
        )
        return base

    def import_package(self, data: Dict[str, Any]) -> None:
        """Импортирует полный пакет: параметры, ключи, блоки и шифротекст."""
        self.import_keys(data)
        self.last_plain_digits = str(data.get("last_plain_digits", ""))

        blocks_raw = data.get("blocks", [])
        self.last_blocks = [(int(b["m"]), int(b["len"])) for b in blocks_raw]

        cipher_raw = data.get("cipher_blocks", [])
        self.last_cipher_blocks = [int(x) for x in cipher_raw]


class RSAGui:
    """GUI-приложение для лабораторной: генерация ключей, шифрование/расшифрование, загрузка/сохранение файлов."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Лабораторная работа №4: Алгоритм RSA")
        self.root.geometry("1000x720")

        self.rsa = RSACipher()

        self.create_menu()
        self.create_widgets()
        self.load_variant20()

    def create_menu(self) -> None:
        """Создает меню приложения с функциями загрузки/выгрузки."""
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Открыть текст…", command=self.load_plaintext_from_file)
        file_menu.add_command(label="Сохранить текст…", command=self.save_plaintext_to_file)
        file_menu.add_separator()
        file_menu.add_command(label="Сохранить ключи…", command=self.save_keys_to_file)
        file_menu.add_command(label="Загрузить ключи…", command=self.load_keys_from_file)
        file_menu.add_separator()
        file_menu.add_command(label="Сохранить пакет шифрования…", command=self.save_package_to_file)
        file_menu.add_command(label="Загрузить пакет…", command=self.load_package_from_file)
        file_menu.add_separator()
        file_menu.add_command(label="Сохранить расшифрованный текст…", command=self.save_decrypted_text_to_file)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.destroy)

        menubar.add_cascade(label="Файл", menu=file_menu)
        self.root.config(menu=menubar)

    def create_widgets(self) -> None:
        """Создает вкладки и элементы интерфейса."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.key_frame = ttk.Frame(self.notebook)
        self.encrypt_frame = ttk.Frame(self.notebook)
        self.decrypt_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.key_frame, text="Генерация ключей")
        self.notebook.add(self.encrypt_frame, text="Шифрование")
        self.notebook.add(self.decrypt_frame, text="Расшифрование")

        self.create_key_tab()
        self.create_encrypt_tab()
        self.create_decrypt_tab()

    def create_key_tab(self) -> None:
        """Вкладка генерации ключей и параметров."""
        input_frame = ttk.LabelFrame(self.key_frame, text="Параметры RSA", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)

        param_frame = ttk.Frame(input_frame)
        param_frame.pack()

        ttk.Label(param_frame, text="p:").grid(row=0, column=0, padx=5)
        self.p_entry = ttk.Entry(param_frame, width=10)
        self.p_entry.grid(row=0, column=1, padx=5)

        ttk.Label(param_frame, text="q:").grid(row=0, column=2, padx=5)
        self.q_entry = ttk.Entry(param_frame, width=10)
        self.q_entry.grid(row=0, column=3, padx=5)

        ttk.Button(param_frame, text="Вариант 20", command=self.load_variant20).grid(row=0, column=4, padx=10)
        ttk.Button(param_frame, text="Проверить", command=self.check_primes).grid(row=0, column=5, padx=5)

        self.params_info = ttk.Label(input_frame, text="", foreground="blue")
        self.params_info.pack(pady=5)

        gen_frame = ttk.LabelFrame(self.key_frame, text="Управление ключами", padding=10)
        gen_frame.pack(fill="x", padx=10, pady=5)

        button_frame = ttk.Frame(gen_frame)
        button_frame.pack()

        ttk.Label(button_frame, text="Количество (≥ 3):").pack(side="left", padx=5)
        self.key_count = ttk.Spinbox(button_frame, from_=3, to=20, width=5)
        self.key_count.set(3)
        self.key_count.pack(side="left", padx=5)

        ttk.Button(button_frame, text="Генерировать", command=self.generate_keys).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Добавить свой (e)", command=self.add_custom_key).pack(side="left", padx=5)

        keys_frame = ttk.LabelFrame(self.key_frame, text="Сгенерированные ключи", padding=10)
        keys_frame.pack(fill="both", expand=True, padx=10, pady=5)

        columns = ("№", "e (открытый)", "d (закрытый)", "ed mod φ(n)")
        self.key_tree = ttk.Treeview(keys_frame, columns=columns, show="headings", height=12)

        for col in columns:
            self.key_tree.heading(col, text=col)

        self.key_tree.column("№", width=50, anchor="center")
        self.key_tree.column("e (открытый)", width=260, anchor="center")
        self.key_tree.column("d (закрытый)", width=260, anchor="center")
        self.key_tree.column("ed mod φ(n)", width=120, anchor="center")

        self.key_tree.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(keys_frame, orient="vertical", command=self.key_tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.key_tree.configure(yscrollcommand=scrollbar.set)

    def create_encrypt_tab(self) -> None:
        """Вкладка шифрования текста."""
        input_frame = ttk.LabelFrame(self.encrypt_frame, text="Открытый текст", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)

        self.plaintext = scrolledtext.ScrolledText(input_frame, height=4, wrap=tk.WORD)
        self.plaintext.pack(fill="x")

        control_frame = ttk.Frame(self.encrypt_frame)
        control_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(control_frame, text="Ключ для шифрования:").pack(side="left", padx=5)
        self.encrypt_key = ttk.Combobox(control_frame, width=30, state="readonly")
        self.encrypt_key.pack(side="left", padx=5)

        ttk.Button(control_frame, text="Шифровать", command=self.encrypt_text).pack(side="left", padx=10)

        result_frame = ttk.LabelFrame(self.encrypt_frame, text="Процесс шифрования", padding=10)
        result_frame.pack(fill="both", expand=True, padx=10, pady=5)

        ttk.Label(result_frame, text="Цифровое представление:").pack(anchor="w")
        self.digit_display = scrolledtext.ScrolledText(result_frame, height=2, wrap=tk.WORD)
        self.digit_display.pack(fill="x", pady=5)

        ttk.Label(result_frame, text="Разбиение на блоки (M, длина):").pack(anchor="w")
        self.blocks_display = scrolledtext.ScrolledText(result_frame, height=5, wrap=tk.WORD)
        self.blocks_display.pack(fill="x", pady=5)

        ttk.Label(result_frame, text="Зашифрованные блоки:").pack(anchor="w")
        self.encrypted_display = scrolledtext.ScrolledText(result_frame, height=4, wrap=tk.WORD)
        self.encrypted_display.pack(fill="x", pady=5)

        info_frame = ttk.LabelFrame(result_frame, text="Для расшифровки потребуется", padding=5)
        info_frame.pack(fill="x", pady=5)

        self.decrypt_info = ttk.Label(info_frame, text="", foreground="red")
        self.decrypt_info.pack()

    def create_decrypt_tab(self) -> None:
        """Вкладка расшифрования блоков."""
        input_frame = ttk.LabelFrame(self.decrypt_frame, text="Зашифрованные блоки", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(input_frame, text="Введите блоки через пробел:").pack(anchor="w")
        self.ciphertext = scrolledtext.ScrolledText(input_frame, height=3, wrap=tk.WORD)
        self.ciphertext.pack(fill="x", pady=5)

        ttk.Button(input_frame, text="Использовать последний результат", command=self.use_last_encrypted).pack()

        lengths_frame = ttk.LabelFrame(self.decrypt_frame, text="Длины исходных блоков", padding=10)
        lengths_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(lengths_frame, text="Введите длины через пробел:").pack(anchor="w")
        self.lengths_entry = ttk.Entry(lengths_frame, width=50)
        self.lengths_entry.pack(fill="x", pady=5)

        ttk.Button(lengths_frame, text="Использовать сохраненные", command=self.use_saved_lengths).pack()

        control_frame = ttk.Frame(self.decrypt_frame)
        control_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(control_frame, text="Ключ для расшифрования:").pack(side="left", padx=5)
        self.decrypt_key = ttk.Combobox(control_frame, width=30, state="readonly")
        self.decrypt_key.pack(side="left", padx=5)

        ttk.Button(control_frame, text="Расшифровать", command=self.decrypt_text).pack(side="left", padx=10)

        result_frame = ttk.LabelFrame(self.decrypt_frame, text="Результат", padding=10)
        result_frame.pack(fill="both", expand=True, padx=10, pady=5)

        ttk.Label(result_frame, text="Расшифрованные блоки (M, длина):").pack(anchor="w")
        self.decrypted_blocks = scrolledtext.ScrolledText(result_frame, height=4, wrap=tk.WORD)
        self.decrypted_blocks.pack(fill="x", pady=5)

        ttk.Label(result_frame, text="Восстановленный текст:").pack(anchor="w")
        self.decrypted_text = scrolledtext.ScrolledText(result_frame, height=5, wrap=tk.WORD)
        self.decrypted_text.pack(fill="x", pady=5)

    def load_variant20(self) -> None:
        """Устанавливает p и q для варианта 20."""
        self.p_entry.delete(0, tk.END)
        self.p_entry.insert(0, "191")
        self.q_entry.delete(0, tk.END)
        self.q_entry.insert(0, "337")
        self.check_primes()

    def check_primes(self) -> None:
        """Проверяет простоту p и q и отображает n и φ(n)."""
        try:
            p = int(self.p_entry.get())
            q = int(self.q_entry.get())
            p_prime = self.rsa.is_prime(p)
            q_prime = self.rsa.is_prime(q)
            if p_prime and q_prime and p != q:
                n = p * q
                phi = (p - 1) * (q - 1)
                self.params_info.config(text=f"✓ Оба числа простые. n = {n}, φ(n) = {phi}", foreground="green")
            else:
                errors = []
                if not p_prime:
                    errors.append(f"p={p} не простое")
                if not q_prime:
                    errors.append(f"q={q} не простое")
                if p == q:
                    errors.append("p и q должны быть различными")
                self.params_info.config(text=", ".join(errors), foreground="red")
        except ValueError:
            self.params_info.config(text="Введите корректные числа", foreground="red")

    def update_key_table(self) -> None:
        """Обновляет таблицу ключей."""
        for item in self.key_tree.get_children():
            self.key_tree.delete(item)
        phi = self.rsa.phi or 0
        for i, (e, d) in enumerate(self.rsa.keys, 1):
            check = (e * d) % phi if phi else ""
            self.key_tree.insert("", "end", values=(i, e, d, check))

    def update_key_combos(self) -> None:
        """Обновляет выпадающие списки ключей."""
        if not self.rsa.keys:
            self.encrypt_key["values"] = []
            self.decrypt_key["values"] = []
            return
        enc_vals = [f"e={e}" for e, _ in self.rsa.keys]
        dec_vals = [f"d={d} (e={e})" for e, d in self.rsa.keys]
        self.encrypt_key["values"] = enc_vals
        self.decrypt_key["values"] = dec_vals
        self.encrypt_key.current(0)
        self.decrypt_key.current(0)

    def generate_keys(self) -> None:
        """Генерирует пары ключей и обновляет UI."""
        try:
            p = int(self.p_entry.get())
            q = int(self.q_entry.get())
            count = int(self.key_count.get())
            keys = self.rsa.generate_keys_for_primes(p, q, count)
            self.update_key_table()
            self.update_key_combos()
            messagebox.showinfo("Успех", f"Сгенерировано {len(keys)} пар ключей")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def add_custom_key(self) -> None:
        """Диалог добавления пользовательского e."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Добавить ключ")
        dialog.geometry("320x140")
        dialog.grab_set()

        ttk.Label(dialog, text="Введите e:").pack(pady=10)
        e_entry = ttk.Entry(dialog, width=24)
        e_entry.pack(pady=5)

        def add() -> None:
            try:
                e = int(e_entry.get())
                e_val, d_val = self.rsa.add_custom_key(e)
                self.update_key_table()
                self.update_key_combos()
                messagebox.showinfo("Успех", f"Добавлен ключ: e={e_val}, d={d_val}")
                dialog.destroy()
            except Exception as ex:
                messagebox.showerror("Ошибка", str(ex))

        ttk.Button(dialog, text="Добавить", command=add).pack(pady=10)

    def encrypt_text(self) -> None:
        """Шифрует текст из поля ввода."""
        text = self.plaintext.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Предупреждение", "Введите текст")
            return
        if not self.rsa.keys:
            messagebox.showwarning("Предупреждение", "Сначала сгенерируйте или загрузите ключи")
            return
        if self.rsa.n is None:
            messagebox.showwarning("Предупреждение", "Параметры n не заданы")
            return

        try:
            key_idx = self.encrypt_key.current()
            if key_idx < 0:
                raise ValueError("Выберите ключ для шифрования")
            e, _ = self.rsa.keys[key_idx]

            digits = self.rsa.text_to_digits(text)
            self.rsa.last_plain_digits = digits
            self.digit_display.delete("1.0", tk.END)
            self.digit_display.insert("1.0", digits)

            blocks = self.rsa.split_into_blocks(digits, self.rsa.n)
            blocks_str = "".join([f"M{i + 1} = {m} (длина: {ln})\n" for i, (m, ln) in enumerate(blocks)])
            self.blocks_display.delete("1.0", tk.END)
            self.blocks_display.insert("1.0", blocks_str)

            encrypted = self.rsa.encrypt_blocks(blocks, e)
            self.encrypted_display.delete("1.0", tk.END)
            self.encrypted_display.insert("1.0", " ".join(map(str, encrypted)))

            lengths = [ln for _, ln in blocks]
            info = f"Блоки: {' '.join(map(str, encrypted))}\nДлины: {' '.join(map(str, lengths))}\nМодуль n: {self.rsa.n}"
            self.decrypt_info.config(text=info)

            messagebox.showinfo("Успех", "Текст зашифрован")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def use_last_encrypted(self) -> None:
        """Копирует последний шифротекст в поле расшифрования."""
        if self.rsa.last_cipher_blocks:
            self.ciphertext.delete("1.0", tk.END)
            self.ciphertext.insert("1.0", " ".join(map(str, self.rsa.last_cipher_blocks)))
        else:
            encrypted_text = self.encrypted_display.get("1.0", tk.END).strip()
            if encrypted_text:
                self.ciphertext.delete("1.0", tk.END)
                self.ciphertext.insert("1.0", encrypted_text)

    def use_saved_lengths(self) -> None:
        """Копирует сохраненные длины блоков в поле расшифрования."""
        if self.rsa.last_blocks:
            lengths = [ln for _, ln in self.rsa.last_blocks]
            self.lengths_entry.delete(0, tk.END)
            self.lengths_entry.insert(0, " ".join(map(str, lengths)))

    def decrypt_text(self) -> None:
        """Расшифровывает блоки из поля ввода."""
        cipher_str = self.ciphertext.get("1.0", tk.END).strip()
        lengths_str = self.lengths_entry.get().strip()

        if not cipher_str:
            messagebox.showwarning("Предупреждение", "Введите зашифрованные блоки")
            return
        if not lengths_str:
            messagebox.showwarning("Предупреждение", "Введите длины блоков")
            return
        if not self.rsa.keys:
            messagebox.showwarning("Предупреждение", "Сначала сгенерируйте или загрузите ключи")
            return
        if self.rsa.n is None:
            messagebox.showwarning("Предупреждение", "Параметры n не заданы")
            return

        try:
            cipher_blocks = [int(x) for x in cipher_str.split()]
            lengths = [int(x) for x in lengths_str.split()]
            if len(cipher_blocks) != len(lengths):
                raise ValueError("Количество блоков не совпадает с количеством длин")

            key_idx = self.decrypt_key.current()
            if key_idx < 0:
                raise ValueError("Выберите ключ для расшифрования")
            _, d = self.rsa.keys[key_idx]

            decrypted = self.rsa.decrypt_blocks(cipher_blocks, d)

            blocks_str = "".join([f"M{i + 1} = {m} (длина: {ln})\n" for i, (m, ln) in enumerate(zip(decrypted, lengths))])
            self.decrypted_blocks.delete("1.0", tk.END)
            self.decrypted_blocks.insert("1.0", blocks_str)

            text = self.rsa.digits_to_text(decrypted, lengths)
            self.decrypted_text.delete("1.0", tk.END)
            self.decrypted_text.insert("1.0", text)

            messagebox.showinfo("Успех", "Текст расшифрован")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def load_plaintext_from_file(self) -> None:
        """Загружает открытый текст из файла."""
        path = filedialog.askopenfilename(
            title="Открыть текст",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.plaintext.delete("1.0", tk.END)
            self.plaintext.insert("1.0", content)
            self.notebook.select(self.encrypt_frame)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def save_plaintext_to_file(self) -> None:
        """Сохраняет открытый текст в файл."""
        path = filedialog.asksaveasfilename(
            title="Сохранить текст",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            content = self.plaintext.get("1.0", tk.END)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def save_decrypted_text_to_file(self) -> None:
        """Сохраняет расшифрованный текст в файл."""
        path = filedialog.asksaveasfilename(
            title="Сохранить расшифрованный текст",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            content = self.decrypted_text.get("1.0", tk.END)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def save_keys_to_file(self) -> None:
        """Сохраняет параметры и ключи в JSON."""
        if not self.rsa.keys:
            messagebox.showwarning("Предупреждение", "Нет ключей для сохранения")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить ключи",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            data = self.rsa.export_keys()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Успех", "Ключи сохранены")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def load_keys_from_file(self) -> None:
        """Загружает параметры и ключи из JSON."""
        path = filedialog.askopenfilename(
            title="Загрузить ключи",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.rsa.import_keys(data)

            self.p_entry.delete(0, tk.END)
            self.p_entry.insert(0, str(self.rsa.p))
            self.q_entry.delete(0, tk.END)
            self.q_entry.insert(0, str(self.rsa.q))
            self.check_primes()

            self.update_key_table()
            self.update_key_combos()
            messagebox.showinfo("Успех", "Ключи загружены")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def save_package_to_file(self) -> None:
        """Сохраняет пакет (ключи + последний результат шифрования) в JSON."""
        if not self.rsa.keys:
            messagebox.showwarning("Предупреждение", "Сначала сгенерируйте или загрузите ключи")
            return
        if not self.rsa.last_blocks or not self.rsa.last_cipher_blocks:
            messagebox.showwarning("Предупреждение", "Нет результата шифрования для сохранения")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить пакет",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            data = self.rsa.export_package()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Успех", "Пакет сохранен")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def load_package_from_file(self) -> None:
        """Загружает пакет (ключи + блоки) и заполняет поля расшифрования."""
        path = filedialog.askopenfilename(
            title="Загрузить пакет",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.rsa.import_package(data)

            self.p_entry.delete(0, tk.END)
            self.p_entry.insert(0, str(self.rsa.p))
            self.q_entry.delete(0, tk.END)
            self.q_entry.insert(0, str(self.rsa.q))
            self.check_primes()

            self.update_key_table()
            self.update_key_combos()

            if self.rsa.last_plain_digits:
                self.digit_display.delete("1.0", tk.END)
                self.digit_display.insert("1.0", self.rsa.last_plain_digits)

            if self.rsa.last_blocks:
                blocks_str = "".join([f"M{i + 1} = {m} (длина: {ln})\n" for i, (m, ln) in enumerate(self.rsa.last_blocks)])
                self.blocks_display.delete("1.0", tk.END)
                self.blocks_display.insert("1.0", blocks_str)
                self.lengths_entry.delete(0, tk.END)
                self.lengths_entry.insert(0, " ".join(str(ln) for _, ln in self.rsa.last_blocks))

            if self.rsa.last_cipher_blocks:
                self.encrypted_display.delete("1.0", tk.END)
                self.encrypted_display.insert("1.0", " ".join(map(str, self.rsa.last_cipher_blocks)))
                self.ciphertext.delete("1.0", tk.END)
                self.ciphertext.insert("1.0", " ".join(map(str, self.rsa.last_cipher_blocks)))

            self.notebook.select(self.decrypt_frame)
            messagebox.showinfo("Успех", "Пакет загружен")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def run(self) -> None:
        """Запуск главного цикла Tkinter."""
        self.root.mainloop()


if __name__ == "__main__":
    RSAGui().run()