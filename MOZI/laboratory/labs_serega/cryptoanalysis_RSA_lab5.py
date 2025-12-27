import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import math
import time
import threading
from typing import Tuple, Optional, List
import json


class RSACryptanalysis:
    """Класс для криптоанализа RSA с факторизацией методом пробного деления."""

    REVERSE_ALPHABET = {
        10: 'А', 11: 'Б', 12: 'В', 13: 'Г', 14: 'Д', 15: 'Е', 16: 'Ж', 17: 'З',
        18: 'И', 19: 'Й', 20: 'К', 21: 'Л', 22: 'М', 23: 'Н', 24: 'О', 25: 'П',
        26: 'Р', 27: 'С', 28: 'Т', 29: 'У', 30: 'Ф', 31: 'Х', 32: 'Ц', 33: 'Ч',
        34: 'Ш', 35: 'Щ', 36: 'Ъ', 37: 'Ы', 38: 'Ь', 39: 'Э', 40: 'Ю', 41: 'Я', 99: ' '
    }

    def __init__(self):
        """Инициализация класса криптоанализа."""
        self.stop_flag = False

    def extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        """
        Расширенный алгоритм Евклида.

        Возвращает кортеж (gcd, x, y), где gcd - НОД(a, b),
        x и y - коэффициенты такие, что a*x + b*y = gcd.
        """
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = self.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y

    def mod_inverse(self, e: int, phi: int) -> Optional[int]:
        """
        Находит обратный элемент e по модулю phi.

        Возвращает d такое, что e * d ≡ 1 (mod phi),
        или None, если обратный элемент не существует.
        """
        gcd, x, _ = self.extended_gcd(e, phi)
        if gcd != 1:
            return None
        return x % phi

    def fast_pow(self, base: int, exp: int, mod: int) -> int:
        """
        Быстрое возведение в степень по модулю.

        Вычисляет base^exp mod mod за O(log exp) операций.
        """
        result = 1
        base %= mod
        while exp > 0:
            if exp & 1:
                result = (result * base) % mod
            base = (base * base) % mod
            exp >>= 1
        return result

    def generate_primes(self, limit: int):
        """Генерирует список простых чисел до limit включительно."""
        if limit < 2:
            return []
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i in range(2, limit + 1) if sieve[i]]

    def trial_division_factorize(self, n: int, progress_callback=None):
        """
        Факторизация методом пробных делений с группировкой делителей через НОД.
        """
        self.stop_flag = False
        temp_n = n
        factors = []
        
        # 1. Выделяем все степени 2 и 3 отдельно, как указано в теории
        for p in [2, 3]:
            while temp_n % p == 0:
                factors.append(p)
                temp_n //= p
                if progress_callback:
                    progress_callback(f"Найден делитель: {p}", 5)

        # 2. Подготовка базы простых чисел (от 5 до корня из n)
        limit = int(math.sqrt(temp_n))
        primes = [p for p in self.generate_primes(limit) if p > 3]
        
        if progress_callback:
            progress_callback(f"Сформирована база из {len(primes)} простых чисел", 20)

        # 3. Групповой метод с использованием НОД
        # Группируем простые числа по 3, как в примере
        block_size = 3
        for i in range(0, len(primes), block_size):
            if self.stop_flag or temp_n == 1:
                break
                
            # Формируем блок (например, q1 = 5 * 7 * 11)
            block = primes[i : i + block_size]
            qs = 1
            for p in block:
                qs *= p
                
            # Проверяем общие делители через НОД
            d = math.gcd(temp_n, qs)
            
            while d > 1:
                # Если НОД > 1, значит в блоке есть как минимум один делитель
                for p in block:
                    while temp_n % p == 0:
                        factors.append(p)
                        temp_n //= p
                        if progress_callback:
                            progress_callback(f"Найден делитель: {p}", 50)
                
                # Пересчитываем НОД для текущего temp_n (на случай кратных делителей)
                d = math.gcd(temp_n, qs)

            if progress_callback and i % 100 == 0:
                progress = 20 + int((i / len(primes)) * 70)
                progress_callback(f"Проверка блока {i//block_size + 1}...", progress)

        # 4. Если после всех проверок temp_n > 1, то это последнее простое число
        if temp_n > 1:
            factors.append(temp_n)

        # Возвращаем два множителя (для RSA) или список всех множителей
        if len(factors) >= 2:
            if progress_callback:
                progress_callback("Факторизация завершена", 100)
            # Если нужно именно p и q:
            return factors[0], n // factors[0]
        
        return None

    def decrypt_blocks(self, cipher_blocks: List[int], d: int, n: int) -> List[int]:
        """
        Расшифровывает список зашифрованных блоков.

        Для каждого блока C вычисляет M = C^d mod n.
        """
        return [self.fast_pow(c, d, n) for c in cipher_blocks]

    def blocks_to_text(self, blocks: List[int]) -> str:
        """
        Преобразует числовые блоки в текст.

        Объединяет блоки в строку цифр и декодирует
        по 2 цифры согласно таблице REVERSE_ALPHABET.
        """
        all_digits = ''.join(str(num) for num in blocks)
        text = ""
        i = 0
        while i + 1 < len(all_digits):
            code = int(all_digits[i:i + 2])
            text += self.REVERSE_ALPHABET.get(code, '?')
            i += 2
        return text

    def stop_factorization(self):
        """Останавливает процесс факторизации."""
        self.stop_flag = True


class RSACryptanalysisGUI:
    """Графический интерфейс для криптоанализа RSA."""

    def __init__(self):
        """Инициализация графического интерфейса."""
        self.root = tk.Tk()
        self.root.title("Лабораторная работа №5: Криптоанализ RSA")
        self.root.geometry("900x700")

        self.analyzer = RSACryptanalysis()
        self.crack_thread = None

        self.create_widgets()
        self.load_variant20()

    def create_widgets(self):
        """Создает все элементы интерфейса."""
        title = ttk.Label(
            self.root,
            text="КРИПТОАНАЛИЗ ШИФРА RSA — взлом через факторизацию n",
            font=('Arial', 12, 'bold')
        )
        title.pack(pady=10)

        input_frame = ttk.LabelFrame(self.root, text="Входные данные", padding=10)
        input_frame.pack(fill='x', padx=10, pady=5)

        key_frame = ttk.Frame(input_frame)
        key_frame.pack(fill='x', pady=5)

        ttk.Label(key_frame, text="Открытый ключ (e, n):").pack(side='left', padx=5)
        ttk.Label(key_frame, text="e =").pack(side='left', padx=5)
        self.e_entry = ttk.Entry(key_frame, width=10)
        self.e_entry.pack(side='left', padx=5)

        ttk.Label(key_frame, text="n =").pack(side='left', padx=5)
        self.n_entry = ttk.Entry(key_frame, width=15)
        self.n_entry.pack(side='left', padx=5)

        ttk.Button(key_frame, text="Вариант 20", command=self.load_variant20).pack(side='left', padx=20)

        cipher_frame = ttk.Frame(input_frame)
        cipher_frame.pack(fill='x', pady=5)
        ttk.Label(cipher_frame, text="Зашифрованные блоки C (через пробел):").pack(anchor='w')
        self.cipher_entry = ttk.Entry(cipher_frame, width=60)
        self.cipher_entry.pack(fill='x', pady=5)

        file_frame = ttk.LabelFrame(self.root, text="Работа с файлами", padding=10)
        file_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(
            file_frame,
            text="📂 Загрузить данные из файла",
            command=self.load_from_file
        ).pack(side='left', padx=5)

        ttk.Button(
            file_frame,
            text="💾 Сохранить результаты в файл",
            command=self.save_to_file
        ).pack(side='left', padx=5)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill='x', padx=10, pady=5)

        self.crack_button = ttk.Button(
            control_frame,
            text="🔓 Начать взлом (пробное деление)",
            command=self.start_cracking
        )
        self.crack_button.pack(side='left', padx=5)

        self.stop_button = ttk.Button(
            control_frame,
            text="⏹ Остановить",
            command=self.stop_cracking,
            state='disabled'
        )
        self.stop_button.pack(side='left', padx=5)

        ttk.Button(control_frame, text="📊 Анализ n", command=self.analyze_n).pack(side='left', padx=5)

        progress_frame = ttk.Frame(self.root)
        progress_frame.pack(fill='x', padx=10, pady=5)
        self.progress_label = ttk.Label(progress_frame, text="Готов к взлому")
        self.progress_label.pack()
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.progress_bar.pack(pady=5)

        result_frame = ttk.LabelFrame(self.root, text="Результаты взлома", padding=10)
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.result_text = scrolledtext.ScrolledText(result_frame, height=18, wrap=tk.WORD)
        self.result_text.pack(fill='both', expand=True)

    def load_from_file(self):
        """Загружает входные данные (e, n, зашифрованные блоки) из файла."""
        filepath = filedialog.askopenfilename(
            title="Загрузить входные данные",
            filetypes=[
                ("Текстовые файлы", "*.txt"),
                ("JSON файлы", "*.json"),
                ("Все файлы", "*.*")
            ]
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if filepath.endswith('.json'):
                data = json.loads(content)
                e = data.get('e', '')
                n = data.get('n', '')
                cipher = data.get('cipher', data.get('C', ''))
                if isinstance(cipher, list):
                    cipher = ' '.join(map(str, cipher))
            else:
                lines = content.split('\n')
                if len(lines) >= 3:
                    e = lines[0].strip()
                    n = lines[1].strip()
                    cipher = lines[2].strip()
                elif len(lines) == 1:
                    parts = lines[0].split()
                    e, n = parts[0], parts[1]
                    cipher = ' '.join(parts[2:])
                else:
                    raise ValueError("Неверный формат файла")

            self.e_entry.delete(0, tk.END)
            self.e_entry.insert(0, str(e))
            self.n_entry.delete(0, tk.END)
            self.n_entry.insert(0, str(n))
            self.cipher_entry.delete(0, tk.END)
            self.cipher_entry.insert(0, str(cipher))

            messagebox.showinfo("Успех", f"Данные загружены из {filepath}")

        except Exception as ex:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{ex}")

    def save_to_file(self):
        """Сохраняет результаты взлома в текстовый файл."""
        content = self.result_text.get('1.0', tk.END).strip()
        if not content:
            messagebox.showwarning("Предупреждение", "Нет результатов для сохранения")
            return

        filepath = filedialog.asksaveasfilename(
            title="Сохранить результаты",
            defaultextension=".txt",
            filetypes=[
                ("Текстовые файлы", "*.txt"),
                ("Все файлы", "*.*")
            ]
        )
        if not filepath:
            return

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            messagebox.showinfo("Успех", f"Результаты сохранены в {filepath}")
        except Exception as ex:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{ex}")

    def load_variant20(self):
        """Загружает данные варианта 20 из таблицы задания."""
        self.e_entry.delete(0, tk.END)
        self.e_entry.insert(0, "251")
        self.n_entry.delete(0, tk.END)
        self.n_entry.insert(0, "70691")
        self.cipher_entry.delete(0, tk.END)
        self.cipher_entry.insert(0, "18291")

    def analyze_n(self):
        """Выполняет анализ модуля n и выводит информацию о сложности факторизации."""
        try:
            n = int(self.n_entry.get())
            info = f"АНАЛИЗ МОДУЛЯ n = {n}\n{'=' * 50}\n\n"
            info += f"Размер числа: {n.bit_length()} бит\n"
            info += f"√n ≈ {int(math.sqrt(n))}\n"

            ops = int(math.sqrt(n))
            info += f"\nМетод пробного деления потребует до {ops:,} проверок\n"

            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', info)
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректное число")

    def start_cracking(self):
        """Запускает процесс взлома RSA в отдельном потоке."""
        try:
            e = int(self.e_entry.get())
            n = int(self.n_entry.get())
            cipher_blocks = list(map(int, self.cipher_entry.get().split()))

            if not cipher_blocks:
                raise ValueError("Введите хотя бы один зашифрованный блок")

            self.crack_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.progress_bar['value'] = 0

            self.crack_thread = threading.Thread(
                target=self.crack_in_thread,
                args=(e, n, cipher_blocks)
            )
            self.crack_thread.start()

        except ValueError as ex:
            messagebox.showerror("Ошибка", str(ex))

    def crack_in_thread(self, e: int, n: int, cipher_blocks: List[int]):
        """
        Выполняет полный цикл взлома RSA в отдельном потоке.

        Этапы:
        1. Факторизация n методом пробного деления
        2. Вычисление φ(n) и закрытого ключа d
        3. Расшифрование блоков
        """
        start_time = time.time()

        def update_progress(msg, progress):
            self.root.after(0, lambda: self.update_progress(msg, progress))

        update_progress("Факторизация методом пробного деления...", 0)
        p_q = self.analyzer.trial_division_factorize(n, update_progress)

        if not p_q:
            self.root.after(0, lambda: self.show_error("Не удалось факторизовать n"))
            return

        p, q = p_q

        phi = (p - 1) * (q - 1)
        d = self.analyzer.mod_inverse(e, phi)

        if d is None:
            self.root.after(0, lambda: self.show_error("Не удалось найти закрытый ключ d"))
            return

        update_progress("Расшифровка блоков...", 95)
        plain_blocks = self.analyzer.decrypt_blocks(cipher_blocks, d, n)
        text = self.analyzer.blocks_to_text(plain_blocks)

        elapsed = time.time() - start_time

        result = f"ВЗЛОМ ЗАВЕРШЁН УСПЕШНО!\n{'=' * 50}\n\n"
        result += f"Метод: ПРОБНОЕ ДЕЛЕНИЕ (перебор от 2 до √n)\n"
        result += f"Время взлома: {elapsed:.3f} сек\n\n"
        result += f"ЭТАП 1: ФАКТОРИЗАЦИЯ\n"
        result += f"  n = {n} = {p} × {q}\n"
        result += f"  p = {p}\n"
        result += f"  q = {q}\n\n"
        result += f"ЭТАП 2: ВЫЧИСЛЕНИЕ ЗАКРЫТОГО КЛЮЧА\n"
        result += f"  φ(n) = (p-1)(q-1) = ({p}-1)×({q}-1) = {phi}\n"
        result += f"  d = e⁻¹ mod φ(n) = {e}⁻¹ mod {phi} = {d}\n"
        result += f"  Проверка: e×d mod φ(n) = {e}×{d} mod {phi} = {(e * d) % phi}\n\n"
        result += f"ЭТАП 3: РАСШИФРОВАНИЕ\n"
        result += f"  Зашифрованные блоки C: {cipher_blocks}\n"
        result += f"  Расшифрованные блоки M: {plain_blocks}\n"
        result += f"  Числовая строка: {''.join(map(str, plain_blocks))}\n\n"
        result += f"ИСХОДНОЕ СООБЩЕНИЕ:\n  \"{text}\"\n\n"
        result += "=" * 50

        self.root.after(0, lambda: self.show_result(result))

    def update_progress(self, message: str, progress: int):
        """Обновляет индикатор прогресса и текст статуса."""
        self.progress_label.config(text=message)
        self.progress_bar['value'] = progress

    def show_result(self, result: str):
        """Отображает результат взлома в текстовом поле."""
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert('1.0', result)
        self.crack_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.progress_label.config(text="Взлом завершен!")
        self.progress_bar['value'] = 100

    def show_error(self, error: str):
        """Отображает сообщение об ошибке."""
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert('1.0', f"ОШИБКА: {error}")
        self.crack_button.config(state='normal')
        self.stop_button.config(state='disabled')
        messagebox.showerror("Ошибка", error)

    def stop_cracking(self):
        """Останавливает процесс взлома."""
        self.analyzer.stop_factorization()
        self.stop_button.config(state='disabled')
        self.crack_button.config(state='normal')
        self.progress_label.config(text="Взлом остановлен")

    def run(self):
        """Запускает главный цикл приложения."""
        self.root.mainloop()


if __name__ == "__main__":
    app = RSACryptanalysisGUI()
    app.run()