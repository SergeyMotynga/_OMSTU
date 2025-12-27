import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import time
from typing import Optional, Dict, List, Tuple
from datetime import datetime


class DiscreteLogarithm:
    """
    Класс для вычисления дискретного логарифма методом "шаг младенца - шаг великана".

    Решает задачу: найти x такое, что a^x ≡ b (mod p)

    Attributes:
        baby_steps_table: Таблица шагов младенца [(n, y_n), ...]
        giant_steps_table: Таблица шагов великана [(n, z_n), ...]
        k: Параметр алгоритма k = ⌈√p⌉ + 1
    """

    def __init__(self):
        """Инициализация класса."""
        self.baby_steps_table: List[Tuple[int, int]] = []
        self.giant_steps_table: List[Tuple[int, int]] = []
        self.k = 0

    def integer_sqrt(self, n: int) -> int:
        """
        Вычисляет целочисленный квадратный корень методом Ньютона.

        Args:
            n: Неотрицательное целое число

        Returns:
            Целая часть квадратного корня из n

        Raises:
            ValueError: Если n отрицательное
        """
        if n < 0:
            raise ValueError("Нельзя вычислить корень из отрицательного числа")
        if n == 0:
            return 0

        x = n
        y = (x + 1) // 2
        while y < x:
            x = y
            y = (x + n // x) // 2
        return x

    def mod_pow(self, base: int, exp: int, mod: int) -> int:
        """
        Быстрое возведение в степень по модулю (бинарный алгоритм).

        Args:
            base: Основание
            exp: Показатель степени
            mod: Модуль

        Returns:
            base^exp mod mod
        """
        result = 1
        base %= mod
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % mod
            base = (base * base) % mod
            exp //= 2
        return result

    def baby_giant_step(self, a: int, b: int, p: int) -> Optional[int]:
        """
        Алгоритм "шаг младенца - шаг великана" для вычисления дискретного логарифма.

        Строит две последовательности:
        - y_n = a^(n*k) mod p (шаги младенца)
        - z_n = b * a^n mod p (шаги великана)

        При совпадении y_i = z_j вычисляет x = i*k - j

        Args:
            a: Основание
            b: Результат (a^x ≡ b (mod p))
            p: Модуль (простое число)

        Returns:
            x такое, что a^x ≡ b (mod p), или None если не найдено
        """
        self.baby_steps_table = []
        self.giant_steps_table = []

        if p == 1:
            return 0

        self.k = self.integer_sqrt(p) + 1
        k = self.k

        y_to_index: Dict[int, int] = {}
        a_power_k = self.mod_pow(a, k, p)
        y_n = 1

        for n in range(k + 1):
            self.baby_steps_table.append((n, y_n))
            y_to_index[y_n] = n
            y_n = (y_n * a_power_k) % p

        z_n = b
        for n in range(k + 1):
            self.giant_steps_table.append((n, z_n))

            if z_n in y_to_index:
                i = y_to_index[z_n]
                j = n
                x = i * k - j

                if x >= 0:
                    if self.mod_pow(a, x, p) == b:
                        return x

            z_n = (z_n * a) % p

        return None


class DiscreteLogGUI:
    """
    Графический интерфейс для вычисления дискретного логарифма.

    Предоставляет функционал:
    - Ввод параметров a, b, p
    - Загрузка параметров из файла
    - Вычисление дискретного логарифма
    - Сохранение результатов в файл
    - Экспорт таблицы шагов алгоритма
    """

    def __init__(self):
        """Инициализация графического интерфейса."""
        self.root = tk.Tk()
        self.root.title("Лабораторная работа №7: Дискретный логарифм")
        self.root.geometry("900x700")

        self.dlog = DiscreteLogarithm()
        self.last_result = None

        self.create_widgets()

    def create_widgets(self):
        """Создает все элементы графического интерфейса."""
        ttk.Label(self.root, text="Алгоритм «Шаг младенца – шаг великана»",
                  font=('Arial', 14, 'bold')).pack(pady=10)
        ttk.Label(self.root, text="Найти x такое, что: a^x ≡ b (mod p)",
                  font=('Arial', 11)).pack()

        input_frame = ttk.LabelFrame(self.root, text="Параметры", padding=10)
        input_frame.pack(fill='x', padx=10, pady=10)

        param_frame = ttk.Frame(input_frame)
        param_frame.pack()

        ttk.Label(param_frame, text="a (основание):").grid(row=0, column=0, padx=5, pady=5)
        self.a_entry = ttk.Entry(param_frame, width=15)
        self.a_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(param_frame, text="b (результат):").grid(row=0, column=2, padx=5, pady=5)
        self.b_entry = ttk.Entry(param_frame, width=15)
        self.b_entry.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(param_frame, text="p (модуль):").grid(row=0, column=4, padx=5, pady=5)
        self.p_entry = ttk.Entry(param_frame, width=15)
        self.p_entry.grid(row=0, column=5, padx=5, pady=5)

        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="📂 Загрузить из файла",
                   command=self.load_from_file).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="📝 Вариант 20",
                   command=self.load_example).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="🧹 Очистить",
                   command=self.clear_all).pack(side='left', padx=5)

        ttk.Button(btn_frame, text="🔢 Вычислить логарифм",
                   command=self.compute).pack(side='left', padx=15)

        save_frame = ttk.Frame(input_frame)
        save_frame.pack()

        ttk.Button(save_frame, text="💾 Сохранить результаты",
                   command=self.save_results).pack(side='left', padx=5)
        ttk.Button(save_frame, text="📊 Экспорт таблицы шагов",
                   command=self.export_steps_table).pack(side='left', padx=5)

        result_frame = ttk.LabelFrame(self.root, text="Результаты", padding=10)
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.result_text = scrolledtext.ScrolledText(result_frame, height=25, wrap=tk.WORD)
        self.result_text.pack(fill='both', expand=True)

    def load_example(self):
        """Загружает пример из варианта 20 (a=2, b=30994, p=31607)."""
        self.a_entry.delete(0, tk.END)
        self.a_entry.insert(0, "2")
        self.b_entry.delete(0, tk.END)
        self.b_entry.insert(0, "30994")
        self.p_entry.delete(0, tk.END)
        self.p_entry.insert(0, "31607")

    def clear_all(self):
        """Очищает все поля ввода и результаты."""
        self.a_entry.delete(0, tk.END)
        self.b_entry.delete(0, tk.END)
        self.p_entry.delete(0, tk.END)
        self.result_text.delete('1.0', tk.END)
        self.last_result = None

    def load_from_file(self):
        """
        Загружает параметры из текстового файла.

        Формат файла:
            a = значение
            b = значение
            p = значение
        """
        filepath = filedialog.askopenfilename(
            title="Загрузить параметры",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )

        if not filepath:
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            params = {}
            for line in lines:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    params[key.strip().lower()] = value.strip()

            if 'a' in params:
                self.a_entry.delete(0, tk.END)
                self.a_entry.insert(0, params['a'])
            if 'b' in params:
                self.b_entry.delete(0, tk.END)
                self.b_entry.insert(0, params['b'])
            if 'p' in params:
                self.p_entry.delete(0, tk.END)
                self.p_entry.insert(0, params['p'])

            messagebox.showinfo("Успех", f"Параметры загружены из:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{e}")

    def save_results(self):
        """Сохраняет результаты вычислений в текстовый файл."""
        if not self.last_result:
            messagebox.showwarning("Предупреждение", "Сначала выполните вычисление!")
            return

        filepath = filedialog.asksaveasfilename(
            title="Сохранить результаты",
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )

        if not filepath:
            return

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("РЕЗУЛЬТАТЫ ВЫЧИСЛЕНИЯ ДИСКРЕТНОГО ЛОГАРИФМА\n")
                f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")

                f.write("ВХОДНЫЕ ПАРАМЕТРЫ:\n")
                f.write(f"  a = {self.last_result['a']}\n")
                f.write(f"  b = {self.last_result['b']}\n")
                f.write(f"  p = {self.last_result['p']}\n\n")

                f.write("АЛГОРИТМ:\n")
                f.write(f"  k = ⌈√p⌉ + 1 = {self.last_result['k']}\n\n")

                f.write("РЕЗУЛЬТАТ:\n")
                if self.last_result['x'] is not None:
                    f.write(f"  x = {self.last_result['x']}\n")
                    f.write(f"  Проверка: {self.last_result['a']}^{self.last_result['x']} ≡ "
                            f"{self.last_result['check']} (mod {self.last_result['p']})\n")
                    f.write(f"  Проверка пройдена: {'Да' if self.last_result['verified'] else 'Нет'}\n")
                else:
                    f.write("  Дискретный логарифм не найден\n")

                f.write(f"\nВремя вычисления: {self.last_result['time']:.4f} сек\n")

            messagebox.showinfo("Успех", f"Результаты сохранены в:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{e}")

    def export_steps_table(self):
        """Экспортирует таблицу шагов младенца и великана в файл."""
        if not self.dlog.baby_steps_table:
            messagebox.showwarning("Предупреждение", "Сначала выполните вычисление!")
            return

        filepath = filedialog.asksaveasfilename(
            title="Экспорт таблицы шагов",
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("CSV файлы", "*.csv"), ("Все файлы", "*.*")]
        )

        if not filepath:
            return

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                a = self.last_result['a']
                p = self.last_result['p']
                b = self.last_result['b']
                k = self.last_result['k']

                f.write("ТАБЛИЦА ШАГОВ АЛГОРИТМА\n")
                f.write(f"Параметры: a={a}, b={b}, p={p}, k={k}\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"ШАГИ МЛАДЕНЦА: y_n = a^(n*k) mod p = {a}^(n*{k}) mod {p}\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'n':>5} | {'y_n':>15}\n")
                f.write("-" * 40 + "\n")
                for n, y in self.dlog.baby_steps_table[:50]:
                    f.write(f"{n:>5} | {y:>15}\n")
                if len(self.dlog.baby_steps_table) > 50:
                    f.write(f"... (всего {len(self.dlog.baby_steps_table)} записей)\n")

                f.write("\n")

                f.write(f"ШАГИ ВЕЛИКАНА: z_n = b*a^n mod p = {b}*{a}^n mod {p}\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'n':>5} | {'z_n':>15}\n")
                f.write("-" * 40 + "\n")
                for n, z in self.dlog.giant_steps_table[:50]:
                    f.write(f"{n:>5} | {z:>15}\n")
                if len(self.dlog.giant_steps_table) > 50:
                    f.write(f"... (всего {len(self.dlog.giant_steps_table)} записей)\n")

            messagebox.showinfo("Успех", f"Таблица экспортирована в:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось экспортировать:\n{e}")

    def compute(self):
        """Выполняет вычисление дискретного логарифма и отображает результаты."""
        try:
            a = int(self.a_entry.get())
            b = int(self.b_entry.get())
            p = int(self.p_entry.get())

            if a <= 0 or b <= 0 or p <= 1:
                raise ValueError("a, b > 0 и p > 1")

            a = a % p
            b = b % p

            if a == 0:
                raise ValueError("a не должно быть кратно p")

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректные параметры:\n{e}")
            return

        start_time = time.time()
        x = self.dlog.baby_giant_step(a, b, p)
        elapsed = time.time() - start_time

        self.result_text.delete('1.0', tk.END)

        output = "=" * 55 + "\n"
        output += "ВЫЧИСЛЕНИЕ ДИСКРЕТНОГО ЛОГАРИФМА\n"
        output += "=" * 55 + "\n\n"
        output += f"Задача: найти x такое, что {a}^x ≡ {b} (mod {p})\n\n"
        output += f"Параметр k = ⌈√{p}⌉ + 1 = {self.dlog.k}\n\n"

        output += "Шаги младенца (первые 11):\n"
        output += "n  : " + " ".join(f"{n:>4}" for n, _ in self.dlog.baby_steps_table[:11]) + "\n"
        output += "y_n: " + " ".join(f"{y:>4}" for _, y in self.dlog.baby_steps_table[:11]) + "\n\n"

        output += "Шаги великана (первые 11):\n"
        output += "n  : " + " ".join(f"{n:>4}" for n, _ in self.dlog.giant_steps_table[:11]) + "\n"
        output += "z_n: " + " ".join(f"{z:>4}" for _, z in self.dlog.giant_steps_table[:11]) + "\n\n"

        check_val = self.dlog.mod_pow(a, x, p) if x is not None else None
        self.last_result = {
            'a': a, 'b': b, 'p': p, 'k': self.dlog.k,
            'x': x, 'check': check_val,
            'verified': check_val == b if x else False,
            'time': elapsed
        }

        if x is not None:
            output += "=" * 55 + "\n"
            output += f"РЕЗУЛЬТАТ НАЙДЕН: x = {x}\n"
            output += "=" * 55 + "\n\n"
            output += f"Проверка: {a}^{x} mod {p} = {check_val}\n"
            if check_val == b:
                output += f"Проверка пройдена: {check_val} = {b}\n"
            else:
                output += f"ОШИБКА проверки: {check_val} != {b}\n"
        else:
            output += "=" * 55 + "\n"
            output += "Дискретный логарифм НЕ НАЙДЕН\n"
            output += "=" * 55 + "\n"

        output += f"\nВремя вычисления: {elapsed:.4f} сек\n"

        self.result_text.insert('1.0', output)

    def run(self):
        """Запускает главный цикл приложения."""
        self.root.mainloop()


if __name__ == "__main__":
    app = DiscreteLogGUI()
    app.run()