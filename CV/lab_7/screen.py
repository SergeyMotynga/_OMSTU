import os
import time
from datetime import datetime

import mss
import mss.tools
import keyboard


# ====== НАСТРОЙКИ ======

BASE_DIR = r"C:\Users\motyn\Desktop\repositories\_OMSTU\CV\lab_7\screenshots"

START_KEY = "q+f8"       # Начать новую сессию
STOP_KEY = "q+f9"        # Остановить сессию
SCREENSHOT_KEY = "q+f7"  # Сделать один скриншот вручную
EXIT_KEY = "q+f10"       # Закрыть программу

# =======================


is_recording = False
current_session_dir = None
stop_program = False


def create_session_folder():
    """
    Создает новую папку для одной сессии скриншотов.

    Returns:
        str: Путь к созданной папке сессии.
    """
    os.makedirs(BASE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join(BASE_DIR, f"game_{timestamp}")

    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def make_screenshot():
    """
    Делает один скриншот основного монитора и сохраняет его
    в текущую папку сессии.

    Скриншот сохраняется только если сессия уже запущена.
    """
    global is_recording, current_session_dir

    if not is_recording or current_session_dir is None:
        print("[ИНФО] Сначала запусти сессию скриншотов.")
        print(f"[ПОДСКАЗКА] Нажми {START_KEY.upper()}, чтобы создать папку.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    filename = os.path.join(current_session_dir, f"dota_{timestamp}.png")

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)

        mss.tools.to_png(
            screenshot.rgb,
            screenshot.size,
            output=filename
        )

    print(f"[СКРИН] Сохранено: {filename}")


def start_recording():
    """
    Запускает новую сессию ручных скриншотов.

    При запуске создается новая папка, куда будут сохраняться
    все скриншоты, сделанные горячей клавишей SCREENSHOT_KEY.
    """
    global is_recording, current_session_dir

    if is_recording:
        print("[ИНФО] Сессия уже запущена.")
        return

    current_session_dir = create_session_folder()
    is_recording = True

    print("=" * 50)
    print("[СТАРТ] Сессия скриншотов началась.")
    print(f"[ПАПКА] {current_session_dir}")
    print(f"[СКРИНШОТ] Нажми {SCREENSHOT_KEY.upper()}, чтобы сделать скрин.")
    print("=" * 50)


def stop_recording():
    """
    Останавливает текущую сессию скриншотов.
    """
    global is_recording

    if not is_recording:
        print("[ИНФО] Сессия сейчас не запущена.")
        return

    is_recording = False

    print("=" * 50)
    print("[СТОП] Сессия скриншотов остановлена.")
    print("=" * 50)


def exit_program():
    """
    Завершает программу.
    """
    global stop_program, is_recording

    is_recording = False
    stop_program = True

    print("[ВЫХОД] Программа завершается...")


def main():
    """
    Запускает программу для ручного создания скриншотов по горячим клавишам.
    """
    print("Ручные скриншоты для Dota 2 запущены.")
    print(f"Нажми {START_KEY.upper()} — начать новую сессию.")
    print(f"Нажми {SCREENSHOT_KEY.upper()} — сделать один скриншот.")
    print(f"Нажми {STOP_KEY.upper()} — остановить сессию.")
    print(f"Нажми {EXIT_KEY.upper()} — выйти.")

    keyboard.add_hotkey(START_KEY, start_recording)
    keyboard.add_hotkey(SCREENSHOT_KEY, make_screenshot)
    keyboard.add_hotkey(STOP_KEY, stop_recording)
    keyboard.add_hotkey(EXIT_KEY, exit_program)

    while not stop_program:
        time.sleep(0.2)


if __name__ == "__main__":
    main()