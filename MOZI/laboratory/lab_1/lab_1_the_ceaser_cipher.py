import streamlit as st

EXCEPTIONS = '.,?!-" '

def isRussia(letter: str) -> bool:
    return 1040 <= ord(letter) <= 1103

def encryption(text: str, k: int) -> str:
    out = []
    for ch in text:
        if not isRussia(ch) or ch in EXCEPTIONS:
            out.append(ch)
        elif ch.islower():
            out.append(chr(ord('а') + (ord(ch) + k - ord('а')) % 32))
        else:
            out.append(chr(ord('А') + (ord(ch) + k - ord('А')) % 32))
    return "".join(out)

def decrypt_enumeration(text: str):
    res = {}
    for k in range(1, 32):
        out = []
        for ch in text:
            if not isRussia(ch) or ch in EXCEPTIONS:
                out.append(ch)
            elif ch.islower():
                out.append(chr(ord('я') - (ord('я') - ord(ch) + k) % 32))
            else:
                out.append(chr(ord('Я') - (ord('Я') - ord(ch) + k) % 32))
        res[k] = "".join(out)
    return res

def decrypt(text: str, k: int=None):
    if k is None:
        return decrypt_enumeration(text)
    out = []
    for ch in text:
        if not isRussia(ch) or ch in EXCEPTIONS:
            out.append(ch)
        elif ch.islower():
            out.append(chr(ord('я') - (ord('я') - ord(ch) + k) % 32))
        else:
            out.append(chr(ord('Я') - (ord('Я') - ord(ch) + k) % 32))
    return "".join(out)

def validate_inputs(action: str, text: str, shift):
    ok = True
    if not text.strip():
        st.error("Введите текст вручную или загрузите файл.")
        ok = False
    elif not any(isRussia(c) for c in text):
        st.warning("В тексте нет русских букв — результат может быть бессмысленным.")
    if action in ("Шифровка", "Расшифровка") and shift is not None:
        try:
            k = int(shift)
        except Exception:
            st.error("Шаг должен быть целым числом.")
            return False
        if not (1 <= k <= 31):
            st.error("Шаг вне диапазона. Допустимо 1–31.")
            ok = False
    return ok


def main_page():
    st.set_page_config(page_title="Шифр Цезаря", page_icon="🔑", layout="centered")
    st.title("🔑 Шифр Цезаря")

    ss = st.session_state
    ss.setdefault("dec_variants", {})
    ss.setdefault("last_text_for_variants", "")
    ss.setdefault("enum_results", {})
    ss.setdefault("enum_active", False)
    ss.setdefault("enum_text_snapshot", "")
    ss.setdefault("enum_selected_key", 1)
    ss.setdefault("dec_use_step_prev", False)
    ss.setdefault("last_enc", {"text": "", "shift": None, "result": ""})
    ss.setdefault("enc_ready", False)

    # --- Загрузка файла ---
    st.markdown("**Загрузите текстовый файл (.txt)** или введите текст вручную:")
    uploaded_file = st.file_uploader("", type=["txt"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            file_content = uploaded_file.read().decode("windows-1251", errors="ignore")
        st.info(f"Загружен файл: {uploaded_file.name}")
    else:
        file_content = ""

    # --- Ввод текста ---
    user_text = st.text_area("Введите текст:", file_content).replace('ё', 'е').replace('Ё', 'Е')

    # --- Сбросы состояния ---
    if user_text != ss.last_text_for_variants:
        ss.dec_variants = {}
        ss.last_text_for_variants = user_text
    if user_text != ss.enum_text_snapshot:
        ss.enum_active = False
        ss.enum_results = {}
        ss.enum_selected_key = 1
        ss.enum_text_snapshot = user_text
    if user_text != ss.last_enc.get("text", ""):
        ss.enc_ready = False

    # --- Действие ---
    action = st.selectbox("Выберите действие:", ("Шифровка", "Расшифровка"))

    use_step = False
    if action == "Расшифровка":
        use_step = st.checkbox("Выбрать шаг вручную", value=False, key="dec_use_step")
        if use_step != ss.dec_use_step_prev:
            ss.dec_use_step_prev = use_step
            if use_step:
                ss.enum_active = False
                ss.enum_results = {}
                ss.enum_selected_key = 1

    # --- Форма ---
    with st.form("cipher_form", clear_on_submit=False):
        shift_val = None
        if action == "Шифровка":
            shift_val = st.slider("Шаг (1–31)", 1, 31, 1, key="enc_shift")
        elif action == "Расшифровка" and use_step:
            shift_val = st.slider("Шаг (1–31)", 1, 31, 1, key="dec_shift")

        text_ok = bool(user_text.strip())
        shift_needed = (action == "Шифровка") or (action == "Расшифровка" and use_step)
        shift_ok = (not shift_needed) or (shift_val is not None and 1 <= int(shift_val) <= 31)

        submitted = st.form_submit_button(action, disabled=not (text_ok and shift_ok))

    # --- Очистка накопленных ---
    if action == "Расшифровка" and use_step and ss.dec_variants:
        if st.button("Очистить накопленные варианты"):
            ss.dec_variants = {}
            st.info("Список вариантов очищен.")

    # --- Обработка ---
    if submitted:
        if not validate_inputs(action, user_text, shift_val if shift_needed else None):
            ss.enc_ready = False
            st.stop()

        if action == "Шифровка":
            k = int(shift_val)
            result = encryption(user_text, k)
            ss.last_enc = {"text": user_text, "shift": k, "result": result}
            ss.enc_ready = True
        else:
            if not use_step:
                ss.enum_results = decrypt(user_text, None)
                ss.enum_active = True
                ss.enum_selected_key = 1 if 1 in ss.enum_results else list(ss.enum_results.keys())[0]
            else:
                k = int(shift_val)
                ss.dec_variants[k] = decrypt(user_text, k)

    # --- Вывод ---
    if action == "Шифровка" and ss.enc_ready and ss.last_enc["text"] == user_text:
        st.subheader("🔐 Зашифрованный текст:")
        st.code(ss.last_enc["result"], language="text")
        st.download_button(
            "Скачать результат",
            data=f"Исходный текст:\n{ss.last_enc['text']}\n\nКлюч: {ss.last_enc['shift']}\n\nЗашифрованный:\n{ss.last_enc['result']}",
            file_name="Шифр_Цезаря_Шифровка.txt",
            mime="text/plain",
        )

    if action == "Расшифровка" and not use_step and ss.enum_active and ss.enum_results:
        st.subheader("🔍 Результаты расшифровки (перебор ключей):")
        keys_list = list(ss.enum_results.keys())
        ss.enum_selected_key = st.selectbox(
            "Выберите ключ для просмотра:",
            options=keys_list,
            index=keys_list.index(ss.enum_selected_key) if ss.enum_selected_key in keys_list else 0,
            format_func=lambda x: f"Ключ {x}",
            key="enum_selectbox_key"
        )
        st.code(ss.enum_results[ss.enum_selected_key], language="text")
        st.download_button(
            "Скачать все варианты",
            data="\n".join([f"Ключ {k}:\n{txt}\n" for k, txt in ss.enum_results.items()]),
            file_name="Шифр_Цезаря_Расшифровка_Все_Ключи.txt",
            mime="text/plain",
            key="dl_enum_all"
        )

    if action == "Расшифровка" and use_step and ss.dec_variants:
        st.subheader("🔓 Расшифрованный текст (по выбранным ключам):")
        for k in sorted(ss.dec_variants.keys()):
            st.markdown(f"**Ключ {k}:**")
            st.code(ss.dec_variants[k], language="text")
        st.download_button(
            "Скачать выбранные варианты",
            data="\n".join([f"Ключ {k}:\n{txt}\n" for k, txt in ss.dec_variants.items()]),
            file_name="Шифр_Цезаря_Расшифровка.txt",
            mime="text/plain",
            key="dl_dec_selected"
        )