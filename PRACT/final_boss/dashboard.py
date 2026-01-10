import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_RandomForestClassifier.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
OHE_PATH = os.path.join(BASE_DIR, "models", "ohe.pkl")
LE_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")


NUMERIC_FEATURES = [
    'lw_x','lw_y','lw_z',
    'lh_x','lh_y','lh_z',
    'la_x','la_y','la_z',
    'ra_x','ra_y','ra_z',
    'age','height_in','weight_lbs'
]

BINARY_FEATURES = ['right_handed']

OHE_FEATURES = [
    'gender_male',
    'race_black',
    'race_caucasian'
]

MODEL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + OHE_FEATURES

@st.cache_resource
def load_assets():
    try:
        return {
            'model': load(MODEL_PATH),
            'scaler': load(SCALER_PATH),
            'ohe': load(OHE_PATH),
            'le': load(LE_PATH)
        }
    except Exception as e:
        st.error(f"Ошибка загрузки моделей:\n{e}")
        st.stop()

def prepare_features(df: pd.DataFrame, assets: dict) -> np.ndarray:
    scaler = assets['scaler']
    ohe = assets['ohe']

    df = df.drop(columns=['subj_id', 'time_s'], errors='ignore')

    try:
        cat_encoded = ohe.transform(df[['gender', 'race']])
        cat_df = pd.DataFrame(
            cat_encoded,
            columns=ohe.get_feature_names_out(['gender', 'race']),
            index=df.index
        )
        df = df.drop(columns=['gender', 'race']).join(cat_df)
    except Exception as e:
        st.error(f"Ошибка OneHotEncoder:\n{e}")
        st.stop()

    for col in OHE_FEATURES:
        if col not in df.columns:
            df[col] = 0

    required = NUMERIC_FEATURES + BINARY_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Отсутствуют обязательные признаки:\n{missing}")
        st.stop()

    try:
        df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])
    except Exception as e:
        st.error(f"Ошибка масштабирования:\n{e}")
        st.stop()

    return df[MODEL_FEATURES].astype(np.float64).values


def main():
    st.set_page_config(
        page_title="Профиль физической активности",
        layout="wide"
    )

    st.title("Рекомендательная система профиля активности")

    tab_info, tab_pred = st.tabs(["Информация", "Предсказание"])

    with tab_info:
        st.markdown('''# Общая характеристика приложения

Разработанное веб-приложение предназначено для автоматического определения типа физической активности человека на основе данных носимых акселерометров и демографических характеристик пользователя. Приложение реализует модель машинного обучения, которая по входным данным акселерометрии и персональным параметрам участника классифицирует текущий тип активности.

Приложение реализовано с использованием фреймворка **Streamlit**, что обеспечивает удобный пользовательский интерфейс, возможность загрузки пользовательских данных и визуализацию результатов предсказания.

---

# Описание исходного набора данных

В основе приложения лежит набор данных акселерометрии, собранный в рамках исследования  
«Идентификация ходьбы, подъема по лестнице и вождения с использованием носимых акселерометров», проведённого на кафедре биостатистики Школы общественного здравоохранения Фэрбенкса Университета Индианы.

## Характеристика данных

- Данные собраны для **200 здоровых взрослых людей**
- Использовались **3-осевые носимые акселерометры ActiGraph GT3X+**
- **Частота дискретизации:** 100 Гц
- Данные собирались одновременно в четырёх точках тела:
  - левое запястье
  - левое бедро
  - левая лодыжка
  - правая лодыжка
- Все данные обезличены
- Исследование было одобрено институциональным наблюдательным советом, участники дали информированное согласие

---

# Описание входных признаков

## 1. Временной признак

- `time_s` — время с момента запуска устройства (в секундах)

## 2. Акселерометрические данные

Измерения ускорения по трём осям (X, Y, Z), выраженные в единицах g (ускорение свободного падения):

- **Левое запястье:** `lw_x`, `lw_y`, `lw_z`
- **Левое бедро:** `lh_x`, `lh_y`, `lh_z`
- **Левая лодыжка:** `la_x`, `la_y`, `la_z`
- **Правая лодыжка:** `ra_x`, `ra_y`, `ra_z`

## 3. Демографические признаки

- `subj_id` — идентификатор участника  
- `gender` — пол участника  
- `age` — возраст  
- `height_in` — рост (в дюймах)  
- `weight_lbs` — вес (в фунтах)  
- `race` — расовая принадлежность  
- `right_handed` — доминирующая рука (1 — правша, 0 — левша)

---

# Целевой признак

Целевой переменной является **тип физической активности** (`activity`), который принимает следующие значения:

- `walking` — ходьба  
- `ascending` — подъем по лестнице  
- `descending` — спуск по лестнице  
- `driving` — вождение автомобиля  
- `clapping` — хлопки в ладоши  
- `non-study activities` — прочая активность
''')

    with tab_pred:
        uploaded_file = st.file_uploader(
            "Загрузите CSV-файл (разделитель — `$`)",
            type="csv"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, sep='$')
                st.success(f"Загружено строк: {len(df)}")

                with st.expander("Первые строки данных"):
                    st.dataframe(df.head())

                assets = load_assets()

                with st.spinner("Подготовка данных и предсказание..."):
                    X = prepare_features(df, assets)
                    preds_encoded = assets['model'].predict(X)
                    preds = assets['le'].inverse_transform(preds_encoded.astype(int))

                result = pd.DataFrame({
                    'subj_id': df.get('subj_id', range(len(df))),
                    'predicted_activity': preds
                })

                st.subheader("Результаты")
                st.dataframe(result)

                st.download_button(
                    "Скачать результаты",
                    data=result.to_csv(index=False).encode('utf-8'),
                    file_name="activity_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Ошибка:\n{e}")

if __name__ == "__main__":
    main()
