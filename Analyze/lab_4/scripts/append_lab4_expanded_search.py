from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "lab_4_sunspots_statsforecast.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    # Avoid appending twice.
    if any(
        cell.get("cell_type") == "markdown"
        and "Дополнительный эксперимент: расширенный поиск параметров" in "".join(cell.get("source", []))
        for cell in nb["cells"]
    ):
        print("already_appended")
        return

    new_cells = [
        md(
            """
## 8. Дополнительный эксперимент: расширенный поиск параметров

В основной части работы поиск был ограничен параметрами из лабораторной работы №3.

Дополнительно попробуем расширить границы поиска:
- `max_q = 2`
- `max_P = 1`
- `max_Q = 1`

Идея простая: проверить, не связано ли плохое качество прогноза только с слишком узким диапазоном параметров.
"""
        ),
        code(
            """
expanded_model = AutoARIMA(
    d=d,
    D=D,
    season_length=season_length,
    start_p=0,
    start_q=0,
    start_P=0,
    start_Q=0,
    max_p=6,
    max_q=2,
    max_P=1,
    max_Q=1,
    max_order=8,
    stepwise=True,
    nmodels=20,
    approximation=True,
    trace=False,
)

expanded_forecaster = StatsForecast(models=[expanded_model], freq='MS', n_jobs=1)

expanded_cv_start = time.perf_counter()
expanded_cv = expanded_forecaster.cross_validation(
    df=cv_train,
    h=12,
    step_size=12,
    n_windows=5,
)
expanded_cv_sec = time.perf_counter() - expanded_cv_start

expanded_cv['error'] = expanded_cv['y'] - expanded_cv['AutoARIMA']
expanded_cv['abs_error'] = expanded_cv['error'].abs()
expanded_cv['squared_error'] = expanded_cv['error'] ** 2
expanded_cv['smape'] = 100 * 2 * np.abs(expanded_cv['error']) / (
    np.abs(expanded_cv['y']) + np.abs(expanded_cv['AutoARIMA'])
)

expanded_cv_metrics = pd.DataFrame({
    'experiment': ['base search', 'expanded search'],
    'MAE': [
        cv_results['abs_error'].mean(),
        expanded_cv['abs_error'].mean(),
    ],
    'RMSE': [
        np.sqrt(cv_results['squared_error'].mean()),
        np.sqrt(expanded_cv['squared_error'].mean()),
    ],
    'SMAPE': [
        cv_results['smape'].replace([np.inf, -np.inf], np.nan).dropna().mean(),
        expanded_cv['smape'].replace([np.inf, -np.inf], np.nan).dropna().mean(),
    ],
})

expanded_fit_model = AutoARIMA(
    d=d,
    D=D,
    season_length=season_length,
    start_p=0,
    start_q=0,
    start_P=0,
    start_Q=0,
    max_p=6,
    max_q=2,
    max_P=1,
    max_Q=1,
    max_order=8,
    stepwise=True,
    nmodels=20,
    approximation=True,
    trace=False,
)

expanded_fit_forecaster = StatsForecast(models=[expanded_fit_model], freq='MS', n_jobs=1)
expanded_fit_forecaster.fit(cv_train)
expanded_info = expanded_fit_forecaster.fitted_[0][0].model_
expanded_p, expanded_q, expanded_P, expanded_Q, expanded_s, expanded_d, expanded_D = expanded_info['arma']

expanded_params = pd.DataFrame({
    'parameter': ['p', 'd', 'q', 'P', 'D', 'Q', 's', 'AICc', 'BIC', 'cv_time_sec'],
    'value': [
        expanded_p,
        expanded_d,
        expanded_q,
        expanded_P,
        expanded_D,
        expanded_Q,
        expanded_s,
        expanded_info['aicc'],
        expanded_info['bic'],
        expanded_cv_sec,
    ],
})

display(expanded_cv_metrics.round(4))
display(expanded_params)

print(
    'Expanded AutoARIMA model:',
    f'SARIMA({expanded_p}, {expanded_d}, {expanded_q}) x ({expanded_P}, {expanded_D}, {expanded_Q}, {expanded_s})'
)
"""
        ),
        md(
            """
### Вывод по дополнительному эксперименту

Если после расширения диапазона параметров модель остается той же или меняется незначительно, значит проблема не только в узком поиске. В таком случае слабый прогноз связан уже с ограничениями самого семейства ARIMA/SARIMA для этого ряда.

Если же расширенный поиск находит другую модель и метрики становятся лучше, это означает, что исходные границы поиска были слишком жесткими.
"""
        ),
    ]

    nb["cells"].extend(new_cells)
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("appended")


if __name__ == "__main__":
    main()
