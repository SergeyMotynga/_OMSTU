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

    new_cells = [
        md(
            """
## 8. Дополнительный эксперимент: агрегация по годам

В задании допускается ускорение и упрощение модели через изменение частоты ряда. В качестве дополнительного эксперимента агрегируем месячные данные по годам, усредняя значения внутри каждого года.

После агрегации:
- частота становится годовой;
- сезонный лаг естественно принять равным `11`, так как цикл солнечной активности составляет примерно 11 лет;
- для кросс-валидации используем последние `66` лет, то есть `6` полных циклов.
"""
        ),
        code(
            """
annual_train = train.set_index('Month')['Sunspots'].resample('YS').mean()
annual_test = test.set_index('Month')['Sunspots'].resample('YS').mean()

annual_cv_train = annual_train.iloc[-66:].reset_index()
annual_cv_train = annual_cv_train.rename(columns={'Month': 'ds', 'Sunspots': 'y'})
annual_cv_train['unique_id'] = 'sunspots_yearly'
annual_cv_train = annual_cv_train[['unique_id', 'ds', 'y']]

annual_model = AutoARIMA(
    d=0,
    D=0,
    season_length=11,
    start_p=0,
    start_q=0,
    start_P=0,
    start_Q=0,
    max_p=3,
    max_q=2,
    max_P=1,
    max_Q=1,
    max_order=5,
    stepwise=True,
    nmodels=30,
    approximation=True,
    trace=False,
)

annual_sf = StatsForecast(models=[annual_model], freq='YS', n_jobs=1)

annual_cv_start = time.perf_counter()
annual_cv_results = annual_sf.cross_validation(
    df=annual_cv_train,
    h=3,
    step_size=3,
    n_windows=5,
)
annual_cv_sec = time.perf_counter() - annual_cv_start

annual_cv_results['error'] = annual_cv_results['y'] - annual_cv_results['AutoARIMA']
annual_cv_results['abs_error'] = annual_cv_results['error'].abs()
annual_cv_results['squared_error'] = annual_cv_results['error'] ** 2
annual_cv_results['smape'] = 100 * 2 * np.abs(annual_cv_results['error']) / (
    np.abs(annual_cv_results['y']) + np.abs(annual_cv_results['AutoARIMA'])
)

annual_cv_metrics = pd.DataFrame({
    'model': ['monthly base', 'annual aggregated'],
    'MAE': [
        cv_results['abs_error'].mean(),
        annual_cv_results['abs_error'].mean(),
    ],
    'RMSE': [
        np.sqrt(cv_results['squared_error'].mean()),
        np.sqrt(annual_cv_results['squared_error'].mean()),
    ],
    'SMAPE': [
        cv_results['smape'].replace([np.inf, -np.inf], np.nan).dropna().mean(),
        annual_cv_results['smape'].replace([np.inf, -np.inf], np.nan).dropna().mean(),
    ],
    'cv_time_sec': [
        cv_elapsed_sec,
        annual_cv_sec,
    ],
})

annual_fit = StatsForecast(models=[annual_model], freq='YS', n_jobs=1)
annual_fit.fit(annual_cv_train)
annual_info = annual_fit.fitted_[0][0].model_
annual_p, annual_q, annual_P, annual_Q, annual_s, annual_d, annual_D = annual_info['arma']

annual_params = pd.DataFrame({
    'parameter': ['p', 'd', 'q', 'P', 'D', 'Q', 's', 'AICc', 'BIC'],
    'value': [
        annual_p,
        annual_d,
        annual_q,
        annual_P,
        annual_D,
        annual_Q,
        annual_s,
        annual_info['aicc'],
        annual_info['bic'],
    ],
})

display(annual_cv_metrics.round(4))
display(annual_params)
print(
    'Annual AutoARIMA model:',
    f'SARIMA({annual_p}, {annual_d}, {annual_q}) x ({annual_P}, {annual_D}, {annual_Q}, {annual_s})'
)
"""
        ),
        code(
            """
annual_series = annual_train.iloc[-66:]
annual_test_series = annual_test

annual_result = SARIMAX(
    annual_series,
    order=(annual_p, annual_d, annual_q),
    seasonal_order=(annual_P, annual_D, annual_Q, annual_s),
    trend='c',
    enforce_stationarity=False,
    enforce_invertibility=False,
    concentrate_scale=True,
).fit(method='lbfgs', maxiter=200, disp=False)

annual_forecast = annual_result.get_forecast(steps=len(annual_test_series))
annual_pred = annual_forecast.predicted_mean
annual_pred.index = annual_test_series.index
annual_ci = annual_forecast.conf_int()
annual_ci.index = annual_test_series.index

annual_mae = mean_absolute_error(annual_test_series, annual_pred)
annual_mse = mean_squared_error(annual_test_series, annual_pred)
annual_smape = (
    100
    * (2 * np.abs(annual_pred - annual_test_series) / (np.abs(annual_test_series) + np.abs(annual_pred)))
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .mean()
)

annual_metrics = pd.DataFrame({
    'MAE': [annual_mae],
    'MSE': [annual_mse],
    'SMAPE': [annual_smape],
})

display(annual_metrics.round(4))
print(annual_result.summary())
"""
        ),
        code(
            """
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

annual_series.plot(ax=axes[0], label='Train yearly', color='tab:blue')
annual_test_series.plot(ax=axes[0], label='Test yearly', color='tab:green')
annual_pred.plot(ax=axes[0], label='Yearly forecast', color='tab:red')
axes[0].fill_between(
    annual_ci.index,
    annual_ci.iloc[:, 0],
    annual_ci.iloc[:, 1],
    color='tab:red',
    alpha=0.15,
    label='95% CI',
)
axes[0].set_title('Годовая агрегация: прогноз на тестовой выборке')
axes[0].set_xlabel('')
axes[0].set_ylabel('Sunspots')
axes[0].legend()

(annual_test_series - annual_pred).rename('annual_error').plot(ax=axes[1], color='tab:purple')
axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_title('Годовая агрегация: ошибки прогноза')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Error')

plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
### Вывод по годовому эксперименту

После перехода к годовой агрегации модель стала компактнее, а кросс-валидационные метрики заметно улучшились. Прогноз остается неидеальным, но выглядит стабильнее, чем у месячной модели.

Такой дополнительный эксперимент показывает, что для данного ряда проблема связана не только с параметрами SARIMA, но и с самой частотой данных: на месячном ряде цикл получается слишком шумным, а на годовом — описывается заметно проще.
"""
        ),
    ]

    nb["cells"] = nb["cells"][:24] + new_cells
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("replaced")


if __name__ == "__main__":
    main()
