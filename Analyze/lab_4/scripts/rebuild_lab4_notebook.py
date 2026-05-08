from __future__ import annotations

import json
import codecs
from pathlib import Path

import nbformat


def u(text: str) -> str:
    return codecs.decode(text, "unicode_escape")


ROOT = Path(__file__).resolve().parents[3]
LECTURE_PATH = next((ROOT / "Analyze" / "lab_4" / "docs").glob("*.ipynb"))
OUT_PATH = ROOT / "Analyze" / "lab_4" / "lab_4_sunspots_statsforecast.ipynb"


def lines(text: str) -> list[str]:
    return [line + "\n" for line in text.strip().splitlines()]


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(u(text)),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines(text),
    }


cells: list[dict] = []

cells.append(
    md_cell(
        r"""
# \u041b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u043d\u0430\u044f \u0440\u0430\u0431\u043e\u0442\u0430 \u21164

## \u041e\u0431\u0443\u0447\u0435\u043d\u0438\u0435 \u0438 \u0430\u043d\u0430\u043b\u0438\u0437 \u043c\u043e\u0434\u0435\u043b\u0438 SARIMA \u0434\u043b\u044f \u0440\u044f\u0434\u0430 \u0441\u043e\u043b\u043d\u0435\u0447\u043d\u043e\u0439 \u0430\u043a\u0442\u0438\u0432\u043d\u043e\u0441\u0442\u0438

\u0412 \u044d\u0442\u043e\u0439 \u0440\u0430\u0431\u043e\u0442\u0435 \u043f\u0440\u043e\u0434\u043e\u043b\u0436\u0430\u0435\u043c \u0430\u043d\u0430\u043b\u0438\u0437 \u0434\u0430\u043d\u043d\u044b\u0445 \u0438\u0437 \u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u043d\u043e\u0439 \u0440\u0430\u0431\u043e\u0442\u044b \u21163.
\u0418\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u043c \u043f\u0440\u0435\u0434\u043e\u0431\u0440\u0430\u0431\u043e\u0442\u0430\u043d\u043d\u044b\u0435 train/test-\u0434\u0430\u043d\u043d\u044b\u0435 \u0438 \u0433\u0440\u0430\u043d\u0438\u0446\u044b \u043f\u043e\u0438\u0441\u043a\u0430 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u043e\u0432, \u043f\u043e\u043b\u0443\u0447\u0435\u043d\u043d\u044b\u0435 \u0440\u0430\u043d\u0435\u0435:

- `max_p = 6`
- `d = 0`
- `max_q = 0`
- `max_P = 0`
- `D = 0`
- `max_Q = 1`
- `season_length = 132`

\u0422\u0430\u043a \u043a\u0430\u043a \u043c\u0435\u0441\u044f\u0447\u043d\u044b\u0439 \u0440\u044f\u0434 \u0434\u043b\u0438\u043d\u043d\u044b\u0439, \u0434\u043b\u044f \u043a\u0440\u043e\u0441\u0441-\u0432\u0430\u043b\u0438\u0434\u0430\u0446\u0438\u0438 \u0431\u0435\u0440\u0435\u043c \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0435 `396` \u043c\u0435\u0441\u044f\u0446\u0435\u0432 train-\u0432\u044b\u0431\u043e\u0440\u043a\u0438, \u0442\u043e \u0435\u0441\u0442\u044c `3` \u0441\u043e\u043b\u043d\u0435\u0447\u043d\u044b\u0445 \u0446\u0438\u043a\u043b\u0430 \u043f\u043e `132` \u043c\u0435\u0441\u044f\u0446\u0430. \u042d\u0442\u043e \u0443\u0441\u043a\u043e\u0440\u044f\u0435\u0442 \u0432\u044b\u0447\u0438\u0441\u043b\u0435\u043d\u0438\u044f \u0438 \u043f\u0440\u0438 \u044d\u0442\u043e\u043c \u0441\u043e\u0445\u0440\u0430\u043d\u044f\u0435\u0442 \u043d\u0435\u0441\u043a\u043e\u043b\u044c\u043a\u043e \u043f\u043e\u043b\u043d\u044b\u0445 \u0446\u0438\u043a\u043b\u043e\u0432 \u0432 \u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0438\u0445 \u0434\u0430\u043d\u043d\u044b\u0445.
"""
    )
)

cells.append(
    code_cell(
        r"""
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('tab10')
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
"""
    )
)

cells.append(
    md_cell(
        r"""
## 1. \u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u0434\u0430\u043d\u043d\u044b\u0445

\u0417\u0430\u0433\u0440\u0443\u0436\u0430\u0435\u043c \u043f\u0440\u0435\u0434\u043e\u0431\u0440\u0430\u0431\u043e\u0442\u0430\u043d\u043d\u044b\u0435 `train` \u0438 `test` \u0438\u0437 \u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u043d\u043e\u0439 \u0440\u0430\u0431\u043e\u0442\u044b \u21162.
\u0412 \u0441\u0442\u043e\u043b\u0431\u0446\u0435 `Sunspots` \u043f\u043e\u0441\u043b\u0435 \u0438\u043d\u0442\u0435\u0440\u043f\u043e\u043b\u044f\u0446\u0438\u0438 \u0432\u0441\u0442\u0440\u0435\u0447\u0430\u044e\u0442\u0441\u044f \u0434\u0432\u0430 \u043e\u0442\u0440\u0438\u0446\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0445 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f, \u043f\u043e\u044d\u0442\u043e\u043c\u0443 \u043f\u0435\u0440\u0435\u0434 \u043c\u043e\u0434\u0435\u043b\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435\u043c \u043e\u0433\u0440\u0430\u043d\u0438\u0447\u0438\u043c \u0440\u044f\u0434 \u0441\u043d\u0438\u0437\u0443 \u043d\u0443\u043b\u0435\u043c.
"""
    )
)

cells.append(
    code_cell(
        r"""
train = pd.read_csv('Analyze/lab_2/sunspots_train_preprocessed.csv', parse_dates=['Month'])
test = pd.read_csv('Analyze/lab_2/sunspots_test_preprocessed.csv', parse_dates=['Month'])

negative_before_clip = int((train['Sunspots'] < 0).sum() + (test['Sunspots'] < 0).sum())

for df in (train, test):
    df['Sunspots'] = df['Sunspots'].clip(lower=0)

series = train.set_index('Month')['Sunspots'].asfreq('MS')
test_series = test.set_index('Month')['Sunspots'].asfreq('MS')
full_series = pd.concat([series, test_series])

summary_df = pd.DataFrame({
    'dataset': ['train', 'test'],
    'start': [series.index.min(), test_series.index.min()],
    'end': [series.index.max(), test_series.index.max()],
    'n_obs': [len(series), len(test_series)],
})

display(summary_df)
print(f'Количество отрицательных значений до clip: {negative_before_clip}')
print(f'Итоговый размер полного ряда: {len(full_series)} наблюдений')
"""
    )
)

cells.append(
    md_cell(
        r"""
## 2. \u041f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u043a\u0430 \u0434\u0430\u043d\u043d\u044b\u0445 \u0434\u043b\u044f \u043a\u0440\u043e\u0441\u0441-\u0432\u0430\u043b\u0438\u0434\u0430\u0446\u0438\u0438

\u0418\u0437 \u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u043d\u043e\u0439 \u0440\u0430\u0431\u043e\u0442\u044b \u21163 \u0431\u0435\u0440\u0435\u043c \u0433\u0440\u0430\u043d\u0438\u0446\u044b \u043f\u043e\u0438\u0441\u043a\u0430:

- `max_p = 6`
- `d = 0`
- `max_q = 0`
- `max_P = 0`
- `D = 0`
- `max_Q = 1`
- `s = 132`

\u0414\u043b\u044f `StatsForecast.cross_validation(...)` \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u043c \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0435 `396` \u043c\u0435\u0441\u044f\u0446\u0435\u0432 \u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0435\u0439 \u0432\u044b\u0431\u043e\u0440\u043a\u0438. \u042d\u0442\u043e \u0440\u043e\u0432\u043d\u043e `3` \u0441\u043e\u043b\u043d\u0435\u0447\u043d\u044b\u0445 \u0446\u0438\u043a\u043b\u0430, \u0447\u0442\u043e \u0443\u0434\u043e\u0432\u043b\u0435\u0442\u0432\u043e\u0440\u044f\u0435\u0442 \u0443\u0441\u043b\u043e\u0432\u0438\u044e \u0437\u0430\u0434\u0430\u043d\u0438\u044f \u0438 \u0434\u0435\u043b\u0430\u0435\u0442 \u0432\u044b\u0447\u0438\u0441\u043b\u0435\u043d\u0438\u044f \u0440\u0435\u0430\u043b\u0438\u0441\u0442\u0438\u0447\u043d\u044b\u043c\u0438 \u043f\u043e \u0432\u0440\u0435\u043c\u0435\u043d\u0438.
"""
    )
)

cells.append(
    code_cell(
        r"""
max_p, d, max_q = 6, 0, 0
max_P, D, max_Q, season_length = 0, 0, 1, 132

cv_train = train[['Month', 'Sunspots']].rename(columns={'Month': 'ds', 'Sunspots': 'y'}).copy()
cv_train['unique_id'] = 'sunspots'
cv_train = cv_train[['unique_id', 'ds', 'y']].iloc[-396:].reset_index(drop=True)

cv_info = pd.DataFrame({
    'start': [cv_train['ds'].min()],
    'end': [cv_train['ds'].max()],
    'n_obs': [len(cv_train)],
    'season_length': [season_length],
    'cycles_in_cv_train': [len(cv_train) / season_length],
})

display(cv_info)
"""
    )
)

cells.append(
    md_cell(
        r"""
## 3. \u041a\u0440\u043e\u0441\u0441-\u0432\u0430\u043b\u0438\u0434\u0430\u0446\u0438\u044f \u0441 AutoARIMA \u0438\u0437 `statsforecast`

\u041e\u0433\u0440\u0430\u043d\u0438\u0447\u0438\u0432\u0430\u0435\u043c \u043f\u043e\u0438\u0441\u043a \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u043e\u0432 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f\u043c\u0438 \u0438\u0437 \u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u043d\u043e\u0439 \u0440\u0430\u0431\u043e\u0442\u044b \u21163.
\u0414\u043b\u044f rolling forecast validation \u0431\u0435\u0440\u0435\u043c:

- \u0433\u043e\u0440\u0438\u0437\u043e\u043d\u0442 \u043f\u0440\u043e\u0433\u043d\u043e\u0437\u0430 `h = 12` \u043c\u0435\u0441\u044f\u0446\u0435\u0432;
- \u0448\u0430\u0433 \u043e\u043a\u043d\u0430 `step_size = 12` \u043c\u0435\u0441\u044f\u0446\u0435\u0432;
- \u043a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432\u043e \u043e\u043a\u043e\u043d `n_windows = 5`.

\u0422\u0430\u043a \u043c\u044b \u043f\u043e\u043b\u0443\u0447\u0430\u0435\u043c \u0441\u0435\u0440\u0438\u044e \u0433\u043e\u0434\u043e\u0432\u044b\u0445 \u043f\u0440\u043e\u0433\u043d\u043e\u0437\u043e\u0432 \u0438 \u043c\u043e\u0436\u0435\u043c \u043f\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u0442\u044c, \u043a\u0430\u043a \u043c\u0435\u043d\u044f\u0435\u0442\u0441\u044f \u043e\u0448\u0438\u0431\u043a\u0430 \u0441\u043e \u0432\u0440\u0435\u043c\u0435\u043d\u0435\u043c.
"""
    )
)

cells.append(
    code_cell(
        r"""
cv_model = AutoARIMA(
    d=d,
    D=D,
    season_length=season_length,
    start_p=0,
    start_q=0,
    start_P=0,
    start_Q=0,
    max_p=max_p,
    max_q=max_q,
    max_P=max_P,
    max_Q=max_Q,
    max_order=max_p,
    stepwise=True,
    nmodels=12,
    approximation=True,
    trace=False,
)

cv_forecaster = StatsForecast(models=[cv_model], freq='MS', n_jobs=1)

cv_start = time.perf_counter()
cv_results = cv_forecaster.cross_validation(
    df=cv_train,
    h=12,
    step_size=12,
    n_windows=5,
)
cv_elapsed_sec = time.perf_counter() - cv_start

cv_results['error'] = cv_results['y'] - cv_results['AutoARIMA']
cv_results['abs_error'] = cv_results['error'].abs()
cv_results['squared_error'] = cv_results['error'] ** 2
cv_results['ape'] = np.where(
    cv_results['y'] != 0,
    np.abs(cv_results['error'] / cv_results['y']) * 100,
    np.nan,
)
cv_results['smape'] = 100 * 2 * np.abs(cv_results['error']) / (
    np.abs(cv_results['y']) + np.abs(cv_results['AutoARIMA'])
)

cv_metrics = pd.DataFrame({
    'MAE': [cv_results['abs_error'].mean()],
    'RMSE': [np.sqrt(cv_results['squared_error'].mean())],
    'MAPE': [cv_results['ape'].dropna().mean()],
    'SMAPE': [cv_results['smape'].replace([np.inf, -np.inf], np.nan).dropna().mean()],
    'fit_time_sec': [cv_elapsed_sec],
})

cv_by_cutoff = cv_results.groupby('cutoff')[['abs_error', 'squared_error', 'ape', 'smape']].mean()
cv_by_cutoff = cv_by_cutoff.rename(columns={
    'abs_error': 'MAE_by_window',
    'squared_error': 'MSE_by_window',
    'ape': 'MAPE_by_window',
    'smape': 'SMAPE_by_window',
})

print('Первые строки результатов cross-validation:')
display(cv_results.head())
print('Сводные метрики по всем rolling forecast прогнозам:')
display(cv_metrics.round(4))
print('Средние ошибки по отдельным окнам кросс-валидации:')
display(cv_by_cutoff.round(4))
"""
    )
)

cells.append(
    code_cell(
        r"""
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

sns.lineplot(
    data=cv_results,
    x='ds',
    y='abs_error',
    hue='cutoff',
    marker='o',
    ax=axes[0],
)
axes[0].set_title('Rolling forecast error plot: абсолютная ошибка во времени')
axes[0].set_xlabel('')
axes[0].set_ylabel('Absolute error')
axes[0].legend(title='cutoff', ncol=3)

sns.lineplot(
    data=cv_results,
    x='ds',
    y='error',
    hue='cutoff',
    marker='o',
    ax=axes[1],
    legend=False,
)
axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_title('Знак ошибки во времени')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('y - y_pred')

plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md_cell(
        r"""
### \u0412\u044b\u0432\u043e\u0434 \u043f\u043e \u043a\u0440\u043e\u0441\u0441-\u0432\u0430\u043b\u0438\u0434\u0430\u0446\u0438\u0438

\u041f\u043e rolling forecast error plot \u0432\u0438\u0434\u043d\u043e, \u0447\u0442\u043e \u043e\u0448\u0438\u0431\u043a\u0430 \u0437\u0430\u043c\u0435\u0442\u043d\u043e \u043c\u0435\u043d\u044f\u0435\u0442\u0441\u044f \u043e\u0442 \u043e\u043a\u043d\u0430 \u043a \u043e\u043a\u043d\u0443: \u0432 \u043e\u0434\u043d\u0438 \u043f\u0435\u0440\u0438\u043e\u0434\u044b \u043f\u0440\u043e\u0433\u043d\u043e\u0437 \u0430\u043a\u043a\u0443\u0440\u0430\u0442\u043d\u044b\u0439, \u0430 \u0432 \u0434\u0440\u0443\u0433\u0438\u0435 \u0440\u0435\u0437\u043a\u043e \u0443\u0445\u0443\u0434\u0448\u0430\u0435\u0442\u0441\u044f. \u0417\u043d\u0430\u0447\u0438\u0442, \u043f\u0440\u0435\u0434\u0441\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c\u043d\u0430\u044f \u0441\u043f\u043e\u0441\u043e\u0431\u043d\u043e\u0441\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u0438 \u043d\u0435\u0441\u0442\u0430\u0431\u0438\u043b\u044c\u043d\u0430 \u0438 \u0437\u0430\u0432\u0438\u0441\u0438\u0442 \u043e\u0442 \u0444\u0430\u0437\u044b \u0441\u043e\u043b\u043d\u0435\u0447\u043d\u043e\u0433\u043e \u0446\u0438\u043a\u043b\u0430.

\u0417\u043d\u0430\u0447\u0435\u043d\u0438\u0435 `MAPE` \u043f\u043e\u043b\u0443\u0447\u0430\u0435\u0442\u0441\u044f \u043e\u0447\u0435\u043d\u044c \u0431\u043e\u043b\u044c\u0448\u0438\u043c, \u043f\u043e\u0442\u043e\u043c\u0443 \u0447\u0442\u043e \u0432 \u043c\u0438\u043d\u0438\u043c\u0443\u043c\u0430\u0445 \u0446\u0438\u043a\u043b\u0430 \u0438\u0441\u0442\u0438\u043d\u043d\u044b\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f \u0431\u043b\u0438\u0437\u043a\u0438 \u043a \u043d\u0443\u043b\u044e. \u041f\u043e\u044d\u0442\u043e\u043c\u0443 \u0434\u043b\u044f \u044d\u0442\u043e\u0433\u043e \u0440\u044f\u0434\u0430 \u0431\u043e\u043b\u0435\u0435 \u0438\u043d\u0444\u043e\u0440\u043c\u0430\u0442\u0438\u0432\u043d\u044b `MAE`, `RMSE` \u0438 `SMAPE`.
"""
    )
)

cells.append(
    md_cell(
        r"""
## 4. \u041f\u043e\u0432\u0442\u043e\u0440\u043d\u043e\u0435 \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u0435 AutoARIMA \u0438 \u0432\u044b\u0431\u043e\u0440 \u043b\u0443\u0447\u0448\u0438\u0445 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u043e\u0432

\u0422\u0435\u043f\u0435\u0440\u044c \u043e\u0431\u0443\u0447\u0430\u0435\u043c `AutoARIMA` \u043d\u0430 \u0442\u0435\u0445 \u0436\u0435 `396` \u043d\u0430\u0431\u043b\u044e\u0434\u0435\u043d\u0438\u044f\u0445, \u043d\u0430 \u043a\u043e\u0442\u043e\u0440\u044b\u0445 \u0432\u044b\u043f\u043e\u043b\u043d\u044f\u043b\u0430\u0441\u044c \u043a\u0440\u043e\u0441\u0441-\u0432\u0430\u043b\u0438\u0434\u0430\u0446\u0438\u044f, \u0438 \u0438\u0437\u0432\u043b\u0435\u043a\u0430\u0435\u043c \u043b\u0443\u0447\u0448\u0443\u044e \u043a\u043e\u043d\u0444\u0438\u0433\u0443\u0440\u0430\u0446\u0438\u044e.
"""
    )
)

cells.append(
    code_cell(
        r"""
auto_model = AutoARIMA(
    d=d,
    D=D,
    season_length=season_length,
    start_p=0,
    start_q=0,
    start_P=0,
    start_Q=0,
    max_p=max_p,
    max_q=max_q,
    max_P=max_P,
    max_Q=max_Q,
    max_order=max_p,
    stepwise=True,
    nmodels=12,
    approximation=True,
    trace=False,
)

auto_forecaster = StatsForecast(models=[auto_model], freq='MS', n_jobs=1)

auto_start = time.perf_counter()
auto_forecaster.fit(cv_train)
auto_elapsed_sec = time.perf_counter() - auto_start

model_info = auto_forecaster.fitted_[0][0].model_
p_best, q_best, P_best, Q_best, s_best, d_best, D_best = model_info['arma']

best_params = pd.DataFrame({
    'parameter': ['p', 'd', 'q', 'P', 'D', 'Q', 's', 'AICc', 'BIC', 'fit_time_sec'],
    'value': [p_best, d_best, q_best, P_best, D_best, Q_best, s_best, model_info['aicc'], model_info['bic'], auto_elapsed_sec],
})

display(best_params)
print('Найденная AutoARIMA модель:')
print(f'SARIMA({p_best}, {d_best}, {q_best}) x ({P_best}, {D_best}, {Q_best}, {s_best})')
"""
    )
)

cells.append(
    md_cell(
        r"""
### \u0412\u044b\u0431\u0440\u0430\u043d\u043d\u0430\u044f \u043c\u043e\u0434\u0435\u043b\u044c

\u0410\u0432\u0442\u043e\u043c\u0430\u0442\u0438\u0447\u0435\u0441\u043a\u0438\u0439 \u043f\u043e\u0434\u0431\u043e\u0440 \u043d\u0430 \u043e\u0441\u043d\u043e\u0432\u0435 `statsforecast` \u0432\u044b\u0431\u0440\u0430\u043b \u043c\u043e\u0434\u0435\u043b\u044c:

`SARIMA(4, 0, 0) x (0, 0, 1, 132)`

\u041e\u043d\u0430 \u043d\u0435\u043c\u043d\u043e\u0433\u043e \u043f\u0440\u043e\u0449\u0435, \u0447\u0435\u043c \u0440\u0443\u0447\u043d\u0430\u044f \u043a\u043e\u043d\u0444\u0438\u0433\u0443\u0440\u0430\u0446\u0438\u044f \u0438\u0437 \u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u043d\u043e\u0439 \u0440\u0430\u0431\u043e\u0442\u044b \u21163, \u043d\u043e \u043e\u0441\u0442\u0430\u0435\u0442\u0441\u044f \u0432 \u0442\u043e\u043c \u0436\u0435 \u0441\u0435\u043c\u0435\u0439\u0441\u0442\u0432\u0435: \u043e\u0431\u044b\u0447\u043d\u0430\u044f AR-\u043a\u043e\u043c\u043f\u043e\u043d\u0435\u043d\u0442\u0430 \u0438 \u0441\u0435\u0437\u043e\u043d\u043d\u0430\u044f MA-\u043a\u043e\u043c\u043f\u043e\u043d\u0435\u043d\u0442\u0430 \u0441 \u043f\u0435\u0440\u0438\u043e\u0434\u043e\u043c `132` \u043c\u0435\u0441\u044f\u0446\u0430.
"""
    )
)

cells.append(
    md_cell(
        r"""
## 5. \u041e\u0431\u0443\u0447\u0435\u043d\u0438\u0435 \u0438\u0442\u043e\u0433\u043e\u0432\u043e\u0439 \u043c\u043e\u0434\u0435\u043b\u0438 \u0432 `statsmodels`

\u041f\u043e \u0443\u0441\u043b\u043e\u0432\u0438\u044e \u0437\u0430\u0434\u0430\u043d\u0438\u044f \u0441\u0442\u0440\u043e\u0438\u043c \u043c\u043e\u0434\u0435\u043b\u044c \u0441 \u043b\u0443\u0447\u0448\u0438\u043c\u0438 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u0430\u043c\u0438 \u0443\u0436\u0435 \u0432 `statsmodels`, \u0432\u044b\u0432\u043e\u0434\u0438\u043c `summary()` \u0438 \u0430\u043d\u0430\u043b\u0438\u0437\u0438\u0440\u0443\u0435\u043c \u0437\u043d\u0430\u0447\u0438\u043c\u043e\u0441\u0442\u044c \u043a\u043e\u044d\u0444\u0444\u0438\u0446\u0438\u0435\u043d\u0442\u043e\u0432.
"""
    )
)

cells.append(
    code_cell(
        r"""
sm_start = time.perf_counter()
final_model = SARIMAX(
    series,
    order=(p_best, d_best, q_best),
    seasonal_order=(P_best, D_best, Q_best, s_best),
    trend='c',
    enforce_stationarity=False,
    enforce_invertibility=False,
    concentrate_scale=True,
)
final_result = final_model.fit(method='lbfgs', maxiter=200, disp=False)
sm_elapsed_min = (time.perf_counter() - sm_start) / 60

print(f'Время обучения statsmodels-модели: {sm_elapsed_min:.2f} мин')
print(final_result.summary())
"""
    )
)

cells.append(
    code_cell(
        r"""
coef_table = pd.concat([
    final_result.params.rename('coef'),
    final_result.bse.rename('std_err'),
    final_result.pvalues.rename('p_value'),
    final_result.conf_int().rename(columns={0: 'ci_low', 1: 'ci_high'}),
], axis=1)
coef_table['significant_05'] = coef_table['p_value'] < 0.05

lb_table = acorr_ljungbox(final_result.resid.dropna(), lags=[12, 24, 36], return_df=True)

print('Таблица коэффициентов:')
display(coef_table.round(6))
print('Ljung-Box для остатков:')
display(lb_table.round(6))
"""
    )
)

cells.append(
    md_cell(
        r"""
### \u0412\u044b\u0432\u043e\u0434 \u043f\u043e \u043a\u043e\u044d\u0444\u0444\u0438\u0446\u0438\u0435\u043d\u0442\u0430\u043c

\u0412\u0441\u0435 \u043a\u043e\u044d\u0444\u0444\u0438\u0446\u0438\u0435\u043d\u0442\u044b \u0438\u0442\u043e\u0433\u043e\u0432\u043e\u0439 \u043c\u043e\u0434\u0435\u043b\u0438 \u0437\u043d\u0430\u0447\u0438\u043c\u044b \u043d\u0430 \u0443\u0440\u043e\u0432\u043d\u0435 `0.05`: \u0443 \u0432\u0441\u0435\u0445 `p-value < 0.05`, \u0430 \u0434\u043e\u0432\u0435\u0440\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0435 \u0438\u043d\u0442\u0435\u0440\u0432\u0430\u043b\u044b \u043d\u0435 \u0441\u043e\u0434\u0435\u0440\u0436\u0430\u0442 \u043d\u043e\u043b\u044c. \u042d\u0442\u043e \u0437\u043d\u0430\u0447\u0438\u0442, \u0447\u0442\u043e \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u043c\u043e\u0434\u0435\u043b\u0438 \u0441\u0442\u0430\u0442\u0438\u0441\u0442\u0438\u0447\u0435\u0441\u043a\u0438 \u043e\u0431\u043e\u0441\u043d\u043e\u0432\u0430\u043d\u044b.

\u041d\u043e \u0441\u0430\u043c \u043f\u043e \u0441\u0435\u0431\u0435 \u044d\u0442\u043e\u0442 \u0444\u0430\u043a\u0442 \u0435\u0449\u0435 \u043d\u0435 \u0434\u0435\u043b\u0430\u0435\u0442 \u043c\u043e\u0434\u0435\u043b\u044c \u0438\u0434\u0435\u0430\u043b\u044c\u043d\u043e\u0439: \u043d\u0443\u0436\u043d\u043e \u043f\u0440\u043e\u0432\u0435\u0440\u0438\u0442\u044c, \u043d\u0430\u0441\u043a\u043e\u043b\u044c\u043a\u043e \u0441\u043b\u0443\u0447\u0430\u0439\u043d\u044b \u043e\u0441\u0442\u0430\u0442\u043a\u0438.
"""
    )
)

cells.append(
    code_cell(
        r"""
final_result.plot_diagnostics(figsize=(14, 10))
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md_cell(
        r"""
### \u0412\u044b\u0432\u043e\u0434 \u043f\u043e \u0434\u0438\u0430\u0433\u043d\u043e\u0441\u0442\u0438\u043a\u0435 \u043e\u0441\u0442\u0430\u0442\u043a\u043e\u0432

\u041f\u043e \u0434\u0438\u0430\u0433\u043d\u043e\u0441\u0442\u0438\u0447\u0435\u0441\u043a\u0438\u043c \u0433\u0440\u0430\u0444\u0438\u043a\u0430\u043c \u0438 \u043f\u043e \u0442\u0435\u0441\u0442\u0443 Ljung-Box \u0432\u0438\u0434\u043d\u043e, \u0447\u0442\u043e \u043e\u0441\u0442\u0430\u0442\u043a\u0438 \u043d\u0435\u043b\u044c\u0437\u044f \u0441\u0447\u0438\u0442\u0430\u0442\u044c \u043f\u043e\u043b\u043d\u043e\u0441\u0442\u044c\u044e \u0441\u043b\u0443\u0447\u0430\u0439\u043d\u044b\u043c\u0438:

- \u0432 \u043e\u0441\u0442\u0430\u0442\u043a\u0430\u0445 \u0441\u043e\u0445\u0440\u0430\u043d\u044f\u0435\u0442\u0441\u044f \u0430\u0432\u0442\u043e\u043a\u043e\u0440\u0440\u0435\u043b\u044f\u0446\u0438\u043e\u043d\u043d\u0430\u044f \u0441\u0442\u0440\u0443\u043a\u0442\u0443\u0440\u0430;
- `p-value` \u0442\u0435\u0441\u0442\u0430 Ljung-Box \u043d\u0430 \u0440\u044f\u0434\u0435 \u043b\u0430\u0433\u043e\u0432 \u043e\u0447\u0435\u043d\u044c \u043c\u0430\u043b\u044b;
- \u0440\u0430\u0441\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435 \u043e\u0448\u0438\u0431\u043e\u043a \u043e\u0442\u043a\u043b\u043e\u043d\u044f\u0435\u0442\u0441\u044f \u043e\u0442 \u043d\u043e\u0440\u043c\u0430\u043b\u044c\u043d\u043e\u0433\u043e.

\u0417\u043d\u0430\u0447\u0438\u0442, \u043c\u043e\u0434\u0435\u043b\u044c \u043f\u043e\u043b\u0435\u0437\u043d\u0430 \u043a\u0430\u043a \u0440\u0430\u0431\u043e\u0447\u0435\u0435 \u043f\u0440\u0438\u0431\u043b\u0438\u0436\u0435\u043d\u0438\u0435, \u043d\u043e \u043d\u0435 \u043e\u043f\u0438\u0441\u044b\u0432\u0430\u0435\u0442 \u0440\u044f\u0434 \u0438\u0434\u0435\u0430\u043b\u044c\u043d\u043e.
"""
    )
)

cells.append(md_cell(r"""## 6. \u041f\u0440\u043e\u0433\u043d\u043e\u0437 \u043d\u0430 \u0442\u0435\u0441\u0442\u043e\u0432\u043e\u0439 \u0432\u044b\u0431\u043e\u0440\u043a\u0435 \u0438 \u043c\u0435\u0442\u0440\u0438\u043a\u0438 \u043a\u0430\u0447\u0435\u0441\u0442\u0432\u0430"""))

cells.append(
    code_cell(
        r"""
forecast = final_result.get_forecast(steps=len(test_series))
pred = forecast.predicted_mean
pred.index = test_series.index
conf_int = forecast.conf_int()
conf_int.index = test_series.index

mae = mean_absolute_error(test_series, pred)
mse = mean_squared_error(test_series, pred)
mask = test_series != 0
mape = (np.abs((test_series[mask] - pred[mask]) / test_series[mask]).mean() * 100)
smape = (
    100
    * (2 * np.abs(pred - test_series) / (np.abs(test_series) + np.abs(pred)))
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .mean()
)

metrics = pd.DataFrame({
    'MAE': [mae],
    'MSE': [mse],
    'MAPE': [mape],
    'SMAPE': [smape],
})

display(metrics.round(4))
"""
    )
)

cells.append(
    code_cell(
        r"""
fig, ax = plt.subplots(figsize=(15, 7))
series.iloc[-8 * 12:].plot(ax=ax, label='Train (tail)', color='tab:blue')
test_series.plot(ax=ax, label='Test', color='tab:green')
pred.plot(ax=ax, label='SARIMA forecast', color='tab:red')
ax.fill_between(
    conf_int.index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    color='tab:red',
    alpha=0.15,
    label='95% CI',
)
ax.set_title('Прогноз итоговой SARIMA-модели на тестовой выборке')
ax.set_xlabel('Month')
ax.set_ylabel('Sunspots')
ax.legend()
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    code_cell(
        r"""
forecast_errors = (test_series - pred).rename('forecast_error')

fig, ax = plt.subplots(figsize=(15, 5))
forecast_errors.plot(ax=ax, color='tab:purple')
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_title('Ошибки прогноза во времени (test - forecast)')
ax.set_xlabel('Month')
ax.set_ylabel('Error')
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md_cell(
        r"""
## 7. \u0418\u0442\u043e\u0433\u043e\u0432\u044b\u0439 \u0432\u044b\u0432\u043e\u0434

1. \u0414\u043b\u044f \u043a\u0440\u043e\u0441\u0441-\u0432\u0430\u043b\u0438\u0434\u0430\u0446\u0438\u0438 \u0431\u044b\u043b\u0430 \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u043d\u0430 \u0431\u0438\u0431\u043b\u0438\u043e\u0442\u0435\u043a\u0430 `statsforecast` \u0438 \u043c\u043e\u0434\u0435\u043b\u044c `AutoARIMA`.
2. \u0418\u0437-\u0437\u0430 \u0432\u044b\u0441\u043e\u043a\u043e\u0439 \u0432\u044b\u0447\u0438\u0441\u043b\u0438\u0442\u0435\u043b\u044c\u043d\u043e\u0439 \u0441\u043b\u043e\u0436\u043d\u043e\u0441\u0442\u0438 SARIMA \u043d\u0430 \u043c\u0435\u0441\u044f\u0447\u043d\u043e\u043c \u0440\u044f\u0434\u0435 \u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0430\u044f \u0432\u044b\u0431\u043e\u0440\u043a\u0430 \u0431\u044b\u043b\u0430 \u0441\u043e\u043a\u0440\u0430\u0449\u0435\u043d\u0430 \u0434\u043e \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0445 `396` \u043c\u0435\u0441\u044f\u0446\u0435\u0432, \u0442\u043e \u0435\u0441\u0442\u044c \u0434\u043e `3` \u0441\u043e\u043b\u043d\u0435\u0447\u043d\u044b\u0445 \u0446\u0438\u043a\u043b\u043e\u0432.
3. \u041f\u043e \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u0430\u043c `AutoARIMA` \u043b\u0443\u0447\u0448\u0435\u0439 \u043e\u043a\u0430\u0437\u0430\u043b\u0430\u0441\u044c \u043c\u043e\u0434\u0435\u043b\u044c `SARIMA(4, 0, 0) x (0, 0, 1, 132)`.
4. \u0412 \u043c\u043e\u0434\u0435\u043b\u0438 `statsmodels` \u0432\u0441\u0435 \u043e\u0446\u0435\u043d\u0435\u043d\u043d\u044b\u0435 \u043a\u043e\u044d\u0444\u0444\u0438\u0446\u0438\u0435\u043d\u0442\u044b \u043e\u043a\u0430\u0437\u0430\u043b\u0438\u0441\u044c \u0441\u0442\u0430\u0442\u0438\u0441\u0442\u0438\u0447\u0435\u0441\u043a\u0438 \u0437\u043d\u0430\u0447\u0438\u043c\u044b\u043c\u0438.
5. \u041f\u0440\u0438 \u044d\u0442\u043e\u043c \u0434\u0438\u0430\u0433\u043d\u043e\u0441\u0442\u0438\u043a\u0430 \u043e\u0441\u0442\u0430\u0442\u043a\u043e\u0432 \u043f\u043e\u043a\u0430\u0437\u0430\u043b\u0430, \u0447\u0442\u043e \u043e\u0448\u0438\u0431\u043a\u0438 \u043d\u0435 \u044f\u0432\u043b\u044f\u044e\u0442\u0441\u044f \u043f\u043e\u043b\u043d\u043e\u0441\u0442\u044c\u044e \u0441\u043b\u0443\u0447\u0430\u0439\u043d\u044b\u043c\u0438, \u0442\u043e \u0435\u0441\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c \u043d\u0435 \u043e\u043f\u0438\u0441\u044b\u0432\u0430\u0435\u0442 \u043f\u0440\u043e\u0446\u0435\u0441\u0441 \u0438\u0434\u0435\u0430\u043b\u044c\u043d\u043e.
6. \u041d\u0430 \u0442\u0435\u0441\u0442\u043e\u0432\u043e\u043c \u0438\u043d\u0442\u0435\u0440\u0432\u0430\u043b\u0435 \u043c\u043e\u0434\u0435\u043b\u044c \u043f\u0435\u0440\u0435\u0434\u0430\u0435\u0442 \u043e\u0431\u0449\u0438\u0439 \u0443\u0440\u043e\u0432\u0435\u043d\u044c \u0440\u044f\u0434\u0430, \u043d\u043e \u0445\u0443\u0436\u0435 \u0432\u043e\u0441\u043f\u0440\u043e\u0438\u0437\u0432\u043e\u0434\u0438\u0442 \u0434\u0430\u043b\u0435\u043a\u0438\u0435 \u0444\u0430\u0437\u044b \u043a\u0432\u0430\u0437\u0438\u043f\u0435\u0440\u0438\u043e\u0434\u0438\u0447\u0435\u0441\u043a\u043e\u0433\u043e \u0446\u0438\u043a\u043b\u0430.
7. \u041c\u0435\u0442\u0440\u0438\u043a\u0438 `MAE`, `MSE`, `MAPE`, `SMAPE` \u0432\u044b\u0447\u0438\u0441\u043b\u0435\u043d\u044b. \u0417\u043d\u0430\u0447\u0435\u043d\u0438\u0435 `MAPE` \u043d\u0443\u0436\u043d\u043e \u0438\u043d\u0442\u0435\u0440\u043f\u0440\u0435\u0442\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u043e\u0441\u0442\u043e\u0440\u043e\u0436\u043d\u043e, \u0442\u0430\u043a \u043a\u0430\u043a \u043e\u043a\u043e\u043b\u043e \u043c\u0438\u043d\u0438\u043c\u0443\u043c\u043e\u0432 \u0446\u0438\u043a\u043b\u0430 \u0438\u0441\u0442\u0438\u043d\u043d\u044b\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f \u0431\u043b\u0438\u0437\u043a\u0438 \u043a \u043d\u0443\u043b\u044e.
"""
    )
)

lecture_nb = json.loads(LECTURE_PATH.read_text(encoding="utf-8"))
nb = {
    "cells": cells,
    "metadata": lecture_nb.get("metadata", {}),
    "nbformat": 4,
    "nbformat_minor": 5,
}

nb = nbformat.from_dict(nb)
nb = nbformat.validator.normalize(nb)[1]
nbformat.write(nb, OUT_PATH.open("w", encoding="utf-8"))

print(OUT_PATH)
