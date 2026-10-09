"""
Prophet - базовая модель организаторов. Две версии:
  prophet_default - Prophet() «из коробки», отдельная модель на каждый ряд (как обычно делают бейзлайн).
                    Важно: при < 2 лет истории Prophet сам ОТКЛЮЧАЕТ годовую сезонность.
  prophet_tuned   - честно настроенный Prophet: годовая сезонность включена принудительно,
                    мультипликативный режим, осторожный тренд. Нужен, чтобы сравнение было справедливым.
Prophet медленный (~0,1 с на ряд), поэтому поддерживается выборка рядов и параллельный запуск (n_jobs).
"""
import logging

import numpy as np
import pandas as pd

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


def _fit_one(ds, y, h_max, variant):
    from prophet import Prophet
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.full(h_max, np.nan)
    df = pd.DataFrame({"ds": ds[ok], "y": y[ok]})
    if variant == "default":
        m = Prophet(uncertainty_samples=0)
    else:
        m = Prophet(yearly_seasonality=3, weekly_seasonality=False, daily_seasonality=False,
                    seasonality_mode="multiplicative", changepoint_prior_scale=0.01,
                    uncertainty_samples=0)
    m.fit(df)
    fut = pd.DataFrame({"ds": pd.date_range(ds[-1] + pd.offsets.MonthBegin(1), periods=h_max, freq="MS")})
    return m.predict(fut).yhat.values


class ProphetModel:
    def __init__(self, variant="default", series_mask=None, n_jobs=1):
        self.variant, self.mask, self.n_jobs = variant, series_mask, n_jobs
        self.name = f"prophet_{variant}"

    def predict(self, pm, ti, h_max, ctx):
        idx = np.where(self.mask)[0] if self.mask is not None else np.arange(len(pm.Y))
        ds = pm.months[: ti + 1]
        if self.n_jobs > 1:
            from joblib import Parallel, delayed
            preds = Parallel(n_jobs=self.n_jobs)(delayed(_fit_one)(ds, pm.Y[i, : ti + 1], h_max, self.variant)
                                                 for i in idx)
        else:
            preds = [_fit_one(ds, pm.Y[i, : ti + 1], h_max, self.variant) for i in idx]
        P = np.full((len(pm.Y), h_max), np.nan)
        P[idx] = np.vstack(preds)
        return P
