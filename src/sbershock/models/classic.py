"""
Классические статистические модели (statsforecast): AutoETS, AutoARIMA, AutoTheta - в том же интерфейсе,
что и foundation-модели (FoundationModel), и в тех же режимах подачи данных:
  raw       - исходный ряд уровней, сезонность 12 мес. (стандартное применение «из коробки», как Prophet);
  stf_resid - ряд в логарифмах, очищенный от сезонного профиля STF; модель без сезонности прогнозирует
              уровень/тренд, сезонность возвращается обратно.
При истории < 2 лет statsforecast сам отключает сезонные варианты, которые нельзя оценить.
"""
import numpy as np
import pandas as pd

from .foundation import FoundationModel


class _StatsForecastBackend:
    def __init__(self, kind: str, season: int, n_jobs: int = -1):
        from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
        self.model = {"ets": lambda: AutoETS(season_length=season),
                      "arima": lambda: AutoARIMA(season_length=season),
                      "theta": lambda: AutoTheta(season_length=season)}[kind]()
        self.n_jobs = n_jobs

    def forecast(self, ctxs, h):
        from statsforecast import StatsForecast
        ds = pd.date_range("2000-01-01", periods=max(len(c) for c in ctxs), freq="MS")
        df = pd.concat([pd.DataFrame({"unique_id": i, "ds": ds[-len(c):], "y": np.asarray(c, float)})
                        for i, c in enumerate(ctxs)], ignore_index=True)
        from statsforecast.models import Naive
        # если модель не оценивается (очень короткий или постоянный ряд неполной панели), наивный прогноз
        sf = StatsForecast(models=[self.model], freq="MS", n_jobs=self.n_jobs, fallback_model=Naive())
        f = sf.forecast(df=df, h=h)
        col = [c for c in f.columns if c not in ("unique_id", "ds")][0]
        f = f.sort_values(["unique_id", "ds"])
        P = f[col].values.reshape(len(ctxs), h)
        return P


class ClassicModel(FoundationModel):
    def __init__(self, kind: str, mode: str = "raw", stf_kwargs=None, n_jobs: int = -1):
        super().__init__(kind, mode, stf_kwargs, device="cpu")
        self.name = f"{kind}_{mode}"
        self._b = _StatsForecastBackend(kind, 12 if mode == "raw" else 1, n_jobs)
