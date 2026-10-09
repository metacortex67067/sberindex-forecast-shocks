"""
Метрики качества прогноза.

MAE (руб. на жителя в месяц) - основная метрика организаторов.
R² считаем двумя способами и объясняем почему:
  r2_level  - R² по уровням (как просят организаторы). Почти всегда ≈ 0,99, потому что МО сильно
              различаются по уровню расходов: даже «прогноз = прошлое значение» его получает.
  r2_yoy    - R² по годовому приросту (y_t / y_{t-12} - 1): показывает, объясняет ли модель именно
              ДИНАМИКУ, а не различия между МО. Это честная мера качества прогноза.
Плюс: MAPE, доля рядов, где модель лучше эталона, и бутстреп-интервал для разницы MAE
(кластерный по МО - ошибки одного МО в разных категориях и месяцах зависимы).
"""
import numpy as np
import pandas as pd


def _r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def add_lag12(res: pd.DataFrame, pm) -> pd.DataFrame:
    ti = res.target.map({m: i for i, m in enumerate(pm.months)}).values - 12
    lag = np.full(len(res), np.nan)
    ok = ti >= 0
    lag[ok] = pm.Y[res.series.values[ok], ti[ok]]
    return res.assign(y_lag12=lag)


def add_mase_scale(res: pd.DataFrame, pm) -> pd.DataFrame:
    """Масштаб MASE: средняя |y_t - y_{t-1}| ряда на обучающем отрезке (до точки прогноза) - Hyndman & Koehler (2006).
    MASE < 1 - модель лучше наивного прогноза «как в прошлом месяце» внутри выборки."""
    D = np.abs(np.diff(pm.Y, axis=1))
    oi = res.origin.map({m: i for i, m in enumerate(pm.months)}).values
    sc = np.full(len(res), np.nan)
    for o in np.unique(oi):
        m = oi == o
        sc[m] = np.nanmean(D[:, :o], axis=1)[res.series.values[m]]
    return res.assign(mase_scale=sc)


def holdout(res: pd.DataFrame) -> pd.DataFrame:
    """Классический режим организаторов: одна точка прогноза на горизонт - T = 2024-12 - h."""
    last = pd.Timestamp("2024-12-01")
    return res[res.origin == res.h.map(lambda h: last - pd.DateOffset(months=int(h)))]


def core(res: pd.DataFrame) -> pd.DataFrame:
    """Основной набор оценки: полные ряды (24 мес.), есть факт и прогноз."""
    return res[res.complete & res.y_true.notna() & res.y_pred.notna()]


def summary(res: pd.DataFrame, by=("model", "h")) -> pd.DataFrame:
    r = core(res).copy()
    r["ae"] = (r.y_true - r.y_pred).abs()
    r["ape"] = r.ae / r.y_true
    rows = []
    for k, g in r.groupby(list(by), observed=True):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(by, k))
        row.update(MAE=g.ae.mean(), MAPE=100 * g.ape.mean(), r2_level=_r2(g.y_true, g.y_pred), n=len(g))
        if "mase_scale" in g:
            row["MASE"] = (g.ae / g.mase_scale).replace([np.inf], np.nan).mean()
        if "y_lag12" in g and g.y_lag12.notna().any():
            m = g.y_lag12.notna()
            row["r2_yoy"] = _r2(g.y_true[m] / g.y_lag12[m] - 1, g.y_pred[m] / g.y_lag12[m] - 1)
        rows.append(row)
    return pd.DataFrame(rows)


def compare(res: pd.DataFrame, model: str, ref: str, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """ΔMAE (model - ref) по горизонтам с 95 % кластерным бутстреп-интервалом по МО и долей рядов-побед."""
    r = core(res)
    a = r[r.model == model].set_index(["series", "origin", "h"])
    b = r[r.model == ref].set_index(["series", "origin", "h"])
    j = a[["y_true", "y_pred", "territory_id"]].join(b[["y_pred"]], rsuffix="_ref", how="inner").reset_index()
    j["d"] = (j.y_true - j.y_pred).abs() - (j.y_true - j.y_pred_ref).abs()
    rng = np.random.default_rng(seed)
    rows = []
    for h, g in j.groupby("h"):
        per_mo = g.groupby("territory_id").d.agg(["sum", "count"])
        s, n = per_mo["sum"].values, per_mo["count"].values
        idx = rng.integers(0, len(s), size=(n_boot, len(s)))
        boots = s[idx].sum(1) / n[idx].sum(1)
        ser = g.groupby("series").d.mean()
        rows.append(dict(h=h, model=model, ref=ref, dMAE=g.d.mean(),
                         ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5),
                         win_share=(ser < 0).mean()))
    return pd.DataFrame(rows)
