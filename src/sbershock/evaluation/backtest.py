"""
Бэктест со скользящей точкой прогноза (rolling origin) - единая процедура для ВСЕХ моделей.

Точка прогноза T - последний месяц, данные которого известны модели. Модель видит только Y[:, :T]
и внешние ряды до T включительно; прогнозирует T+1 … T+12. Оценка для горизонта h - по всем T,
для которых T+h ≤ 2024-12 и история ≥ 12 мес.:
    h=1: 12 точек (2023-12 … 2024-11)   h=3: 10 точек   h=6: 7 точек   h=12: 1 точка (2023-12)
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import CATEGORIES, HORIZONS, PANEL_END, PANEL_START, eval_origins

MONTHS = pd.date_range(PANEL_START, PANEL_END, freq="MS")


@dataclass
class PanelMatrix:
    Y: np.ndarray
    keys: pd.DataFrame
    months: pd.DatetimeIndex = field(default_factory=lambda: MONTHS)

    @classmethod
    def from_panel(cls, panel: pd.DataFrame):
        p = panel.copy()
        p["category"] = p.category.astype(str)
        w = p.pivot_table(index=["territory_id", "category"], columns="month", values="y", observed=True)
        w = w.reindex(columns=MONTHS)
        meta = p.drop_duplicates(["territory_id", "category"]).set_index(["territory_id", "category"])
        keys = w.index.to_frame(index=False)
        keys["region_code"] = meta.loc[w.index, "region_code"].values
        keys["complete"] = meta.loc[w.index, "complete"].values
        keys["cat_idx"] = keys.category.map({c: i for i, c in enumerate(CATEGORIES)})
        return cls(Y=w.values.astype(float), keys=keys)

    def t_index(self, month) -> int:
        return int(self.months.get_loc(pd.Timestamp(month)))


def run_backtest(model, pm: PanelMatrix, ctx: dict, horizons=HORIZONS, verbose=True) -> pd.DataFrame:
    """model.predict(pm, T_idx, h_max, ctx) -> массив n_series × h_max (прогноз уровней)."""
    all_T = sorted({T for h in horizons for T in eval_origins(h)})
    out = []
    for T in all_T:
        ti = pm.t_index(T)
        hs = [h for h in horizons if T in eval_origins(h)]
        h_max = max(hs)
        P = model.predict(pm, ti, h_max, ctx)
        for h in hs:
            y_true = pm.Y[:, ti + h]
            out.append(pd.DataFrame({
                "series": np.arange(len(pm.keys)), "origin": T, "h": h,
                "target": pm.months[ti + h], "y_true": y_true, "y_pred": P[:, h - 1]}))
        if verbose:
            print(f"  {model.name}: T={T.date()} готово", end="\r")
    res = pd.concat(out, ignore_index=True)
    res = res.join(pm.keys[["territory_id", "category", "complete"]], on="series")
    res["model"] = model.name
    return res
