"""
Итоговые таблицы сравнения моделей прогноза - три режима оценки на одних и тех же 12 096 полных рядах:
  rolling   - все точки прогноза T = 2023-12 … 2024-11 (основной, максимум наблюдений);
  holdout   - одна точка на горизонт, T = 2024-12 - h (классический режим «последние h месяцев»);
  h2_2024   - все прогнозы с целевым месяцем 2024-07 … 2024-12 (тестовое полугодие; как у части участников).
Метрики: MAE (основная), R² уровня, R² годового прироста, MAPE, MASE; final против Prophet - ΔMAE с 95 %
кластерным бутстреп-интервалом по МО и доля рядов, где final точнее.
Выход: results/leaderboard_<режим>.csv, results/final_vs_prophet_all_modes.csv
"""
import glob, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from sbershock.config import PROCESSED, RESULTS
from sbershock.evaluation.backtest import PanelMatrix
from sbershock.evaluation import metrics as M

pm = PanelMatrix.from_panel(pd.read_parquet(PROCESSED / "panel.parquet"))
parts = []
for f in sorted(glob.glob(str(RESULTS / "bt_*.parquet"))):
    r = pd.read_parquet(f)
    if "y_lag12" not in r:
        r = M.add_lag12(r, pm)
    parts.append(r)
res = pd.concat(parts, ignore_index=True)
res = M.add_mase_scale(res, pm)
res = M.core(res)
# сравнение строго на одних данных: строки (ряд, точка, h), где есть прогнозы всех моделей
n_models = res.model.nunique()
cnt = res.groupby(["series", "origin", "h"]).model.transform("nunique")
common = res[cnt == n_models]
print(f"моделей: {n_models}; строк в общем наборе: {len(common) // n_models} из {res.groupby('model').size().max()}")

MODES = {"rolling": lambda r: r, "holdout": M.holdout,
         "h2_2024": lambda r: r[(r.target >= "2024-07-01") & (r.target <= "2024-12-01")]}
lb, cmp, cmp_t = [], [], []
for mode, sel in MODES.items():
    r = sel(common)
    s = M.summary(r)
    s["mode"] = mode
    lb.append(s)
    c = M.compare(r, "final", "prophet_default", n_boot=1000)
    p = s[s.model == "prophet_default"].set_index("h").MAE
    f = s[s.model == "final"].set_index("h").MAE
    c["MAE_prophet"] = c.h.map(p); c["MAE_final"] = c.h.map(f)
    c["delta_pct"] = 100 * (c.MAE_final / c.MAE_prophet - 1)
    c["mode"] = mode
    c["n_forecasts"] = c.h.map(r[r.model == "final"].groupby("h").size())
    cmp.append(c)
    if "prophet_tuned" in set(r.model):
        ct = M.compare(r, "final", "prophet_tuned", n_boot=1000)
        pt = s[s.model == "prophet_tuned"].set_index("h").MAE
        ct["MAE_prophet_tuned"] = ct.h.map(pt); ct["MAE_final"] = ct.h.map(f)
        ct["delta_pct"] = 100 * (ct.MAE_final / ct.MAE_prophet_tuned - 1); ct["mode"] = mode
        cmp_t.append(ct)
lb = pd.concat(lb, ignore_index=True)
cmp = pd.concat(cmp, ignore_index=True)
for mode in MODES:
    x = lb[lb["mode"] == mode]
    w = x.pivot(index="model", columns="h", values="MAE")
    w.columns = [f"MAE h={h}" for h in w.columns]
    for met in ("r2_yoy", "r2_level", "MASE", "MAPE"):
        y = x.pivot(index="model", columns="h", values=met)
        y.columns = [f"{met} h={h}" for h in y.columns]
        w = w.join(y)
    w = w.sort_values("MAE h=1")
    w.to_csv(RESULTS / f"leaderboard_{mode}.csv")
cmp.to_csv(RESULTS / "final_vs_prophet_all_modes.csv", index=False)
if cmp_t:
    pd.concat(cmp_t, ignore_index=True).to_csv(RESULTS / "final_vs_prophet_tuned_all_modes.csv", index=False)
pd.set_option("display.width", 250)
for mode in MODES:
    print(f"\n=== {mode}: MAE (общий набор рядов)")
    print(pd.read_csv(RESULTS / f"leaderboard_{mode}.csv", index_col=0).filter(like="MAE h=").round(0).to_string())
print("\n=== final против Prophet")
print(cmp[["mode", "h", "n_forecasts", "MAE_prophet", "MAE_final", "delta_pct", "ci_low", "ci_high", "win_share"]].round(3).to_string(index=False))
