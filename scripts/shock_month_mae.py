"""MAE в месяцы шоков против обычных месяцев. Шок = тревога панельного детектора (порог: 2 % тревог на 2024 г.)
по остаткам прогноза STF на 1 мес. Это «необычные» месяцы ряда - то, где прогноз сложнее всего."""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from sbershock.config import PROCESSED, RESULTS
from sbershock.evaluation.backtest import PanelMatrix
from sbershock.changepoint import detectors as D
pm = PanelMatrix.from_panel(pd.read_parquet(PROCESSED / "panel.parquet"))
comp = np.where(pm.keys.complete.values)[0]; keys = pm.keys.iloc[comp].reset_index(drop=True)
E = np.load(RESULTS / "cp_residuals_clean.npy")
S = D.panel_score(E, keys.cat_idx.values, keys.region_code.values)
thr = np.nanquantile(S[:, 12:], 0.98)
A = np.nan_to_num(S) > thr
full2comp = {s: i for i, s in enumerate(comp)}
rows = []
for name, f in [("Prophet", "bt_prophet_default"), ("Финал", "bt_final"), ("STF-ансамбль", "bt_ens_stf9")]:
    r = pd.read_parquet(RESULTS / f"{f}.parquet")
    r = r[r.complete & r.y_true.notna() & r.y_pred.notna()].copy()
    ti = r.target.map({m: i for i, m in enumerate(pm.months)}).values
    ci = r.series.map(full2comp).values
    r["shock"] = A[ci, ti]
    r["ae"] = (r.y_true - r.y_pred).abs()
    r["ape"] = 100 * r.ae / r.y_true
    g = r.groupby(["h", "shock"]).agg(MAE=("ae", "mean"), MAPE=("ape", "mean"), n=("ae", "size")).reset_index()
    g["model"] = name; rows.append(g)
T = pd.concat(rows); T.to_csv(RESULTS / "shock_month_mae.csv", index=False)
print("доля ряд-месяцев с шоком (2024):", round(A[:, 12:].mean(), 4))
print(T.pivot_table(index=["model", "shock"], columns="h", values="MAPE").round(1).to_string())
print(T.pivot_table(index=["model", "shock"], columns="h", values="MAE").round(0).to_string())
