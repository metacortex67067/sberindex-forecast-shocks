"""Контроль воспроизводимости foundation-моделей: повтор точки T = 2023-12 при прогнозе 2025 г. (results/fccheck_*.parquet)
против исходного бэктеста (results/bt_*_extended.parquet) -> results/fm_repro_check.csv"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from sbershock.config import RESULTS
rows = []
for b in ("chronos2", "timesfm25"):
    c = pd.read_parquet(RESULTS / f"fccheck_{b}_extended.parquet")
    bt = pd.read_parquet(RESULTS / f"bt_{b}_extended.parquet")
    m = c.merge(bt[bt.origin == "2023-12-01"][["series", "h", "y_pred"]], on=["series", "h"], suffixes=("_new", "_bt"))
    rel = np.abs(m.y_pred_new / m.y_pred_bt - 1)
    rows.append({"модель": f"{b}_extended", "прогнозов сравнено": len(m), "макс. отн. расхождение": float(rel.max()),
                 "медиана отн. расхождения": float(rel.median())})
d = pd.DataFrame(rows); d.to_csv(RESULTS / "fm_repro_check.csv", index=False); print(d.to_string(index=False))
