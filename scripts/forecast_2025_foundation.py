"""
Прогноз на 2025 год foundation-моделями (члены финального ансамбля), нужен GPU.
Одна точка прогноза T = 2024-12 (последний известный месяц), горизонты h = 1…12 (2025-01 … 2025-12).

  python scripts/forecast_2025_foundation.py --backends chronos2 timesfm25

Выход:
  results/fc2025_<backend>_extended.parquet  - прогноз на 2025 г. (13 122 ряда × 12 мес.)
  results/fccheck_<backend>_extended.parquet - контроль воспроизводимости: повтор бэктеста в точке T = 2023-12
                                               (должен совпасть с bt_<backend>_extended.parquet из бэктеста)
Устройство выбирается само: GPU, если есть, иначе CPU (на CPU дольше, но работает).

Строгость по времени: для прогноза 2025 г. национальный ряд обрезается по 2024-12 (модель не видит 2025 г.).
Контекст (национальный ряд + недельные категории) - тот же, что в бэктесте (scripts/run_foundation.py);
синтетическая предыстория режима extended строится только по данным до точки прогноза (исправлено 08.10.2026).
"""
import argparse, os, sys, time, traceback, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from sbershock.config import PROCESSED, RESULTS, load_yaml
from sbershock.data import loaders as L
from sbershock.evaluation.backtest import PanelMatrix

ap = argparse.ArgumentParser()
ap.add_argument("--backends", nargs="+", default=["chronos2", "timesfm25"])
ap.add_argument("--mode", default="extended")
ap.add_argument("--device", default="auto")
ap.add_argument("--no-check", action="store_true", help="не повторять бэктест в точке 2023-12")
a = ap.parse_args()
if a.device == "auto":
    try:
        import torch
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        a.device = "cpu"
print("устройство:", a.device, flush=True)

from sbershock.models.foundation import FoundationModel
pm = PanelMatrix.from_panel(pd.read_parquet(PROCESSED / "panel.parquet"))
nat = L.load_national_spending()
wk = L.load_weekly_categories()
stf_kw = {k: v for k, v in load_yaml("models.yaml")["stf"].items()}
RESULTS.mkdir(exist_ok=True)


def to_long(P, ti, h_max, model_name):
    T = pm.months[ti]
    rows = []
    for h in range(1, h_max + 1):
        tgt = T + pd.DateOffset(months=h)
        y_true = pm.Y[:, ti + h] if ti + h < pm.Y.shape[1] else np.full(len(pm.keys), np.nan)
        rows.append(pd.DataFrame({"series": np.arange(len(pm.keys)), "origin": T, "h": h, "target": tgt,
                                  "y_true": y_true, "y_pred": P[:, h - 1]}))
    res = pd.concat(rows, ignore_index=True).join(pm.keys[["territory_id", "category", "complete"]], on="series")
    res["model"] = model_name
    return res


for b in a.backends:
    t0 = time.time()
    try:
        m = FoundationModel(b, a.mode, stf_kw, a.device)
        # прогноз на 2025 г.: T = 2024-12, национальный ряд только до 2024-12
        ti = len(pm.months) - 1
        ctx = {"national": nat[nat.index <= pm.months[ti]], "weekly": wk[wk.index <= pm.months[ti] + pd.offsets.MonthEnd(0)]}
        P = m.predict(pm, ti, 12, ctx)
        out = to_long(P, ti, 12, m.name)
        out.to_parquet(RESULTS / f"fc2025_{b}_{a.mode}.parquet", index=False)
        c = out[out.complete]
        print(f"[OK] {b}/{a.mode} прогноз 2025: {time.time()-t0:.0f} c; рядов {len(pm.keys)}; "
              f"NaN {int(np.isnan(P).sum())}; средний прогноз (полные ряды) по h:",
              c.groupby("h").y_pred.mean().round(0).to_dict(), flush=True)
        # контроль: та же процедура в точке 2023-12, результат сверяется с bt_*.parquet
        if not a.no_check:
            t1 = time.time()
            ti0 = pm.t_index("2023-12-01")
            P0 = m.predict(pm, ti0, 12, {"national": nat, "weekly": wk})
            chk = to_long(P0, ti0, 12, m.name)
            chk.to_parquet(RESULTS / f"fccheck_{b}_{a.mode}.parquet", index=False)
            cc = chk[chk.complete]
            print(f"[OK] {b}/{a.mode} контроль T=2023-12: {time.time()-t1:.0f} c; MAE по h:",
                  cc.assign(ae=(cc.y_true - cc.y_pred).abs()).groupby("h").ae.mean().round(0).to_dict(), flush=True)
    except Exception as e:
        print(f"[FAIL] {b}/{a.mode}: {e}", flush=True); traceback.print_exc()
