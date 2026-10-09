"""
Бэктест классических моделей (statsforecast) -> results/bt_<ets|arima|theta>_<raw|stf_resid>.parquet
  python scripts/run_classic.py --kinds ets theta arima --modes raw stf_resid
Та же процедура, что для всех моделей (scripts/run_backtest.py): скользящая точка прогноза, все 13 122 ряда.
"""
import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from sbershock.config import PROCESSED, RESULTS, load_yaml
from sbershock.data import loaders as L
from sbershock.evaluation.backtest import PanelMatrix, run_backtest
from sbershock.evaluation.metrics import add_lag12
from sbershock.models.classic import ClassicModel

ap = argparse.ArgumentParser()
ap.add_argument("--kinds", nargs="+", default=["ets", "theta", "arima"])
ap.add_argument("--modes", nargs="+", default=["raw", "stf_resid"])
ap.add_argument("--n-jobs", type=int, default=-1)
ap.add_argument("--subsample", type=int, default=0, help="для проверки скорости: первые N рядов")
ap.add_argument("--weekly", action="store_true", help="передать недельные категории СберИндекса (как у STF); "
                "результат пишется в results/sensitivity/ (проверка чувствительности, в основное сравнение не входит)")
a = ap.parse_args()
pm = PanelMatrix.from_panel(pd.read_parquet(PROCESSED / "panel.parquet"))
if a.subsample:
    pm = PanelMatrix(Y=pm.Y[:a.subsample], keys=pm.keys.iloc[:a.subsample].reset_index(drop=True))
ctx = {"national": L.load_national_spending()}
OUT = RESULTS
if a.weekly:
    ctx["weekly"] = L.load_weekly_categories()
    OUT = RESULTS / "sensitivity"; OUT.mkdir(parents=True, exist_ok=True)
stf_kw = load_yaml("models.yaml")["stf"]
RESULTS.mkdir(exist_ok=True)
for k in a.kinds:
    for mode in a.modes:
        t0 = time.time()
        m = ClassicModel(k, mode, stf_kw, a.n_jobs)
        res = add_lag12(run_backtest(m, pm, ctx, verbose=False), pm)
        if not a.subsample:
            res.to_parquet(OUT / f"bt_{m.name}.parquet", index=False)
        c = res[res.complete & res.y_true.notna()]
        print(f"[{m.name}] {time.time()-t0:.0f} c; MAE:", c.assign(ae=(c.y_true - c.y_pred).abs()).groupby("h").ae.mean().round(0).to_dict(),
              "NaN:", int(res.y_pred.isna().sum()), "<=0:", int((res.y_pred <= 0).sum()), flush=True)
