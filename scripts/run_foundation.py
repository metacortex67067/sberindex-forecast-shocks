"""
Бэктест foundation-моделей (нужен GPU) -> results/bt_<backend>_<mode>.parquet
  python scripts/run_foundation.py --backends chronos2 timesfm25 tirex --modes raw stf_resid extended
Каждая комбинация в try/except: если какая-то модель не установилась, остальные всё равно посчитаются.
"""
import argparse, os, sys, time, traceback, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from sbershock.config import PROCESSED, RESULTS, load_yaml
from sbershock.data import loaders as L
from sbershock.evaluation.backtest import PanelMatrix, run_backtest
from sbershock.evaluation.metrics import add_lag12

ap = argparse.ArgumentParser()
ap.add_argument("--backends", nargs="+", default=["chronos2", "timesfm25", "tirex"])
ap.add_argument("--modes", nargs="+", default=["raw", "stf_resid", "extended"])
ap.add_argument("--device", default="auto")
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
ctx = {"national": L.load_national_spending(), "weekly": L.load_weekly_categories()}
stf_kw = {k: v for k, v in load_yaml("models.yaml")["stf"].items()}
RESULTS.mkdir(exist_ok=True)
for b in a.backends:
    for mode in a.modes:
        t0 = time.time()
        try:
            m = FoundationModel(b, mode, stf_kw, a.device)
            res = add_lag12(run_backtest(m, pm, ctx, verbose=False), pm)
            res.to_parquet(RESULTS / f"bt_{b}_{mode}.parquet", index=False)
            c = res[res.complete & res.y_true.notna()]
            print(f"[OK] {b}/{mode}: {time.time()-t0:.0f} c; MAE по h:",
                  c.assign(ae=(c.y_true - c.y_pred).abs()).groupby("h").ae.mean().round(0).to_dict(), flush=True)
        except Exception as e:
            print(f"[FAIL] {b}/{mode}: {e}", flush=True); traceback.print_exc()
