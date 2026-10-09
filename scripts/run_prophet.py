"""
Бэктест Prophet. Полный прогон (все 13 тыс. рядов × 12 точек) около 5 ч на 1 ядре, лучше запускать с --n-jobs 4.
Здесь по умолчанию - случайная выборка МО (--sample-mo), все 6 категорий.
  python scripts/run_prophet.py --variant default --sample-mo 300
  python scripts/run_prophet.py --variant default --n-jobs 4          # полный прогон
"""
import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from sbershock.config import PROCESSED, RESULTS
from sbershock.evaluation.backtest import PanelMatrix, run_backtest
from sbershock.models.prophet_model import ProphetModel

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="default", choices=["default", "tuned"])
ap.add_argument("--sample-mo", type=int, default=0, help="0 = все МО")
ap.add_argument("--n-jobs", type=int, default=1)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

pm = PanelMatrix.from_panel(pd.read_parquet(PROCESSED / "panel.parquet"))
mask = None
if a.sample_mo:
    mos = pm.keys[pm.keys.complete].territory_id.drop_duplicates().sample(a.sample_mo, random_state=a.seed)
    mask = pm.keys.territory_id.isin(mos).values
t0 = time.time()
res = run_backtest(ProphetModel(a.variant, mask, a.n_jobs), pm, {}, verbose=True)
res = res[res.y_pred.notna()]
RESULTS.mkdir(exist_ok=True)
tag = f"_sample{a.sample_mo}" if a.sample_mo else ""
res.to_parquet(RESULTS / f"bt_prophet_{a.variant}{tag}.parquet", index=False)
print(f"\nготово за {time.time()-t0:.0f} c, строк {len(res)}")
