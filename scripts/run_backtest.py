"""
Бэктест моделей -> results/bt_<model>.parquet
  python scripts/run_backtest.py --models naive snaive snaive_growth stf stf_lgbm stf_lgbm_noexog
"""
import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from sbershock.config import PROCESSED, RESULTS, load_yaml
from sbershock.data import loaders as L
from sbershock.evaluation.backtest import PanelMatrix, run_backtest
from sbershock.evaluation.metrics import add_lag12
from sbershock.models.stf import Naive, SeasonalNaive, SeasonalGrowth, STF
from sbershock.models.ml import STFLGBM

ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True); a = ap.parse_args()
cfg = load_yaml("models.yaml")
pm = PanelMatrix.from_panel(pd.read_parquet(PROCESSED / "panel.parquet"))
ctx = {"national": L.load_national_spending(), "weekly": L.load_weekly_categories()}
stf = STF(**cfg["stf"], name="stf")
def make(name):
    if name == "naive": return Naive()
    if name == "snaive": return SeasonalNaive()
    if name == "snaive_growth": return SeasonalGrowth(0.5, name="snaive_growth")
    if name == "stf": return stf
    static = pd.read_parquet(PROCESSED / "static_features.parquet").set_index("territory_id")
    exog = pd.read_parquet(PROCESSED / "exogenous.parquet")
    c = cfg["stf_lgbm"]
    return STFLGBM(stf, static, exog, first_origin_idx=c["first_origin_idx"], retrain_idx=c["retrain_idx"], train_frac=c["train_frac"], params=c["params"],
                   use_exog=(name == "stf_lgbm"), name=name)
RESULTS.mkdir(exist_ok=True)
for name in a.models:
    t0 = time.time(); m = make(name)
    res = add_lag12(run_backtest(m, pm, ctx, verbose=False), pm)
    res.to_parquet(RESULTS / f"bt_{name}.parquet", index=False)
    if hasattr(m, "_models"):
        import pickle
        imp = {k: dict(zip(v.feats_, v.feature_importances_)) for k, v in m._models.items() if v is not None}
        pd.DataFrame(imp).to_csv(RESULTS / f"importance_{name}.csv")
    print(f"[{name}] {time.time()-t0:.0f} c", flush=True)
