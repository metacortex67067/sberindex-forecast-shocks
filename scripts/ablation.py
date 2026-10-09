"""
Абляция прогноза: «лестница» улучшений от наивного прогноза до финального ансамбля - каждый шаг пересчитывается
текущим кодом (scripts/ablation.py -> results/ablation.csv). Шаги:
  0. Prophet (по умолчанию)                      - results/bt_prophet_default.parquet (готовый результат)
  1. наивный: последнее значение
  2. STF: уровень + рост + сезонность            - рост по национальному ряду / панели, без затухания
  3. + ансамбль окон роста (3 STF: 1, 3, 6 мес.)
  4. + затухающий рост и региональный рост (9 STF: φ ∈ {0,8; 0,9; 0,95} × окно {1, 3, 6})
  5. + недельные категории СберИндекса (маркетплейсы, здоровье, транспорт)  = ens_stf9
  6. + Chronos-2 и TimesFM 2.5 (extended)        = final
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from sbershock.config import PROCESSED, RESULTS, load_yaml
from sbershock.data import loaders as L
from sbershock.evaluation.backtest import PanelMatrix, run_backtest
from sbershock.evaluation import metrics as M
from sbershock.models.stf import STF

pm = PanelMatrix.from_panel(pd.read_parquet(PROCESSED / "panel.parquet"))
nat_only = {"national": L.load_national_spending()}
with_weekly = {"national": nat_only["national"], "weekly": L.load_weekly_categories()}
cfg, stf_cfg = load_yaml("models.yaml")["ensemble"], load_yaml("models.yaml")["stf"]
base = {k: v for k, v in stf_cfg.items() if k in ("lam_region", "lam_mo", "k_level")}
KEY = ["series", "origin", "h"]


def bt(model, ctx):
    return M.add_lag12(run_backtest(model, pm, ctx, verbose=False), pm)


def ens(results, name):
    b = results[0][KEY + ["target", "y_true", "territory_id", "category", "complete", "y_lag12"]].copy()
    b["y_pred"] = np.exp(np.mean([np.log(r.y_pred.values) for r in results], axis=0))
    b["model"] = name
    return b


steps = {}
steps["0. Prophet (по умолчанию)"] = pd.read_parquet(RESULTS / "bt_prophet_default.parquet")
steps["1. Наивный: последнее значение"] = pd.read_parquet(RESULTS / "bt_naive.parquet")
steps["2. STF: уровень + рост + сезонность"] = bt(STF(**base, k_growth=3, name="s2"), nat_only)
steps["3. + ансамбль окон роста (3 STF)"] = ens([bt(STF(**base, k_growth=k, name=f"s3_{k}"), nat_only) for k in cfg["k_growth"]], "s3")
nine = [bt(STF(**base, k_growth=k, damp=d, anchor_years=cfg["anchor_years"], lam_growth_region=cfg["lam_growth_region"],
               name=f"s4_{d}_{k}"), nat_only) for d in cfg["damp"] for k in cfg["k_growth"]]
steps["4. + затухающий рост, рег. рост (9 STF)"] = ens(nine, "s4")
steps["5. + недельные категории СберИндекса (ens_stf9)"] = pd.read_parquet(RESULTS / "bt_ens_stf9.parquet")
steps["6. + Chronos-2 и TimesFM (final)"] = pd.read_parquet(RESULTS / "bt_final.parquet")
rows = []
for name, r in steps.items():
    if "y_lag12" not in r:
        r = M.add_lag12(r, pm)
    r = M.add_mase_scale(r, pm)
    for mode, rr in [("rolling", r), ("holdout", M.holdout(r)),
                     ("h2_2024", r[(r.target >= "2024-07-01") & (r.target <= "2024-12-01")])]:
        s = M.summary(rr); s["model"] = name; s["mode"] = mode; rows.append(s)
A = pd.concat(rows, ignore_index=True)
A.to_csv(RESULTS / "ablation.csv", index=False)
pd.set_option("display.width", 250)
for mode in ("rolling", "holdout", "h2_2024"):
    print(f"\n=== {mode}")
    print(A[A["mode"] == mode].pivot(index="model", columns="h", values="MAE").round(0).to_string())
