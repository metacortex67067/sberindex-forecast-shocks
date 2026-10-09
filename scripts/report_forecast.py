"""
Сводка качества прогноза по всем results/bt_*.parquet -> results/forecast_*.csv
Два режима: rolling (все точки прогноза) и holdout (одна точка T = 2024-12 - h, как у организаторов).
Для сравнения с Prophet, посчитанным на выборке МО, сравнение делается на ТОЙ ЖЕ выборке рядов.
"""
import glob, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from sbershock.config import PROCESSED, RESULTS
from sbershock.evaluation.backtest import PanelMatrix
from sbershock.evaluation import metrics as M

pm = PanelMatrix.from_panel(pd.read_parquet(PROCESSED / "panel.parquet"))
res = pd.concat([(lambda r: r if "y_lag12" in r else M.add_lag12(r, pm))(pd.read_parquet(f))
                 for f in sorted(glob.glob(str(RESULTS / "bt_*.parquet")))], ignore_index=True)
res = M.add_mase_scale(res, pm)
out = []
for mode, r in [("rolling", res), ("holdout", M.holdout(res))]:
    s = M.summary(r); s["mode"] = mode; s["set"] = "all"; out.append(s)
    for ref in [m for m in res.model.unique() if m.startswith("prophet")]:
        ser = set(r[r.model == ref].series)
        s2 = M.summary(r[r.series.isin(ser)]); s2["mode"] = mode; s2["set"] = f"sample_{ref}"; out.append(s2)
S = pd.concat(out, ignore_index=True)
S.to_csv(RESULTS / "forecast_summary.csv", index=False)
cat = M.summary(res, by=("model", "h", "category")); cat.to_csv(RESULTS / "forecast_by_category.csv", index=False)
pd.set_option("display.width", 200)
for mode in ["rolling", "holdout"]:
    x = S[(S["mode"] == mode) & (S["set"] == "all")]
    print(f"\n=== {mode}: MAE"); print(x.pivot(index="model", columns="h", values="MAE").round(0).to_string())
    print(f"=== {mode}: MASE"); print(x.pivot(index="model", columns="h", values="MASE").round(3).to_string())
    print(f"=== {mode}: R2 yoy"); print(x.pivot(index="model", columns="h", values="r2_yoy").round(3).to_string())
    print(f"=== {mode}: R2 level"); print(x.pivot(index="model", columns="h", values="r2_level").round(4).to_string())
for ref in [m for m in res.model.unique() if m.startswith("prophet")]:
    x = S[(S["mode"] == "rolling") & (S["set"] == f"sample_{ref}")]
    print(f"\n=== rolling, выборка {ref}: MAE"); print(x.pivot(index="model", columns="h", values="MAE").round(0).to_string())
