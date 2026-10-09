"""
Сравнение детекторов шоков: полусинтетика + реальные события -> results/cp_*.csv
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from sbershock.config import PROCESSED, RESULTS, load_yaml
from sbershock.data import loaders as L
from sbershock.evaluation.backtest import PanelMatrix
from sbershock.models.stf import STF
from sbershock.changepoint import detectors as D
from sbershock.changepoint.evaluation import inject, inject_regional, threshold_for_far, evaluate

cfg = load_yaml("changepoint.yaml")
pm0 = PanelMatrix.from_panel(pd.read_parquet(PROCESSED / "panel.parquet"))
keep = pm0.keys.complete.values
pm0 = PanelMatrix(Y=pm0.Y[keep], keys=pm0.keys[keep].reset_index(drop=True))
ctx = {"national": L.load_national_spending()}
stf = STF(**load_yaml("models.yaml")["stf"])
MONTHS_EVAL = list(range(12, 24))  # тревоги оцениваются в 2024 г.

def residuals(pm):
    """Остатки прогноза STF на 1 мес.: e_t = log y_t - log ŷ_{t|t-1} (только прошлое)."""
    E = np.full(pm.Y.shape, np.nan)
    for t in range(5, pm.Y.shape[1] - 1):
        E[:, t + 1] = np.log(pm.Y[:, t + 1]) - np.log(stf.predict(pm, t, 1, ctx)[:, 0])
    return E

def scores(pm, E_raw, sub_rupt):
    t0 = time.time()
    S = D.all_scores(E_raw, pm.keys, cfg, slow_rows=sub_rupt)
    print(f"  детекторы: {time.time()-t0:.0f} c", flush=True)
    return S

# local: шок в одном ряду (МО × категория); regional: один шок во всех МО региона в одной категории
def run_scenario(name, rng):
    if name == "local":
        Y2, labels, shocked = inject(pm0.Y, rng, frac=0.5, t_min=12, t_max=22, size=tuple(cfg["shock_size"]))
    else:
        Y2, labels, shocked = inject_regional(pm0.Y, pm0.keys, rng, frac_regions=cfg.get("regional_frac", 0.3),
                                              t_min=12, t_max=22, size=tuple(cfg["shock_size"]))
    pm = PanelMatrix(Y=Y2, keys=pm0.keys)
    E = residuals(pm)
    n = len(Y2)
    calib = rng.random(n) < 0.5  # половина рядов для калибровки порогов, половина для теста
    sub = None if cfg["slow_frac"] >= 1 else np.where(rng.random(n) < cfg["slow_frac"])[0]
    S = scores(pm, E, sub)
    rows = []
    for det, sc in S.items():
        avail = np.isfinite(sc).any(1)
        thr = threshold_for_far(sc, calib & ~shocked & avail, MONTHS_EVAL, far=cfg["far"])
        test = ~calib & avail
        pos = {s: i for i, s in enumerate(np.where(test)[0])}
        lab = labels[labels.series.isin(np.where(test)[0])]
        r = evaluate(sc[test], lab.assign(series=lab.series.map(pos)), shocked[test], thr, MONTHS_EVAL)
        r["detector"] = det
        for typ in ["step", "dip", "ramp", "spike"]:
            lt = lab[lab.type == typ]
            r[f"recall_{typ}"] = evaluate(sc[test], lt.assign(series=lt.series.map(pos)), shocked[test], thr, MONTHS_EVAL)["recall"]
        rows.append(r)
    res = pd.DataFrame(rows).set_index("detector").sort_values("F1", ascending=False)
    res["scenario"] = name
    return res


RESULTS.mkdir(exist_ok=True)
pd.set_option("display.width", 220)
out = {}
for k, name in enumerate(["local", "regional"]):
    out[name] = run_scenario(name, np.random.default_rng(cfg["seed"] + k))
    print(f"\n=== сценарий {name}"); print(out[name].round(3).to_string())
out["local"].to_csv(RESULTS / "cp_semisynthetic.csv")
out["regional"].to_csv(RESULTS / "cp_semisynthetic_regional.csv")
np.save(RESULTS / "cp_residuals_clean.npy", residuals(pm0))
print("остатки на чистых рядах сохранены", flush=True)
