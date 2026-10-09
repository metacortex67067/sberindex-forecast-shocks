"""Бэктест членов ансамбля STF (затухание × окно роста) -> results/members/bt_<name>.parquet"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from sbershock.config import PROCESSED, RESULTS, load_yaml
from sbershock.data import loaders as L
from sbershock.evaluation.backtest import PanelMatrix, run_backtest
from sbershock.evaluation.metrics import add_lag12
from sbershock.models.stf import STF
cfg = load_yaml("models.yaml")["ensemble"]
pm = PanelMatrix.from_panel(pd.read_parquet(PROCESSED / "panel.parquet")); ctx = {"national": L.load_national_spending(), "weekly": L.load_weekly_categories()}
base = {k: v for k, v in load_yaml("models.yaml")["stf"].items() if k in ("lam_region", "lam_mo", "k_level")}
out = RESULTS / cfg.get("members_dir", "members_v2"); out.mkdir(parents=True, exist_ok=True)
for d in cfg["damp"]:
    for kg in cfg["k_growth"]:
        name = f"stf_d{d}_kg{kg}"
        if (out / f"bt_{name}.parquet").exists(): continue
        m = STF(**base, k_growth=kg, damp=d, anchor_years=cfg["anchor_years"], lam_growth_region=cfg["lam_growth_region"], damp_direct_only=cfg.get("damp_direct_only", False), name=name)
        add_lag12(run_backtest(m, pm, ctx, verbose=False), pm).to_parquet(out / f"bt_{name}.parquet", index=False)
        print(name, flush=True)
