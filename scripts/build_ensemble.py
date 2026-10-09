"""Финальный ансамбль STF: равные веса (в логарифмах) 9 членов «затухание × окно роста» -> results/bt_ens_stf9.parquet"""
import glob, os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from sbershock.config import RESULTS
KEY = ["series", "origin", "h"]
from sbershock.config import load_yaml
cfg = load_yaml("models.yaml")["ensemble"]
mem = {os.path.basename(f)[3:-8]: pd.read_parquet(f) for f in sorted(glob.glob(str(RESULTS / cfg.get("members_dir", "members_v2") / "bt_*.parquet")))}
base = next(iter(mem.values()))[KEY + ["target", "y_true", "territory_id", "category", "complete", "y_lag12"]].copy()
idx = pd.MultiIndex.from_frame(base[KEY])
base["y_pred"] = np.exp(np.mean([np.log(m.set_index(KEY).loc[idx].y_pred.values) for m in mem.values()], 0))
base["model"] = "ens_stf9"
base.to_parquet(RESULTS / "bt_ens_stf9.parquet", index=False)
print("членов:", len(mem), "->", RESULTS / "bt_ens_stf9.parquet")
