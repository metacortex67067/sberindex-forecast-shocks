"""Финальный ансамбль: 9 вариантов STF + Chronos-2 и TimesFM 2.5 в режиме «удлинённый контекст», равные веса
(среднее логарифмов) -> results/bt_final.parquet"""
import glob, os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from sbershock.config import RESULTS, load_yaml
KEY = ["series", "origin", "h"]
cfg = load_yaml("models.yaml")["ensemble"]
files = sorted(glob.glob(str(RESULTS / cfg["members_dir"] / "bt_*.parquet"))) + \
        [str(RESULTS / f"bt_{m}.parquet") for m in cfg["foundation_members"]]
base = pd.read_parquet(files[0])[KEY + ["target", "y_true", "territory_id", "category", "complete", "y_lag12"]]
idx = pd.MultiIndex.from_frame(base[KEY])
base["y_pred"] = np.exp(np.mean([np.log(pd.read_parquet(f).set_index(KEY).loc[idx].y_pred.values) for f in files], 0))
base["model"] = "final"
base.to_parquet(RESULTS / "bt_final.parquet", index=False)
print("членов:", len(files))
