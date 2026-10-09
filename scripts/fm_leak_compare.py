"""Foundation-модели (extended) и финальный ансамбль до и после исправления утечки в синтетической предыстории
(results/archive_leak_v1/ - версия с утечкой) -> results/fm_leak_compare.csv"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from sbershock.config import RESULTS
rows = []
for m in ("chronos2_extended", "timesfm25_extended", "final"):
    for ver, f in (("с утечкой (до 08.10)", RESULTS / "archive_leak_v1" / f"bt_{m}.parquet"), ("исправлено", RESULTS / f"bt_{m}.parquet")):
        if not f.exists():
            continue
        r = pd.read_parquet(f); r = r[r.complete & r.y_true.notna() & r.y_pred.notna()]
        mae = r.assign(ae=(r.y_true - r.y_pred).abs()).groupby("h").ae.mean()
        rows.append({"модель": m, "версия": ver, **{f"MAE h={h}": mae.get(h, np.nan) for h in (1, 3, 6, 12)}})
d = pd.DataFrame(rows)
if (RESULTS / "archive_leak_v1").exists():
    d.to_csv(RESULTS / "fm_leak_compare.csv", index=False)
else:  # архивной версии в репозитории нет, поэтому сохранённая таблица не перезаписывается
    print("нет results/archive_leak_v1 - оставляю сохранённый results/fm_leak_compare.csv")
print(d.to_string(index=False))
