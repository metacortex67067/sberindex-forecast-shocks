"""
Сборка данных: панель расходов, статические признаки МО, внешние признаки МО×месяц (v1).
  python scripts/build_data.py
Требует data/raw (см. README → «Данные»). Новостные признаки v1 - из data/processed/news_events_*.parquet
(scripts/process_news.py), v2 - отдельно scripts/build_news_v2.py.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sbershock.config import PROCESSED, RAW
from sbershock.data.panel import build_panel, build_static_features
from sbershock.features.exogenous import build_exogenous

if not (RAW / "rosstat" / "region_income_spending.csv").exists():
    import runpy
    runpy.run_path(os.path.join(os.path.dirname(__file__), "convert_income_bulletins.py"))
p = build_panel()
print("панель:", p.groupby(["territory_id", "category"], observed=True).ngroups, "рядов;",
      p[p.complete].groupby(["territory_id", "category"], observed=True).ngroups, "полных")
s = build_static_features(p)
print("статические признаки:", s.shape)
terr = p[["territory_id", "region_code"]].drop_duplicates()
ex = build_exogenous(terr)
ex.to_parquet(PROCESSED / "exogenous.parquet", index=False)
print("внешние признаки:", ex.shape)
