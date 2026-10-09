"""Новости v2 -> data/processed/exogenous_v2.parquet (внешние признаки v1 + новостные признаки v2) и incidents_v2.parquet"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from sbershock.config import PROCESSED
from sbershock.features.news_v2 import build_features
p = pd.read_parquet(PROCESSED / "panel.parquet"); terr = p[["territory_id", "region_code"]].drop_duplicates()
f2, inc, ev = build_features(PROCESSED / "llm_full", terr)
ex = pd.read_parquet(PROCESSED / "exogenous.parquet")
keep = [c for c in ex.columns if not c.startswith("news_")]
ex2 = ex[keep].merge(f2, on=["territory_id", "month"], how="left")
ex2.to_parquet(PROCESSED / "exogenous_v2.parquet", index=False); inc.to_parquet(PROCESSED / "incidents_v2.parquet", index=False)
print(f"сообщений-событий: {len(ev)}, происшествий: {len(inc)} (сжатие в {len(ev)/len(inc):.1f} раза), каналы: {sorted(ev.channel.unique())}")
print("происшествий по тяжести:", inc.severity.value_counts().sort_index().to_dict())
