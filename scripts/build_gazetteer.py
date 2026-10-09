"""Строит data/processed/gazetteer.parquet из справочника МО, полигонов и ИНИД (запускается один раз)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sbershock.news.gazetteer import build

ap = argparse.ArgumentParser()
ap.add_argument("--dict", default="data/raw/t_dict_municipal_districts.xlsx")
ap.add_argument("--poly", default="data/raw/t_dict_municipal_districts_poly.gpkg")
ap.add_argument("--settlements", default="data/raw/inid_settlements.csv")
ap.add_argument("--out", default="data/processed/gazetteer.parquet")
a = ap.parse_args()
gz = build(a.dict, a.poly, a.settlements)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
gz.to_parquet(a.out, index=False)
print(gz.level.value_counts().to_dict(), "алиасов:", gz.alias.nunique())
