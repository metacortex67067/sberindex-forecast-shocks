"""
Случайная выборка привязанных событий для ручной проверки (оценка точности геопривязки и классификации).
Пользователь заполняет колонки place_ok и type_ok (1 - верно, 0 - неверно), затем
scripts/score_news_validation.py считает точность с доверительным интервалом.
"""
import argparse, glob
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--events", default="data/processed/news_events_*.parquet")
ap.add_argument("--dict", default="data/raw/t_dict_municipal_districts.xlsx")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--out", default="data/processed/news_validation_sample.csv")
a = ap.parse_args()

ev = pd.concat([pd.read_parquet(f) for f in glob.glob(a.events)], ignore_index=True)
ev = ev[(ev.scope == "local") & (ev.level != "unresolved")]
ev = ev.drop_duplicates(["channel", "msg_id"]).sample(a.n, random_state=42)
d = pd.read_excel(a.dict)
mo = d.groupby("territory_id").municipal_district_name.last()
rg = d.groupby("region_code").region_name.first()
ev["place"] = [mo.get(t, "") if pd.notna(t) else "весь регион" for t in ev.territory_id]
ev["region"] = ev.region_code.map(rg)
ev["place_ok"], ev["type_ok"], ev["comment"] = "", "", ""
cols = ["channel", "msg_id", "published_msk", "snippet", "event_type", "place", "region", "level",
        "confidence", "place_ok", "type_ok", "comment"]
ev[cols].to_csv(a.out, index=False, encoding="utf-8-sig")
print("сохранено:", a.out, len(ev))
