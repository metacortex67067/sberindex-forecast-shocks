"""
Обработка экспортов Telegram -> data/processed/news_events_<канал>.parquet

Примеры:
  python scripts/process_news.py --input-dir data/raw/news          # все *.json в папке, канал = имя файла
  python scripts/process_news.py --input tass.json ria.json --channel tass ria
Нужно: pip install natasha pandas pyarrow ; файл data/processed/gazetteer.parquet
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from sbershock.news.pipeline import GeoResolver, process, read_telegram

ap = argparse.ArgumentParser()
ap.add_argument("--input", nargs="+")
ap.add_argument("--channel", nargs="+")
ap.add_argument("--input-dir", help="папка с экспортами *.json (поиск рекурсивный)")
ap.add_argument("--gazetteer", default="data/processed/gazetteer.parquet")
ap.add_argument("--outdir", default="data/processed")
ap.add_argument("--start", default="2022-01-01")
ap.add_argument("--end", default="2024-12-31")
a = ap.parse_args()
if a.input_dir:
    import glob
    a.input = sorted(glob.glob(os.path.join(a.input_dir, "**", "*.json"), recursive=True))
    a.channel = [os.path.splitext(os.path.basename(p))[0].replace("_news", "").lower() for p in a.input]
    print("найдены файлы:", dict(zip(a.channel, a.input)))
assert a.input and len(a.input) == len(a.channel), "укажите --input-dir или пары --input/--channel"

resolver = GeoResolver(pd.read_parquet(a.gazetteer))
for path, ch in zip(a.input, a.channel):
    t0 = time.time()
    df = read_telegram(path, ch)
    df = df[(df.published_msk >= a.start) & (df.published_msk < pd.Timestamp(a.end) + pd.Timedelta(days=1))]
    ev = process(df, resolver)
    out = os.path.join(a.outdir, f"news_events_{ch}.parquet")
    ev.to_parquet(out, index=False)
    loc = ev[ev.scope == "local"]
    print(f"[{ch}] сообщений: {len(df)}, строк событий: {len(ev)}, "
          f"локальных с привязкой: {(loc.level.isin(['mo','settlement','region'])).sum()}, "
          f"без привязки: {(loc.level == 'unresolved').sum()}, время: {time.time()-t0:.0f} c -> {out}")
