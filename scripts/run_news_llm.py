"""
Новости v2: LLM-разметка Telegram-экспортов -> data/processed/news_llm_events_<канал>.parquet
  python scripts/run_news_llm.py --input-dir data/raw/news_json                    # весь набор
  python scripts/run_news_llm.py --input-dir ... --sample-csv news_validation_sample.csv   # только сообщения проверочной выборки
"""
import argparse, glob, os, re, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from sbershock.config import PROCESSED, RAW, load_yaml
from sbershock.news.pipeline import read_telegram
from sbershock.news.llm_pipeline import LLM, CandidateIndex, run

ap = argparse.ArgumentParser()
ap.add_argument("--input-dir", required=True)
ap.add_argument("--sample-csv", default=None)
ap.add_argument("--limit", type=int, default=0, help="для пробного запуска: первые N сообщений на канал")
ap.add_argument("--mock", action="store_true", help="без GPU: заглушка LLM (проверка обвязки)")
ap.add_argument("--channels", nargs="*", help="обработать только эти каналы (например: tass)")
ap.add_argument("--shard", type=int, default=0, help="номер части (0..nshards-1)")
ap.add_argument("--nshards", type=int, default=1, help="на сколько частей делить сообщения (для параллельного запуска)")
ap.add_argument("--max-chars", type=int, default=1500)
ap.add_argument("--max-tokens", type=int, default=300)
ap.add_argument("--exact-shortcut", action="store_true", help="точное единственное совпадение места без этапа B (по умолчанию выкл. - как в v3)")
a = ap.parse_args()
cfg = load_yaml("news_llm.yaml")
gz = pd.read_parquet(PROCESSED / "gazetteer.parquet")
d = pd.read_excel(RAW / "t_dict_municipal_districts.xlsx")
idx = CandidateIndex(gz, d.groupby("territory_id").municipal_district_name.last(), d.groupby("territory_id").region_name.last())
if a.mock:
    from sbershock.news.llm_mock import MockLLM; llm = MockLLM()
else:
    llm = LLM(cfg["model"], cfg["tensor_parallel"], cfg["max_model_len"], quantization=cfg.get("quantization"))
rx = re.compile(cfg["prefilter"], re.I)
sample = None
if a.sample_csv:
    sample = pd.read_csv(a.sample_csv, sep=None, engine="python", encoding="utf-8-sig")[["channel", "msg_id"]]
files = sorted(glob.glob(os.path.join(a.input_dir, "**", "*.json"), recursive=True))
for path in files:
    ch = os.path.splitext(os.path.basename(path))[0].replace("_news", "").lower()
    if a.channels and ch not in a.channels:
        continue
    t0 = time.time()
    df = read_telegram(path, ch)
    df = df[(df.published_msk >= cfg["start"]) & (df.published_msk < pd.Timestamp(cfg["end"]) + pd.Timedelta(days=1))]
    if sample is not None:
        df = df[df.msg_id.isin(sample[sample.channel == ch].msg_id)]
    else:
        df = df[df.text.str.lower().str.replace("ё", "е").str.contains(rx)]
    if a.limit:
        df = df.head(a.limit)
    if a.nshards > 1:  # детерминированное деление по номеру сообщения
        df = df[df.msg_id % a.nshards == a.shard]
    print(f"[{ch}] часть {a.shard + 1}/{a.nshards}: сообщений на вход LLM {len(df)}", flush=True)
    ev = run(df, llm, idx, batch=cfg["batch"], max_chars=a.max_chars, max_tokens_a=a.max_tokens,
             exact_shortcut=a.exact_shortcut)
    tag = "_sample" if sample is not None else (f"_s{a.shard}of{a.nshards}" if a.nshards > 1 else "")
    out = PROCESSED / f"news_llm_events_{ch}{tag}.parquet"
    ev.to_parquet(out, index=False)
    print(f"[{ch}] на вход LLM: {len(df)}, событий: {len(ev)}, с МО: {ev.territory_id.notna().sum() if len(ev) else 0}, "
          f"время: {time.time()-t0:.0f} c -> {out}", flush=True)
