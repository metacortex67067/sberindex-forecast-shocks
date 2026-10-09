"""
Точность новостной разметки v1 (правила + Natasha + словарь) и v2 (Qwen2.5-7B + выбор МО из кандидатов)
на одних и тех же 200 проверочных сообщениях (data/validation/news_validation_labeled.csv).

Эталон:
  * проверочная выборка - 200 событий v1 (случайно по 4 каналам); для каждого: место v1 верно? тип v1 верен?
    (16 строк проверены вручную, 184 размечены LLM-ассистентом Claude по тем же правилам; place_final / type_final);
  * ответ v2 по тому же сообщению (все строки v2 этого сообщения) оценивается так:
      - автоматически ВЕРНО, если v1 был верен и v2 указал то же место (тот же МО или тот же регион
        на уровне региона) / тот же тип;
      - остальное проверено по тексту сообщения (data/validation/news_v2_manual_labels.csv, Claude, с комментарием).
  * сообщение v2 верно по месту (типу), если верны ВСЕ строки v2 этого сообщения (строгое правило).
Правило «уровень региона верен, если регион верен» - то же, что при разметке v1.
«Не-событие» - сообщение, у которого тип v1 признан неверным (type_final = 0); «верное событие v1» - оба поля верны.
Выход: results/news_accuracy_v1_v2.csv, results/news_accuracy_rows.csv
"""
import glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from sbershock.config import PROCESSED, RESULTS, ROOT
from sbershock.data.loaders import territory_table
from sbershock.features.news_v2 import T2G

VAL = ROOT / "data" / "validation"


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - w, c + w


tt = territory_table()
reg = tt.drop_duplicates("region_code").set_index("region_code").region_name
v = pd.read_csv(VAL / "news_validation_labeled.csv", sep=";", encoding="utf-8-sig")
man = pd.read_csv(VAL / "news_v2_manual_labels.csv", sep=";", encoding="utf-8-sig")
ev = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PROCESSED / "llm_full" / "news_llm_events_*.parquet")))
                if "_sample" not in f], ignore_index=True)
ev = ev.drop_duplicates(["channel", "msg_id", "event_type", "territory_id", "region_code"]).reset_index(drop=True)
keys = set(zip(v.channel, v.msg_id))
e = ev[[k in keys for k in zip(ev.channel, ev.msg_id)]].copy().reset_index(drop=True)
e["mo_name"] = e.territory_id.map(tt.municipal_district_name)
e["reg_name"] = e.region_code.map(reg)
e["row"] = e.groupby(["channel", "msg_id"]).cumcount()
vv = v.set_index(["channel", "msg_id"])


def auto(r):
    t = vv.loc[(r.channel, r.msg_id)]
    p_ok = t_ok = np.nan
    if t.place_final == 1:
        if t.level == "region" and r.level == "region" and r.reg_name == t.region:
            p_ok = 1
        elif t.level == "mo" and r.level == "mo" and r.mo_name == t.place:
            p_ok = 1
    if t.type_final == 1 and r.event_type == t.event_type:
        t_ok = 1
    return p_ok, t_ok


a = np.array([auto(r) for r in e.itertuples()], dtype=float)
e["p_auto"], e["t_auto"] = a[:, 0], a[:, 1]
e = e.merge(man[["channel", "msg_id", "row", "place_ok", "type_ok"]], on=["channel", "msg_id", "row"], how="left")
e["p"] = e.p_auto.fillna(e.place_ok); e["t"] = e.t_auto.fillna(e.type_ok)
missing = e[e.p.isna() | e.t.isna()]
if len(missing):
    print("НЕТ МЕТКИ для строк v2:\n", missing[["channel", "msg_id", "row", "event_type", "level", "mo_name", "reg_name"]].to_string())
e["g_ok"] = e.t  # группа типа верна, если верен тип,
same_group = e.apply(lambda r: T2G.get(r.event_type) == T2G.get(vv.loc[(r.channel, r.msg_id)].event_type)
                     and vv.loc[(r.channel, r.msg_id)].type_final == 1, axis=1)
e.loc[e.t == 0, "g_ok"] = same_group[e.t == 0].astype(float)  # или тип другой, но из той же группы, что верный тип v1
strict_min = lambda x: x.min(skipna=False)  # строка без метки не считается верной
msg = e.groupby(["channel", "msg_id"]).agg(p2=("p", strict_min), t2=("t", strict_min), g2=("g_ok", strict_min), n_rows=("p", "size"))
v = v.join(msg, on=["channel", "msg_id"])
v["kept"] = v.p2.notna()
v["both2"] = (v.p2 == 1) & (v.t2 == 1)
v["both1"] = (v.place_final == 1) & (v.type_final == 1)

rows = []
def add(name, s, sub):
    s = s[sub].dropna(); k, n = int(s.sum()), len(s); lo, hi = wilson(k, n)
    rows.append({"метрика": name, "верно": k, "всего": n, "доля": round(k / n, 3) if n else np.nan,
                 "ДИ95_низ": round(lo, 3), "ДИ95_верх": round(hi, 3)})

allm = pd.Series(True, index=v.index)
add("v1: место (все 200)", v.place_final, allm)
add("v1: тип (все 200)", v.type_final, allm)
add("v1: оба (все 200)", v.both1.astype(float), allm & v.place_final.notna())
k = v.kept
add("v1: место (сообщения, оставленные v2)", v.place_final, k)
add("v1: тип (сообщения, оставленные v2)", v.type_final, k)
add("v1: оба (сообщения, оставленные v2)", v.both1.astype(float), k & v.place_final.notna())
add("v2: место", v.p2, k)
add("v2: тип (строго)", v.t2, k)
add("v2: группа типа (для признаков)", v.g2, k)
add("v2: оба (строго)", v.both2.astype(float), k)
non_event = v.type_final == 0
true_ev = v.both1
add("v2 отбросил «не-события» (тип v1 неверен)", (~v.kept).astype(float), non_event)
add("v2 потерял верные события v1", (~v.kept).astype(float), true_ev)
res = pd.DataFrame(rows)
RESULTS.mkdir(exist_ok=True)
res.to_csv(RESULTS / "news_accuracy_v1_v2.csv", index=False)
e.to_csv(RESULTS / "news_accuracy_rows.csv", index=False)
print(res.to_string(index=False))
print(f"\nсообщений: {len(v)}, v2 оставил: {int(v.kept.sum())}; строк v2: {len(e)}, "
      f"автометок места {int(e.p_auto.notna().sum())}, типа {int(e.t_auto.notna().sum())}, ручных строк {int(e.place_ok.notna().sum())}")
