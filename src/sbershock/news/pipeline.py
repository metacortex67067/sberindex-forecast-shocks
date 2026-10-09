"""
Новостной конвейер: экспорт Telegram (JSON) -> события с привязкой к territory_id.

Шаги (они же - «корректный способ согласования новостей с данными СберИндекса»):
  1. Разбор:   текст сообщения + ВРЕМЯ ПУБЛИКАЦИИ (date_unixtime, МСК). Время правок (edited) игнорируется,
               чтобы в признаки не попала информация из будущего.
  2. События:  прозрачные правила (events.py) -> локальные и национальные типы событий.
  3. Места:    NER (Natasha) находит топонимы и приводит их к именительному падежу.
  4. Привязка: топоним -> регион / МО / населённый пункт по газеттиру -> territory_id.
               Неоднозначные названия принимаются только при упоминании их региона в том же сообщении.
  5. Выход:    одна строка = (сообщение, тип события, территория, уровень привязки, уверенность).
Агрегация в признаки МО × месяц делается отдельно (features.py), с учётом точки прогноза.
"""
import json
from collections import defaultdict

import pandas as pd

from .events import EventClassifier
from .gazetteer import RISKY_NAMES, norm, strip_generic

BIG_CITY = 50_000  # население, при котором однозначное название города принимается без упоминания региона
SKIP_ENTITY_TYPES = {"link", "text_link", "mention", "hashtag", "email", "phone", "bot_command"}


def read_telegram(path: str, channel: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rows = []
    for m in d.get("messages", []):
        if m.get("type") != "message":
            continue
        ents = m.get("text_entities") or []
        text = "".join(e.get("text", "") for e in ents if e.get("type") not in SKIP_ENTITY_TYPES).strip()
        if not text:
            continue
        rows.append((channel, int(m["id"]), int(m["date_unixtime"]), text))
    df = pd.DataFrame(rows, columns=["channel", "msg_id", "ts", "text"])
    df["published_msk"] = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
    df["month"] = df.published_msk.dt.to_period("M").dt.to_timestamp()
    return df.drop(columns="ts")


class GeoResolver:
    def __init__(self, gazetteer: pd.DataFrame):
        self.by_alias = defaultdict(list)
        for r in gazetteer.itertuples(index=False):
            self.by_alias[r.alias].append(r)
        from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsNERTagger, Segmenter
        emb = NewsEmbedding()
        self._Doc, self.seg, self.mv = Doc, Segmenter(), MorphVocab()
        self.morph, self.ner = NewsMorphTagger(emb), NewsNERTagger(emb)

    def locations(self, text: str):
        doc = self._Doc(text)
        doc.segment(self.seg)
        doc.tag_morph(self.morph)  # без морфологии Natasha не приводит «Курской области» к «Курская область»
        doc.tag_ner(self.ner)
        out, fallback = [], []
        for sp in doc.spans:
            if sp.type != "LOC":
                continue
            sp.normalize(self.mv)
            out.append(norm(sp.normal or sp.text))
        # запасной путь: NER иногда относит город к организации («Администрация Орска») или пропускает его;
        # леммы слов с заглавной буквы проверяются по газеттиру по тем же строгим правилам
        toks = doc.tokens
        for i, t in enumerate(toks):
            if t.text[:1].isupper() and len(t.text) >= 3:
                t.lemmatize(self.mv)
                fallback.append(norm(t.lemma))
                if i + 1 < len(toks) and toks[i + 1].text[:1].isupper():
                    toks[i + 1].lemmatize(self.mv)
                    fallback.append(norm(f"{t.lemma} {toks[i + 1].lemma}"))
        ner = list(dict.fromkeys(out))
        return ner, [k for k in dict.fromkeys(fallback) if k not in ner]

    def resolve(self, text: str):
        """-> список dict(territory_id|None, region_code, level, confidence, matched)"""
        ner_locs, fb_locs = self.locations(text)
        cands, fb_keys = [], set()
        for loc in ner_locs:
            for key in {loc, strip_generic(loc)}:
                cands += [(key, r) for r in self.by_alias.get(key, [])]
        for key in fb_locs:
            rs = self.by_alias.get(key, [])
            if rs:
                fb_keys.add(key)
                cands += [(key, r) for r in rs]
        def dominant(key):
            """Крупный город, который явно доминирует среди одноимённых пунктов (Орск 228 тыс. против деревни Орск)."""
            if key in RISKY_NAMES:
                return None
            ss = sorted((x for k, x in cands if k == key and x.level == "settlement"),
                        key=lambda x: x.population or 0, reverse=True)
            if not ss or (ss[0].population or 0) < BIG_CITY:
                return None
            if len(ss) > 1 and (ss[0].population or 0) < 10 * max(ss[1].population or 0, 1):
                return None
            return ss[0]

        regions = {r.region_code for _, r in cands if r.level == "region"}
        keys = {k for k, r in cands if r.level != "region"}
        dom = {k: dominant(k) for k in keys}
        regions |= {d.region_code for d in dom.values() if d is not None}

        hits, by_key = [], defaultdict(list)
        for key, r in cands:
            if r.level != "region":
                by_key[key].append(r)
        for key, rs in by_key.items():
            in_reg = [r for r in rs if r.region_code in regions]
            if in_reg:
                pool, conf = in_reg, "high"
            elif dom[key] is not None:
                pool, conf = [x for x in rs if x.territory_id == dom[key].territory_id], "medium"
            elif key not in RISKY_NAMES and key not in fb_keys and \
                    len({x.territory_id for x in rs}) == 1 and any(x.level == "mo" for x in rs):
                pool, conf = rs, "medium"
            else:
                continue
            tids = {r.territory_id for r in pool}
            if len(tids) > 1:  # одно название в нескольких МО региона: самый крупный пункт
                pops = sorted(((r.population or 0), r.territory_id) for r in pool if r.level == "settlement")
                if pops and (len(pops) == 1 or pops[-1][0] >= 10 * max(pops[-2][0], 1)):
                    tids = {pops[-1][1]}
                else:
                    continue
            tid = next(iter(tids))
            r = next(x for x in pool if x.territory_id == tid)
            level = "mo" if any(x.level == "mo" and x.territory_id == tid for x in pool) else "settlement"
            hits.append(dict(territory_id=int(tid), region_code=r.region_code, level=level,
                             confidence=conf, matched=key))
        mo_regions = {h["region_code"] for h in hits}
        for rc in regions - mo_regions:
            hits.append(dict(territory_id=None, region_code=rc, level="region", confidence="high", matched=""))
        seen, uniq = set(), []
        for h in hits:
            k = (h["territory_id"], h["region_code"], h["level"])
            if k not in seen:
                seen.add(k)
                uniq.append(h)
        return uniq


def process(df: pd.DataFrame, resolver: GeoResolver, clf: EventClassifier = None, snippet_len: int = 280):
    clf = clf or EventClassifier()
    out = []
    for r in df.itertuples(index=False):
        loc_types, nat_types = clf.classify(r.text)
        if not loc_types and not nat_types:
            continue
        base = dict(channel=r.channel, msg_id=r.msg_id, published_msk=r.published_msk, month=r.month,
                    snippet=r.text[:snippet_len].replace("\n", " "))
        for t in nat_types:
            out.append({**base, "event_type": t, "scope": "national", "territory_id": None,
                        "region_code": None, "level": "national", "confidence": "high", "matched": ""})
        if loc_types:
            geo = resolver.resolve(r.text)
            for t in loc_types:
                for g in geo:
                    out.append({**base, "event_type": t, "scope": "local", **g})
                if not geo:
                    out.append({**base, "event_type": t, "scope": "local", "territory_id": None,
                                "region_code": None, "level": "unresolved", "confidence": "none", "matched": ""})
    res = pd.DataFrame(out)
    if len(res):
        res["territory_id"] = res.territory_id.astype("Int64")
        res["region_code"] = res.region_code.astype("Int64")
    return res
