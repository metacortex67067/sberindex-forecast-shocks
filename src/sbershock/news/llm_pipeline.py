"""
Новости v2: LLM-разметка (замена правил + Natasha из v1).

Этап A - извлечение (LLM читает сообщение и отвечает JSON):
    реальное ли происшествие, тип (14 типов или "other"), тяжесть 0-3, в России ли,
    места текстом: населённый пункт / район / город / регион.
Этап B - выбор МО (LLM выбирает сама, но ТОЛЬКО из списка реальных МО):
    по тексту места подбираются кандидаты (нечёткое сравнение с названиями МО и населённых пунктов
    внутри названного региона), LLM выбирает номер кандидата, «только регион» или «не определить».
    Так LLM принимает решение по смыслу, а итоговый territory_id всегда существует в данных СберИндекса.
Время события = время публикации (без заглядывания вперёд), как в v1.
Запуск: GPU (NVIDIA T4), vLLM. Модель задаётся в configs/news_llm.yaml.
"""
import json
import re

import pandas as pd

EVENT_TYPES = ["flood", "fire", "wildfire", "emergency_regime", "drone_attack", "shelling", "evacuation",
               "utility_outage", "industrial_accident", "terror", "transport_disruption", "weather_extreme",
               "epidemic", "enterprise_shock", "other"]

SYSTEM_A = (
    "Ты аналитик новостей. По тексту сообщения определи, описывает ли оно РЕАЛЬНОЕ происшествие, "
    "которое случилось или происходит сейчас в конкретном месте. НЕ происшествие: совещания, заявления, планы, "
    "суды и приговоры, предотвращённые теракты, ложные угрозы и проверки минирования, учения, статистика, поздравления, "
    "истории о сотрудниках, спасение отдельного человека или животного (застрял, заблудился), советы и профилактика, "
    "дайджесты из нескольких несвязанных новостей. "
    "Тип emergency_regime — ТОЛЬКО если официально введён режим ЧС; «режим воздушной/ракетной опасности», угроза атаки "
    "беспилотников — это drone_attack или shelling. Если главное в сообщении — эвакуация, тип evacuation. "
    "Имена людей (Владимир Путин, Артём) — не места. Города в названиях рейсов и трасс — не место события. "
    "Ответь строго JSON по схеме.")

USER_A = """Сообщение (опубликовано {date}):
\"\"\"{text}\"\"\"

Типы: flood (паводок/наводнение), fire (пожар), wildfire (лесной/природный пожар), emergency_regime (введён режим ЧС),
drone_attack (атака/угроза БПЛА), shelling (обстрел/ракетная опасность), evacuation (эвакуация), utility_outage
(отключение света/тепла/воды/газа), industrial_accident (взрыв/авария/обрушение), terror (теракт), transport_disruption
(перекрытие дорог/ограничения аэропорта), weather_extreme (аномальная погода), epidemic (вспышка инфекции/отравление),
enterprise_shock (остановка/закрытие предприятия, массовые увольнения), other.
Тяжесть: 0 — нет последствий, 1 — мелкое, 2 — пострадавшие/заметный ущерб/массовые неудобства, 3 — погибшие/эвакуация/режим ЧС.
Мест может быть несколько (до 3) — только места, где ПРОИЗОШЛО событие."""

SCHEMA_A = {  # упрощённая схема: только конструкции, которые поддерживают все движки vLLM
    "type": "object",
    "properties": {
        "is_incident": {"type": "boolean"},
        "event_type": {"type": "string", "enum": EVENT_TYPES},
        "severity": {"type": "integer"},
        "in_russia": {"type": "boolean"},
        "places": {"type": "array", "items": {
            "type": "object",
            "properties": {"settlement": {"type": "string"}, "district": {"type": "string"},
                           "city": {"type": "string"}, "region": {"type": "string"}},
            "required": ["settlement", "district", "city", "region"]}},
    },
    "required": ["is_incident", "event_type", "severity", "in_russia", "places"],
}
JSON_HINT_A = ('\nФормат ответа — только JSON: {"is_incident": true/false, "event_type": "...", "severity": 0-3, '
               '"in_russia": true/false, "places": [{"settlement": "", "district": "", "city": "", "region": ""}]}. '
               'Пустая строка "" — если поле неизвестно.')

SYSTEM_B = ("Ты помогаешь привязать место происшествия к муниципалитету. Выбери номер муниципалитета из списка, "
            "в котором произошло событие. Если по тексту понятен только регион — ответь 0. Если ни один не подходит — -1. "
            "Ответь строго JSON.")
USER_B = """Сообщение: \"\"\"{text}\"\"\"
Место по мнению аналитика: {place}
Кандидаты:
{cands}
Ответ — только JSON: {{"choice": номер}}"""
SCHEMA_B = {"type": "object", "properties": {"choice": {"type": "integer"}}, "required": ["choice"]}


def _n(s):
    return re.sub(r"\s+", " ", str(s).lower().replace("ё", "е")).strip(" .,\"«»") if s else ""


class CandidateIndex:
    """Индекс для подбора кандидатов: названия МО (все варианты) и населённые пункты (с привязкой к МО)."""

    def __init__(self, gazetteer: pd.DataFrame, mo_names: pd.Series, region_names: pd.Series):
        g = gazetteer
        self.reg_alias = g[g.level == "region"].drop_duplicates("alias").set_index("alias").region_code.to_dict()
        self.mo = g[g.level.isin(["mo", "settlement"])].copy()
        self.mo["alias_n"] = self.mo.alias
        self.mo_names, self.region_names = mo_names, region_names
        self._t2r = self.mo.drop_duplicates("territory_id").set_index("territory_id").region_code.to_dict()

    def region_of(self, tid):
        return self._t2r.get(tid)

    def region_code(self, text):
        t = _n(text)
        if not t:
            return None
        if t in self.reg_alias:
            return self.reg_alias[t]
        from rapidfuzz import process, fuzz
        m = process.extractOne(t, list(self.reg_alias), scorer=fuzz.WRatio, score_cutoff=88)
        return self.reg_alias[m[0]] if m else None

    FEDERAL = {"москва": 77, "санкт-петербург": 78, "петербург": 78, "спб": 78, "севастополь": 92}

    def resolve_region_only(self, place: dict):
        """Случаи, когда МО выбирать не нужно -> код региона (или None, если выбор МО нужен).
        1) город федерального значения без района; 2) указан только регион."""
        city = _n(place.get("city")) or _n(place.get("settlement"))
        district = _n(place.get("district"))
        if city in self.FEDERAL and not district:
            return self.FEDERAL[city]
        if not city and not district:
            return self.region_code(place.get("region"))
        return None

    def exact_unique(self, place: dict, rc):
        """Если название пункта/города/района точно совпадает ровно с одним МО в названном регионе - вернуть его."""
        if rc is None:
            return None
        pool = self.mo[self.mo.region_code == rc]
        for f in ("settlement", "city", "district"):
            nm = _n(place.get(f))
            if nm:
                tids = pool[pool.alias_n == nm].territory_id.unique()
                if len(tids) == 1:
                    return int(tids[0])
        return None

    def candidates(self, place: dict, k=8):
        from rapidfuzz import process, fuzz
        rc = self.region_code(place.get("region"))
        if rc is None:  # регион не назван: определяется по точному совпадению с крупным городом или названием региона
            city = _n(place.get("city")) or _n(place.get("settlement"))
            if city in self.FEDERAL:
                rc = self.FEDERAL[city]
            elif city:
                big = self.mo[(self.mo.alias_n == city) & (self.mo.population.fillna(0) >= 50_000)]
                if big.region_code.nunique() == 1:
                    rc = int(big.region_code.iloc[0])
        pool = self.mo[self.mo.region_code == rc] if rc is not None else self.mo[self.mo.population.fillna(0).ge(50_000) | (self.mo.level == "mo")]
        names = [_n(place.get(f)) for f in ("settlement", "city", "district") if place.get(f)]
        scores = {}
        aliases = pool.alias_n.tolist()
        for nm in names:
            for alias, sc, idx in process.extract(nm, aliases, scorer=fuzz.WRatio, limit=40, score_cutoff=80):
                row = pool.iloc[idx]
                tid = int(row.territory_id)
                bonus = (row.population or 0) / 1e7 if pd.notna(row.population) else 0
                scores[tid] = max(scores.get(tid, 0), sc + bonus)
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        return rc, top


class LLM:
    """Обёртка над vLLM с поддержкой JSON-схем (разные версии vLLM называют это по-разному)."""

    def __init__(self, model, tp=1, max_model_len=4096, dtype="half", quantization=None):
        from vllm import LLM as V
        kw = dict(model=model, tensor_parallel_size=tp, max_model_len=max_model_len, dtype=dtype,
                  gpu_memory_utilization=0.90, enforce_eager=False)
        if quantization:
            kw["quantization"] = quantization
        self.v = V(**kw)

    def _params(self, schema, max_tokens):
        from vllm import SamplingParams
        try:
            from vllm.sampling_params import GuidedDecodingParams
            return SamplingParams(temperature=0.0, max_tokens=max_tokens, guided_decoding=GuidedDecodingParams(json=schema))
        except Exception:
            try:
                from vllm.sampling_params import StructuredOutputsParams
                return SamplingParams(temperature=0.0, max_tokens=max_tokens,
                                      structured_outputs=StructuredOutputsParams(json=schema))
            except Exception:
                return SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def json_batch(self, system, users, schema, max_tokens=300):
        msgs = [[{"role": "system", "content": system}, {"role": "user", "content": u}] for u in users]
        outs = None
        if not getattr(self, "_plain", False):
            try:
                outs = self.v.chat(msgs, self._params(schema, max_tokens), use_tqdm=True)
            except Exception as e:  # движок не принял схему, обычная генерация
                print(f"[warn] структурированный вывод недоступен ({type(e).__name__}: {str(e)[:200]}); "
                      f"перехожу на разбор JSON из текста", flush=True)
                self._plain = True
        if outs is None:
            from vllm import SamplingParams
            outs = self.v.chat(msgs, SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=True)
        res = []
        for o in outs:
            txt = o.outputs[0].text
            try:
                res.append(json.loads(txt[txt.find("{"): txt.rfind("}") + 1]))
            except Exception:
                res.append(None)
        return res


def run(df: pd.DataFrame, llm, index: CandidateIndex, batch=2048, max_chars=1500, max_tokens_a=300, log_every=True,
        exact_shortcut=False):
    """df: channel, msg_id, published_msk, month, text -> события в формате v1 (+ severity, decided_by)."""
    out = []
    for s in range(0, len(df), batch):
        part = df.iloc[s:s + batch]
        A = llm.json_batch(SYSTEM_A, [USER_A.format(date=str(r.published_msk)[:10], text=r.text[:max_chars]) + JSON_HINT_A
                                      for r in part.itertuples()], SCHEMA_A, max_tokens=max_tokens_a)
        jobs, direct = [], []
        for r, a in zip(part.itertuples(), A):
            if not a or not a.get("is_incident") or a.get("event_type") == "other":
                continue
            if not a.get("in_russia", True):
                continue
            base = dict(channel=r.channel, msg_id=r.msg_id, published_msk=r.published_msk, month=r.month,
                        snippet=r.text[:280].replace("\n", " "), event_type=a["event_type"],
                        severity=a.get("severity", 0), scope="local")
            for pl in (a.get("places") or [])[:3]:
                pl = {k: (v or None) for k, v in pl.items()} if isinstance(pl, dict) else {}
                ro = index.resolve_region_only(pl)
                if ro is not None:
                    direct.append({**base, "territory_id": None, "region_code": ro, "level": "region",
                                   "confidence": "llm", "matched": json.dumps(pl, ensure_ascii=False)})
                    continue
                rc, cand = index.candidates(pl)
                exact = index.exact_unique(pl, rc) if exact_shortcut else None
                if exact is not None:  # однозначное точное совпадение, LLM не нужна
                    direct.append({**base, "territory_id": exact, "region_code": index.region_of(exact), "level": "mo",
                                   "confidence": "llm_exact", "matched": json.dumps(pl, ensure_ascii=False)})
                    continue
                if not cand and rc is not None:
                    direct.append({**base, "territory_id": None, "region_code": rc, "level": "region",
                                   "confidence": "llm", "matched": json.dumps(pl, ensure_ascii=False)})
                    continue
                jobs.append((base, pl, rc, cand))
        users = []
        for base, pl, rc, cand in jobs:
            lines = [f"{i + 1}. {index.mo_names.get(t, t)} ({index.region_names.get(t, '')})" for i, t in enumerate(cand)]
            users.append(USER_B.format(text=base["snippet"], place=json.dumps(pl, ensure_ascii=False),
                                       cands="\n".join(lines) or "(нет)"))
        B = llm.json_batch(SYSTEM_B, users, SCHEMA_B, max_tokens=20) if users else []
        out.extend(direct)
        if log_every:
            import time as _t
            print(f"[progress] {min(s + batch, len(df))}/{len(df)} сообщений, {_t.strftime('%H:%M:%S')}", flush=True)
        for (base, pl, rc, cand), b in zip(jobs, B):
            ch = (b or {}).get("choice", -1)
            if isinstance(ch, int) and 1 <= ch <= len(cand):
                tid = cand[ch - 1]
                out.append({**base, "territory_id": tid, "region_code": index.region_of(tid),
                            "level": "mo", "confidence": "llm", "matched": json.dumps(pl, ensure_ascii=False)})
            elif rc is not None and ch in (0, -1):
                out.append({**base, "territory_id": None, "region_code": rc, "level": "region", "confidence": "llm",
                            "matched": json.dumps(pl, ensure_ascii=False)})
    res = pd.DataFrame(out)
    if len(res):
        res["territory_id"] = res.territory_id.astype("Int64")
        res["region_code"] = res.region_code.astype("Int64")
    return res
