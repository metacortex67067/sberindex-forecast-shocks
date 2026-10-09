"""
Новости v2 (LLM-разметка) -> признаки «МО × месяц».

1. Склейка повторов в ПРОИСШЕСТВИЯ: сообщения об одном месте (МО или регион), одном типе и с разрывом
   не более GAP_DAYS дней между соседними сообщениями в пределах одного месяца - одно происшествие (цепочка). Так «296 сообщений о
   паводке в Орске» становятся несколькими происшествиями, а громкость (число сообщений) - отдельным признаком.
2. Тяжесть происшествия = максимум тяжести (0-3) по его сообщениям.
3. Признаки месяца t (по дате публикации; только прошлое и настоящее):
   news2_mo_{группа}        - число происшествий в МО (уровень МО)
   news2_mo_sev             - сумма тяжести происшествий в МО
   news2_mo_max_sev         - максимальная тяжесть в МО
   news2_mo_msgs            - число сообщений (громкость)
   news2_reg_{группа}, news2_reg_sev - то же для региона (включая события, привязанные к любому МО региона)
   news2_*_surge            - всплеск относительно среднего за предыдущие 12 мес.
"""
import glob

import numpy as np
import pandas as pd

GAP_DAYS = 3
GROUPS = {"security": ["drone_attack", "shelling", "evacuation", "terror"],
          "disaster": ["flood", "fire", "wildfire", "emergency_regime", "weather_extreme"],
          "infra": ["utility_outage", "transport_disruption", "industrial_accident"],
          "other": ["epidemic", "enterprise_shock"]}
T2G = {t: g for g, ts in GROUPS.items() for t in ts}
MONTHS = pd.date_range("2022-01-01", "2024-12-01", freq="MS")


def load_events(folder) -> pd.DataFrame:
    fs = [f for f in sorted(glob.glob(f"{folder}/news_llm_events_*.parquet")) if "_sample" not in f]
    ev = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    ev = ev.drop_duplicates(["channel", "msg_id", "event_type", "territory_id", "region_code"])
    ev["group"] = ev.event_type.map(T2G)
    ev["severity"] = pd.to_numeric(ev.severity, errors="coerce").fillna(1).clip(0, 3)
    ev["place"] = np.where(ev.territory_id.notna(), "mo:" + ev.territory_id.astype("Int64").astype(str),
                           "reg:" + ev.region_code.astype("Int64").astype(str))
    return ev[ev.group.notna()]


def to_incidents(ev: pd.DataFrame) -> pd.DataFrame:
    """Цепочки сообщений (место, тип) с разрывом ≤ GAP_DAYS -> происшествия.
    Цепочка обрывается на границе месяца: признак месяца t строится только из сообщений месяца t
    (тяжесть и громкость не «подтягиваются» из следующих месяцев, а длительная ситуация - например,
    регулярные атаки БПЛА на регион - учитывается в каждом месяце, когда о ней пишут)."""
    ev = ev.sort_values(["place", "event_type", "published_msk"])
    gap = ev.groupby(["place", "event_type", "month"]).published_msk.diff().dt.days.fillna(1e9)
    ev["incident"] = (gap > GAP_DAYS).cumsum()
    inc = ev.groupby("incident").agg(place=("place", "first"), territory_id=("territory_id", "first"),
                                     region_code=("region_code", "first"), event_type=("event_type", "first"),
                                     group=("group", "first"), start=("published_msk", "min"),
                                     severity=("severity", "max"), msgs=("msg_id", "size"),
                                     channels=("channel", "nunique")).reset_index()
    inc["month"] = inc.start.dt.to_period("M").dt.to_timestamp()
    return inc


def build_features(folder, territories: pd.DataFrame) -> pd.DataFrame:
    """territories: territory_id, region_code (МО панели)."""
    ev = load_events(folder)
    inc = to_incidents(ev)
    tids = territories.territory_id.unique()
    t2r = territories.drop_duplicates("territory_id").set_index("territory_id").region_code
    idx = pd.MultiIndex.from_product([tids, MONTHS], names=["territory_id", "month"])
    out = pd.DataFrame(index=idx)
    mo = inc[inc.territory_id.notna()].astype({"territory_id": int})
    for g in GROUPS:
        out[f"news2_mo_{g}"] = mo[mo.group == g].groupby(["territory_id", "month"]).size().reindex(idx).fillna(0).values
    out["news2_mo_all"] = out[[f"news2_mo_{g}" for g in GROUPS]].sum(1)
    out["news2_mo_sev"] = mo.groupby(["territory_id", "month"]).severity.sum().reindex(idx).fillna(0).values
    out["news2_mo_max_sev"] = mo.groupby(["territory_id", "month"]).severity.max().reindex(idx).fillna(0).values
    msg_mo = ev[ev.territory_id.notna()].astype({"territory_id": int})
    out["news2_mo_msgs"] = msg_mo.groupby(["territory_id", "month"]).msg_id.nunique().reindex(idx).fillna(0).values
    # регион: все происшествия региона (уровень региона и любые МО региона)
    ridx = pd.MultiIndex.from_arrays([idx.get_level_values(0).map(t2r), idx.get_level_values(1)])
    for g in GROUPS:
        out[f"news2_reg_{g}"] = inc[inc.group == g].groupby(["region_code", "month"]).size().reindex(ridx).fillna(0).values
    out["news2_reg_all"] = out[[f"news2_reg_{g}" for g in GROUPS]].sum(1)
    out["news2_reg_sev"] = inc.groupby(["region_code", "month"]).severity.sum().reindex(ridx).fillna(0).values
    out = out.sort_index()
    for col in ["news2_mo_all", "news2_mo_sev", "news2_mo_msgs", "news2_reg_all", "news2_reg_sev"]:
        base = out.groupby(level=0)[col].transform(lambda s: s.shift(1).rolling(12, min_periods=3).mean())
        out[f"{col}_surge"] = np.log1p(out[col]) - np.log1p(base)
    return out.reset_index(), inc, ev
