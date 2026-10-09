"""
Внешние признаки «МО × месяц». Правило без утечки: признак с меткой месяца t содержит только то,
что было ИЗВЕСТНО к концу месяца t (с учётом задержек публикации). В модель при точке прогноза T
попадают признаки с меткой ≤ T.

  Новости (Telegram):  месяц публикации; число событий в самом МО, в его регионе, по стране;
                       «новизна» = всплеск относительно среднего за прошлые 12 мес.
  Погода (ERA5):       аномалии месяца t (известны по окончании месяца).
  Цены (ИПЦ региона):  публикуются в середине следующего месяца -> признак месяца t = ИПЦ за t-1.
  Ключевая ставка:     известна сразу.
  Нац. ряды СберИндекса: публикуются с задержкой ~1 мес. -> признак месяца t = значение за t-1.
"""
import numpy as np
import pandas as pd

from ..data import loaders as L

NEWS_GROUPS = {
    "security": ["drone_attack", "shelling", "evacuation", "terror"],
    "disaster": ["flood", "fire", "wildfire", "emergency_regime", "weather_extreme"],
    "infra": ["utility_outage", "transport_disruption", "industrial_accident"],
    "other": ["epidemic", "enterprise_shock"],
}
TYPE2GROUP = {t: g for g, ts in NEWS_GROUPS.items() for t in ts}
MONTHS_ALL = pd.date_range("2022-01-01", "2024-12-01", freq="MS")


def news_features(territories: pd.DataFrame) -> pd.DataFrame:
    """territories: territory_id, region_code. -> territory_id × month × признаки новостей."""
    ev = L.load_news_events()
    loc = ev[(ev.scope == "local") & (ev.level != "unresolved")].copy()
    loc["group"] = loc.event_type.map(TYPE2GROUP)
    loc = loc.drop_duplicates(["channel", "msg_id", "group", "territory_id", "region_code"])
    idx = pd.MultiIndex.from_product([territories.territory_id.unique(), MONTHS_ALL], names=["territory_id", "month"])
    out = pd.DataFrame(index=idx)
    mo = loc[loc.territory_id.notna()].astype({"territory_id": int})
    cnt = mo.groupby(["territory_id", "month", "group"]).size().unstack("group", fill_value=0)
    for g in NEWS_GROUPS:
        out[f"news_mo_{g}"] = cnt.get(g, pd.Series(dtype=float)).reindex(idx).fillna(0).values
    out["news_mo_all"] = out[[f"news_mo_{g}" for g in NEWS_GROUPS]].sum(1)
    reg = loc.groupby(["region_code", "month", "group"]).size().unstack("group", fill_value=0)
    t2r = territories.drop_duplicates("territory_id").set_index("territory_id").region_code
    ridx = pd.MultiIndex.from_arrays([idx.get_level_values(0).map(t2r), idx.get_level_values(1)])
    for g in NEWS_GROUPS:
        out[f"news_reg_{g}"] = reg.get(g, pd.Series(dtype=float)).reindex(ridx).fillna(0).values
    out["news_reg_all"] = out[[f"news_reg_{g}" for g in NEWS_GROUPS]].sum(1)
    nat = ev[ev.scope == "national"].drop_duplicates(["channel", "msg_id", "event_type"])
    nc = nat.groupby(["month", "event_type"]).size().unstack(fill_value=0).reindex(MONTHS_ALL).fillna(0)
    for col in nc.columns:
        out[f"news_nat_{col}"] = nc[col].reindex(idx.get_level_values(1)).values
    # новизна: логарифм отношения к среднему за предыдущие 12 месяцев (только прошлое)
    out = out.sort_index()
    for col in ["news_mo_all", "news_mo_security", "news_mo_disaster", "news_reg_all", "news_reg_security",
                "news_reg_disaster"]:
        base = out.groupby(level=0)[col].transform(lambda s: s.shift(1).rolling(12, min_periods=3).mean())
        out[f"{col}_surge"] = np.log1p(out[col]) - np.log1p(base)
    return out.reset_index()


def weather_features() -> pd.DataFrame:
    w = L.load_weather()
    keep = ["t2m_c_anom", "precip_mm_day_anom", "snowfall_mm_day_anom", "wind_ms_anom", "t2m_c"]
    return w[["territory_id", "month"] + keep]


def cpi_features(territories: pd.DataFrame) -> pd.DataFrame:
    """ИПЦ региона: признак месяца t = данные за t-1 (задержка публикации)."""
    c = L.load_cpi_regions()
    wide = c.pivot_table(index=["region_code", "month"], columns="group", values="cpi_mom")
    wide.columns = [{"Все товары": "cpi_all", "Продовольственные товары": "cpi_food",
                     "Непродовольственные товары": "cpi_nonfood", "Общественное питание": "cpi_catering"}[k]
                    for k in wide.columns]
    wide = wide.reset_index().sort_values(["region_code", "month"])
    for col in [c for c in wide.columns if c.startswith("cpi_")]:
        g = wide.groupby("region_code")[col]
        wide[col] = g.shift(1)  # задержка публикации
        wide[col + "_3m"] = g.transform(lambda s: s.shift(1).rolling(3).apply(lambda v: np.prod(v / 100) * 100))
    t = territories.drop_duplicates("territory_id")[["territory_id", "region_code"]]
    return t.merge(wide, on="region_code").drop(columns="region_code")


def macro_features() -> pd.DataFrame:
    """Ставка (сразу) и национальные расходы СберИндекса (с лагом 1 мес.) - общие для всех МО."""
    k = L.load_key_rate()
    out = pd.DataFrame(index=k.index)
    out["key_rate"] = k["Ключевая ставка"]
    out["key_rate_chg3"] = out.key_rate - out.key_rate.shift(3)
    out["real_rate"] = k["Ключевая ставка в реальном выражении"]
    n = L.load_national_spending()
    for col, nm in [("Всего", "total"), ("Продовольственные товары", "food"), ("Услуги", "services"),
                    ("Непродовольственные товары", "nonfood"), ("Общественное питание", "catering")]:
        s = n[col]
        out[f"nat_yoy_{nm}"] = (s / s.shift(12)).shift(1).reindex(out.index)
    return out.reset_index().rename(columns={"index": "month"})


def build_exogenous(territories: pd.DataFrame) -> pd.DataFrame:
    """Единая таблица territory_id × month (2022-01 … 2024-12) со всеми внешними признаками."""
    f = news_features(territories)
    f = f.merge(weather_features(), on=["territory_id", "month"], how="left")
    f = f.merge(cpi_features(territories), on=["territory_id", "month"], how="left")
    f = f.merge(macro_features(), on="month", how="left")
    return f
