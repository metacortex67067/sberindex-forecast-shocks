"""
Сборка панели расходов: territory_id × категория × месяц.

Склейка объединённых МО (справочник СберИндекса, изменения union_22 и union_23):
  ряды-предшественники за 2023 г. объединяются в ряд преемника как СРЕДНЕВЗВЕШЕННОЕ по населению
  (значения среднедушевые - складывать их нельзя). Население - на 01.01.2023 (Росстат).
  union_22: Павловский Посад (1471, 91 858 чел.) + Электрогорск (1487, 29 930) -> 3014
  union_23: город Сасово (1847, 21 220) + Сасовский МР (1866, 15 524; в данных СберИндекса отсутствует) -> 3101
            -> за 2023 г. доступен только город Сасово (58 % населения), ряд помечен как приближённый.
У преемников нет января-февраля 2024 г., поэтому в основную оценку (полные 24 мес.) они не входят.
"""
import numpy as np
import pandas as pd

from ..config import CATEGORIES, PANEL_END, PANEL_START, PROCESSED, RAW
from . import loaders as L

UNIONS = {
    3014: {1471: 91858, 1487: 29930},
    3101: {1847: 21220, 1866: 15524},
}


def glue_unions(c: pd.DataFrame) -> pd.DataFrame:
    parts, notes = [c], []
    drop = set()
    for succ, preds in UNIONS.items():
        pre = c[c.territory_id.isin(preds.keys()) & (c.month < "2024-01-01")].copy()
        pre["w"] = pre.territory_id.map(preds)
        g = pre.groupby(["category", "month"]).apply(
            lambda x: pd.Series({"y": np.average(x.y, weights=x.w), "share": x.w.sum() / sum(preds.values())}),
            include_groups=False).reset_index()
        g["territory_id"] = succ
        parts.append(g[["territory_id", "category", "month", "y"]])
        notes.append((succ, float(g.share.min()) if len(g) else 0.0))
        drop |= set(preds)
    out = pd.concat(parts, ignore_index=True)
    out = out[~out.territory_id.isin(drop)]
    return out, dict(notes)


def build_panel(save: bool = True) -> pd.DataFrame:
    c = L.load_consumption()
    c, coverage = glue_unions(c)
    t = L.territory_table()
    months = pd.date_range(PANEL_START, PANEL_END, freq="MS")

    n_obs = c.groupby(["territory_id", "category"]).month.nunique()
    c = c.join(n_obs.rename("n_obs"), on=["territory_id", "category"])
    c["complete"] = c.n_obs == len(months)
    c["glued"] = c.territory_id.isin(UNIONS.keys())
    c["glue_pop_share"] = c.territory_id.map(coverage).fillna(1.0)
    c = c.join(t[["region_code", "municipal_district_type", "municipal_district_status"]], on="territory_id")
    c["category"] = pd.Categorical(c.category, CATEGORIES)
    c = c.sort_values(["territory_id", "category", "month"]).reset_index(drop=True)
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        c.to_parquet(PROCESSED / "panel.parquet", index=False)
    return c


def build_static_features(panel: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Неизменные во времени признаки МО (известны до начала прогнозного периода)."""
    t = L.territory_table()
    ids = panel.territory_id.unique()
    s = t.loc[ids, ["region_code", "municipal_district_type", "municipal_district_status",
                    "municipal_district_center_lat", "municipal_district_center_lon"]].copy()
    s.columns = ["region_code", "mo_type", "mo_status", "lat", "lon"]
    pts = pd.read_csv(RAW / "mo_points.csv").set_index("territory_id")
    s["lat"], s["lon"] = s.lat.fillna(pts.lat), s.lon.fillna(pts.lon)
    s["is_city"] = s.mo_type.eq("городской округ").astype(int)
    s["is_federal_city_district"] = s.mo_type.str.startswith("внутригородская").astype(int)
    s["is_regional_center"] = s.mo_status.fillna("").str.contains("административный центр").astype(int)
    s["is_zato"] = s.mo_status.fillna("").str.contains("ЗАТО").astype(int)
    pop = L.load_population_2024()
    s["population"] = pop.reindex(s.index)
    for succ, preds in UNIONS.items():
        if pd.isna(s.at[succ, "population"]) if succ in s.index else False:
            s.at[succ, "population"] = sum(preds.values())
    s["log_pop"] = np.log(s.population)
    s["market_access"] = L.load_market_access().reindex(s.index)
    # доход МО на душу (тыс. руб. в год -> руб. в месяц), значения 2023 г. (для прогноза 2024 г. это год t-1)
    inc = L.load_income_mo().pivot(index="territory_id", columns="year", values="income_total_th")
    for y in (2022, 2023):
        s[f"mo_income_pc_{y}"] = inc.reindex(s.index)[y] * 1000 / 12 / s.population
    ri = pd.read_csv(RAW / "rosstat" / "region_income_spending.csv")
    for y in (2022, 2023):
        r = ri[ri.year == y].set_index("region_code")
        s[f"reg_income_pc_{y}"] = s.region_code.map(r.income_pc)
        s[f"reg_spending_pc_{y}"] = s.region_code.map(r.spending_pc)
    # для внутригородских территорий и пропусков берётся региональный доход
    s["income_pc_best_2023"] = s.mo_income_pc_2023.fillna(s.reg_income_pc_2023)
    s.index.name = "territory_id"
    if save:
        s.reset_index().to_parquet(PROCESSED / "static_features.parquet", index=False)
    return s
