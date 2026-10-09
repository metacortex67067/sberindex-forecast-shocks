"""
Загрузка и приведение всех источников к опрятному виду.
Ключи: territory_id (МО в постоянных границах), region_code, month (первое число месяца).
"""
import re

import numpy as np
import pandas as pd

from ..config import RAW

MONTHS_RU = {"январь": 1, "февраль": 2, "март": 3, "апрель": 4, "май": 5, "июнь": 6, "июль": 7,
             "август": 8, "сентябрь": 9, "октябрь": 10, "ноябрь": 11, "декабрь": 12}


def _oktmo8(x) -> str:
    return re.sub(r"\D", "", str(x)).zfill(8)[:8] if pd.notna(x) else None


def load_dictionary() -> pd.DataFrame:
    d = pd.read_excel(RAW / "t_dict_municipal_districts.xlsx")
    d["oktmo8"] = d.oktmo.str.replace("-", "").str[:8]
    return d


def territory_table() -> pd.DataFrame:
    """Одна строка на territory_id: последняя версия (название, регион, тип, статус, координаты)."""
    d = load_dictionary().sort_values("year_from")
    last = d.groupby("territory_id").tail(1).set_index("territory_id")
    return last[["region_code", "region_name", "municipal_district_name", "municipal_district_name_short",
                 "municipal_district_type", "municipal_district_status", "oktmo8",
                 "municipal_district_center_lat", "municipal_district_center_lon"]]


def load_consumption() -> pd.DataFrame:
    c = pd.read_parquet(RAW / "sber" / "consumption.parquet")
    c = c.rename(columns={"value": "y"})
    c["month"] = pd.to_datetime(c["date"] + "-01")
    c["territory_id"] = c.territory_id.astype(int)
    return c[["territory_id", "category", "month", "y"]].astype({"y": float})


def load_market_access() -> pd.Series:
    m = pd.read_parquet(RAW / "sber" / "market_access.parquet")
    return m.set_index(m.territory_id.astype(int)).market_access


def load_population_2024() -> pd.Series:
    """Население на 01.01.2024 по territory_id: сопоставление по ОКТМО (8 знаков), остаток - по названию."""
    import openpyxl
    wb = openpyxl.load_workbook(RAW / "rosstat" / "BUL_MO_2024.xlsx", read_only=True)
    rows = [r[:3] for r in wb["Численность_по_МО"].iter_rows(min_row=6, values_only=True) if r[0] and r[2]]
    b = pd.DataFrame(rows, columns=["code", "name", "pop"])
    b["code"] = b.code.astype(str).str.replace(" ", "")
    b = b[b.code.str.len() == 10]
    b["oktmo8"] = b.code.str[:8]
    b["pop"] = pd.to_numeric(b["pop"], errors="coerce")
    t = territory_table()
    pop = t.oktmo8.map(b.drop_duplicates("oktmo8").set_index("oktmo8")["pop"])
    # остаток сопоставляется по нормализованному названию внутри региона
    miss = pop[pop.isna()].index
    if len(miss):
        key = b.name.str.lower().str.replace("ё", "е").str.strip()
        lut = dict(zip(key, b["pop"]))
        for tid in miss:
            nm = str(t.at[tid, "municipal_district_name"]).lower().replace("ё", "е")
            pop.at[tid] = lut.get(nm, np.nan)
    return pop.rename("population")


def load_income_mo(years=(2021, 2022, 2023, 2024)) -> pd.DataFrame:
    """Годовой объём доходов МО, тыс. руб. -> territory_id × year. Для городов федерального значения -
    значение города целиком, присвоенное всем внутригородским территориям (доход на душу считается по городу)."""
    t = territory_table()
    out = []
    xl = pd.ExcelFile(RAW / "rosstat" / "urov_2010-2024.xlsx")
    for y in years:
        x = pd.read_excel(xl, str(y), header=None).iloc[5:, :4]
        x.columns = ["n", "name", "ok", "val"]
        x = x[x.ok.notna()].copy()
        x["oktmo8"] = x.ok.map(_oktmo8)
        x["val"] = pd.to_numeric(x.val, errors="coerce")
        v = t.oktmo8.map(x.drop_duplicates("oktmo8").set_index("oktmo8").val)
        out.append(pd.DataFrame({"territory_id": t.index, "year": y, "income_total_th": v.values}))
    return pd.concat(out, ignore_index=True)


REGION_FIX = {
    "москва": "Москва", "санкт-петербург": "Санкт-Петербург", "севастополь": "Севастополь",
    "кемеровская": "Кемеровская область", "адыгея": "Республика Адыгея", "татарстан": "Республика Татарстан",
    "чувашская": "Чувашская Республика", "северная осетия": "Республика Северная Осетия — Алания",
    "ханты-мансийский": "Ханты-Мансийский автономный округ — Югра",
    "ямало-ненецкий": "Ямало-Ненецкий автономный округ", "ненецкий автономный округ": "Ненецкий автономный округ",
    "республика саха": "Республика Саха (Якутия)",
}


def _match_region(name: str, regions: list) -> str:
    n = re.sub(r"\s+", " ", str(name)).strip().lower().replace("ё", "е").replace("–", "-")
    n = re.sub(r"^в том числе:? ?", "", n)
    if "без автономн" in n:
        n = n.split(" без ")[0].strip()
        return {"архангельская область": "Архангельская область", "тюменская область": "Тюменская область"}.get(n)
    if n in ("архангельская область", "тюменская область"):  # агрегаты, включающие автономные округа
        return None
    if "кроме" in n:  # «Архангельская область (кроме НАО)» = сама область
        n = n.split("(")[0].strip()
    head = n.split("(")[0].strip()
    for k, v in REGION_FIX.items():
        if k in head and not (k == "ненецкий автономный округ" and "ямало" in head):
            return v
    for r in regions:
        if r.lower().replace("ё", "е") == head:
            return r
    return None


def load_cpi_regions() -> pd.DataFrame:
    """ИПЦ к предыдущему месяцу (%) -> region_code × month × группа, плюс накопленный индекс (2019-01 = 100)."""
    x = pd.read_excel(RAW / "rosstat" / "cpi_regions_fedstat.xls", header=None, engine="xlrd")
    years, months, measures = x.iloc[2].ffill(), x.iloc[3].ffill(), x.iloc[4]
    cols = [i for i in range(2, x.shape[1])
            if str(measures[i]).startswith("К предыдущему месяцу") and str(months[i]).lower() in MONTHS_RU]
    d = load_dictionary()
    regions = sorted(d.region_name.unique())
    code = d.drop_duplicates("region_name").set_index("region_name").region_code
    body = x.iloc[5:].copy()
    body[0] = body[0].ffill()
    rows = []
    for _, r in body.iterrows():
        reg = _match_region(r[0], regions)
        if reg is None or pd.isna(r[1]):
            continue
        for i in cols:
            v = pd.to_numeric(r[i], errors="coerce")
            rows.append((code[reg], str(r[1]).strip(), pd.Timestamp(int(float(years[i])), MONTHS_RU[str(months[i]).lower()], 1), v))
    c = pd.DataFrame(rows, columns=["region_code", "group", "month", "cpi_mom"]).drop_duplicates(
        ["region_code", "group", "month"])
    c = c.sort_values(["region_code", "group", "month"])
    c["cpi_mom"] = c.groupby(["region_code", "group"]).cpi_mom.transform(lambda s: s.interpolate(limit_direction="both"))
    c["cpi_index"] = c.groupby(["region_code", "group"]).cpi_mom.transform(lambda s: 100 * np.cumprod(s / 100))
    return c


def load_weather() -> pd.DataFrame:
    return pd.read_parquet(RAW / "weather_mo_monthly.parquet")


def _sber_csv(name):
    d = pd.read_csv(RAW / "sber" / f"{name}.csv", sep=None, engine="python")
    d["month"] = pd.to_datetime(d.period).dt.to_period("M").dt.to_timestamp()
    return d


def load_national_spending() -> pd.DataFrame:
    """Потребительские расходы РФ, млрд руб., по 5 агрегатам, 2018-12 …"""
    d = _sber_csv("consumer-spending")
    return d.pivot_table(index="month", columns="type", values="value")


def load_key_rate() -> pd.DataFrame:
    d = _sber_csv("real-key-interest-rate")
    return d.pivot_table(index="month", columns="key_rate_categories", values="value")


def load_news_events() -> pd.DataFrame:
    from ..config import PROCESSED
    fs = sorted(PROCESSED.glob("news_events_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def load_weekly_categories() -> pd.DataFrame:
    """Недельное изменение трат по 46 категориям, % г/г (СберИндекс), с 11.2023. Индекс - конец недели."""
    d = pd.read_csv(RAW / "sber" / "ver-izmenenie-trat-po-kategoriyam.csv", sep=None, engine="python")
    d["period"] = pd.to_datetime(d.period)
    d["category"] = d.category.str.strip()
    return d.pivot_table(index="period", columns="category", values="value")
