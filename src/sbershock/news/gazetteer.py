"""
Газеттир - словарь «название места → территория СберИндекса».

Три уровня:
  region     - субъект РФ (85 регионов справочника) + разговорные варианты (Кузбасс, Югра, Подмосковье…)
  mo         - муниципалитет верхнего уровня (territory_id) по вариантам его названия
  settlement - населённый пункт ИНИД; привязка к territory_id через попадание точки в полигон МО

Строится один раз скриптом scripts/build_gazetteer.py и сохраняется в data/processed/gazetteer.parquet,
чтобы обработка новостей не требовала полигонов и справочника.
"""
import re

import pandas as pd

GENERIC_PREFIXES = [
    "городской округ город-курорт ", "городской округ город ", "городской округ ", "город-курорт ",
    "город ", "г. ", "г ", "поселок городского типа ", "посёлок городского типа ", "рабочий поселок ",
    "поселок ", "посёлок ", "пгт ", "село ", "деревня ", "станица ", "хутор ", "аул ",
]

REGION_ALIASES = {
    "Москва": ["москва", "столица"],
    "Санкт-Петербург": ["петербург", "санкт-петербург", "питер", "спб"],
    "Московская область": ["подмосковье", "московский регион"],
    "Ленинградская область": ["ленобласть"],
    "Кемеровская область": ["кузбасс", "кемеровская область - кузбасс", "кемеровская область — кузбасс"],
    "Ханты-Мансийский автономный округ — Югра": ["югра", "хмао", "хмао-югра", "ханты-мансийский автономный округ",
                                                "ханты-мансийский автономный округ - югра"],
    "Ямало-Ненецкий автономный округ": ["ямал", "янао", "ямало-ненецкий округ"],
    "Ненецкий автономный округ": ["нао"],
    "Республика Саха (Якутия)": ["якутия", "республика саха", "саха"],
    "Республика Башкортостан": ["башкирия", "башкортостан"],
    "Республика Татарстан": ["татарстан", "татария"],
    "Чувашская Республика": ["чувашия"],
    "Удмуртская Республика": ["удмуртия"],
    "Республика Мордовия": ["мордовия"],
    "Чеченская Республика": ["чечня"],
    "Кабардино-Балкарская Республика": ["кабардино-балкария", "кбр"],
    "Карачаево-Черкесская Республика": ["карачаево-черкесия", "кчр"],
    "Республика Северная Осетия — Алания": ["северная осетия", "северная осетия-алания", "алания"],
    "Республика Крым": ["крым"],
    "Севастополь": ["севастополь"],
    "Приморский край": ["приморье"],
    "Ставропольский край": ["ставрополье"],
    "Краснодарский край": ["кубань"],
    "Забайкальский край": ["забайкалье"],
    "Чукотский автономный округ": ["чукотка"],
    "Камчатский край": ["камчатка"],
    "Республика Хакасия": ["хакасия"],
    "Республика Тыва": ["тыва", "тува"],
    "Республика Бурятия": ["бурятия"],
    "Республика Калмыкия": ["калмыкия"],
    "Республика Карелия": ["карелия"],
    "Республика Коми": ["коми"],
    "Республика Адыгея": ["адыгея"],
    "Республика Алтай": ["горный алтай"],
    "Республика Ингушетия": ["ингушетия"],
    "Республика Дагестан": ["дагестан"],
    "Республика Марий Эл": ["марий эл"],
    "Еврейская автономная область": ["еао"],
    "Пермский край": ["прикамье"],
    "Алтайский край": ["алтайский край"],
}

# названия, совпадающие с объектами вне справочника (новые регионы, зарубежье),
# принимаются только при явном упоминании своего региона в том же сообщении
RISKY_NAMES = {"донецк", "алтай", "мирный", "киров", "троицк", "красноармейск", "октябрьский",
               "советский", "первомайский", "заречный", "лесной", "северск", "кировск"}


def norm(s: str) -> str:
    s = str(s).lower().replace("ё", "е").replace("«", "").replace("»", "").replace('"', "")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip(" .,")
    return s


def strip_generic(s: str) -> str:
    s = norm(s)
    for p in GENERIC_PREFIXES:
        pn = norm(p) + " "
        if s.startswith(pn):
            return s[len(pn):]
    return s


def mo_aliases(name: str, short: str, mtype: str) -> set:
    """Варианты названия муниципалитета так, как они встречаются в новостях после нормализации Natasha."""
    out = {norm(name)}
    sh = norm(short) if isinstance(short, str) and short.strip() else None
    if not sh:
        return out
    if mtype == "городской округ":
        if re.search(r"(ский|цкий|ской|ный)$", sh):
            out |= {f"{sh} городской округ", f"{sh} округ", f"{sh} район"}
        else:
            out |= {sh, f"город {sh}", f"городской округ {sh}"}
    elif mtype in ("муниципальный район", "муниципальный округ"):
        out |= {f"{sh} район", f"{sh} муниципальный район", f"{sh} округ", f"{sh} муниципальный округ",
                f"{sh} городской округ"}
    else:
        out |= {f"район {sh}", f"муниципальный округ {sh}", f"{sh} район", sh}
    return out


def build(dict_xlsx: str, poly_gpkg: str, settlements_csv: str) -> pd.DataFrame:
    import geopandas as gpd

    d = pd.read_excel(dict_xlsx)
    last = d.sort_values("year_from").groupby("territory_id").tail(1)
    regions = d[["region_code", "region_name"]].drop_duplicates()
    reg_code = dict(zip(regions.region_name, regions.region_code))

    rows = []
    for rname, rcode in reg_code.items():
        al = {norm(rname)}
        if rname.startswith("Республика "):
            al.add(norm(rname.replace("Республика ", "")))
        al |= set(REGION_ALIASES.get(rname, []))
        for a in al:
            rows.append(dict(alias=norm(a), level="region", territory_id=pd.NA, region_code=rcode,
                             region_name=rname, population=pd.NA))

    # муниципалитеты: только версии, действовавшие в 2022-2024 (период новостей)
    for r in d[d.year_to > 2022].itertuples():
        for a in mo_aliases(r.municipal_district_name, r.municipal_district_name_short, r.municipal_district_type):
            rows.append(dict(alias=a, level="mo", territory_id=r.territory_id, region_code=r.region_code,
                             region_name=r.region_name, population=pd.NA))

    # населённые пункты ИНИД -> territory_id через полигоны
    s = pd.read_csv(settlements_csv, dtype={"oktmo": str})
    s = s[s.population > 0]
    # города федерального значения обрабатываются как регионы (иначе «Москва» попадёт в один район)
    s = s[~s.settlement.map(norm).isin({"москва", "санкт-петербург", "севастополь"})]
    g = gpd.read_file(poly_gpkg)[["territory_id", "geometry"]]
    g["territory_id"] = g.territory_id.astype(int)
    pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s.longitude_dd, s.latitude_dd), crs="EPSG:4326")
    j = gpd.sjoin(pts, g, how="left", predicate="within").drop_duplicates("id")
    j = j[j.territory_id.notna()]
    tid_region = last.set_index("territory_id")[["region_code", "region_name"]]
    j = j.join(tid_region, on="territory_id")
    for r in j.itertuples():
        rows.append(dict(alias=norm(r.settlement), level="settlement", territory_id=int(r.territory_id),
                         region_code=r.region_code, region_name=r.region_name, population=int(r.population)))

    gz = pd.DataFrame(rows).drop_duplicates(["alias", "level", "territory_id", "region_code"])
    gz = gz[gz.alias.str.len() >= 3]
    gz["territory_id"] = gz.territory_id.astype("Int64")
    gz["population"] = gz.population.astype("Int64")
    return gz.reset_index(drop=True)
