"""
Сбор погоды для всех МО из реанализа ERA5 (Copernicus, ежемесячные средние).

Зачем ERA5, а не Open-Meteo: бесплатный лимит Open-Meteo (~10 тыс. вызовов в сутки)
не позволяет выгрузить дневную погоду для 2 660 МО за несколько лет. ERA5 отдаёт
одну сетку по всей России за один запрос, точки МО вырезаются из неё.

Что нужно сделать один раз:
  1. Зарегистрироваться на https://cds.climate.copernicus.eu
  2. На странице датасета "ERA5 monthly averaged data on single levels"
     принять лицензию (кнопка внизу вкладки Download).
  3. Взять Personal Access Token в профиле и создать файл ~/.cdsapirc:
        url: https://cds.climate.copernicus.eu/api
        key: <ВАШ_ТОКЕН>
     Вместо файла можно задать переменные окружения
     CDSAPI_URL и CDSAPI_KEY.

Запуск:
  pip install cdsapi xarray netCDF4 pandas pyarrow
  python collect_weather_era5.py --points mo_points.csv --out weather_mo_monthly.parquet

Если файл скачан вручную с сайта Copernicus (без токена), подходит .nc, .grib или .zip
(для .grib нужен ещё: pip install cfgrib eccodes):
  python collect_weather_era5.py --from-files ОСНОВНОЙ.nc            # Чукотка возьмётся с края сетки
  python collect_weather_era5.py --from-files ОСНОВНОЙ.nc ЧУКОТКА.nc  # если скачан и второй файл

Результат (прислать в чат): weather_mo_monthly.parquet, несколько МБ.
Колонки: territory_id, month, t2m_c, precip_mm_day, snowfall_mm_day, wind_ms,
         и аномалии *_anom относительно среднего того же месяца за 2015-2022
         (база до начала панели, чтобы не было заглядывания в будущее).
"""
import argparse
import os
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

DATASET = "reanalysis-era5-single-levels-monthly-means"
VARIABLES = ["2m_temperature", "total_precipitation", "snowfall", "10m_wind_speed"]
YEARS = [str(y) for y in range(2015, 2026)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
BASELINE_YEARS = (2015, 2022)

# две области: основная Россия и Чукотка восточнее 180° (там долготы отрицательные)
AREAS = {
    "main": [82, 19, 41, 180],  # [N, W, S, E]
    "east": [72, -180, 62, -168],
}


def download(area_name, area, target):
    if os.path.exists(target):
        print(f"[skip] {target} уже скачан")
        return
    import cdsapi
    req = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": VARIABLES,
        "year": YEARS,
        "month": MONTHS,
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }
    print(f"[cds] запрос {area_name}: {area}")
    cdsapi.Client().retrieve(DATASET, req).download(target)


def is_grib(path):
    with open(path, "rb") as f:
        return f.read(4) == b"GRIB"


def open_grib(path):
    """GRIB (формат по умолчанию в форме CDS). Нужен пакет cfgrib: pip install cfgrib eccodes"""
    import cfgrib
    parts = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    fixed = []
    for ds in parts:
        # в GRIB накопленные величины (осадки, снег) помечены 18:00 последнего дня предыдущего месяца;
        # сдвигаю на +12 ч и округляю вниз до начала месяца, иначе осадки отстанут от температуры на месяц
        t = pd.to_datetime(ds["time"].values) + pd.Timedelta(hours=12)
        month_start = t.to_period("M").to_timestamp()
        ds = ds.drop_vars([c for c in ["valid_time", "step", "surface"] if c in ds.coords])
        fixed.append(ds.assign_coords(time=month_start))
    return xr.merge(fixed, compat="override", join="outer")


def open_any(path):
    """Поддерживаются: NetCDF (.nc), GRIB (.grib), zip с файлами внутри."""
    if is_grib(path):
        print(f"[info] {os.path.basename(path)}: формат GRIB")
        return open_grib(path)
    if zipfile.is_zipfile(path):
        folder = path + "_unzipped"
        with zipfile.ZipFile(path) as z:
            z.extractall(folder)
        parts = []
        for f in sorted(os.listdir(folder)):
            fp = os.path.join(folder, f)
            if f.endswith(".nc"):
                parts.append(xr.open_dataset(fp))
            elif is_grib(fp):
                parts.append(open_grib(fp))
        return xr.merge(parts, compat="override")
    return xr.open_dataset(path)


def normalize(ds):
    if "time" in ds.dims:
        ds = ds.drop_vars([c for c in ["valid_time", "step", "surface"] if c in ds.coords])
    elif "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    for extra in ["expver", "number"]:
        if extra in ds.dims:
            ds = ds.isel({extra: 0})
        if extra in ds.coords:
            ds = ds.drop_vars(extra)
    return ds


def sample(ds, pts):
    lat = xr.DataArray(pts["lat"].values, dims="p")
    lon = xr.DataArray(pts["lon"].values, dims="p")
    s = ds.sel(latitude=lat, longitude=lon, method="nearest")
    df = s.to_dataframe().reset_index()
    df["territory_id"] = pts["territory_id"].values[df["p"].values]
    return df.drop(columns=[c for c in ["p", "latitude", "longitude"] if c in df.columns])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="mo_points.csv")
    ap.add_argument("--out", default="weather_mo_monthly.parquet")
    ap.add_argument("--workdir", default="era5_raw")
    ap.add_argument("--from-files", nargs="*",
                    help="уже скачанные вручную .nc: сначала основной, затем (необязательно) Чукотка")
    a = ap.parse_args()

    pts = pd.read_csv(a.points)
    east = pts["lon"] < 0
    os.makedirs(a.workdir, exist_ok=True)

    files = {}
    if a.from_files:
        files = dict(zip(["main", "east"], a.from_files))
    else:
        for name, area in AREAS.items():
            target = os.path.join(a.workdir, f"era5_{name}.nc")
            download(name, area, target)
            files[name] = target

    frames = []
    if "east" in files:
        groups = [("main", pts[~east]), ("east", pts[east])]
    else:
        # 3 точки Чукотки (долгота < 0) берутся с восточного края сетки (180°)
        print("[info] файла по Чукотке нет - 3 восточные точки берутся с ближайшего края сетки (180°)")
        p2 = pts.copy()
        p2.loc[east, "lon"] = 180.0
        groups = [("main", p2)]
    for name, sub in groups:
        if len(sub) == 0:
            continue
        ds = normalize(open_any(files[name]))
        frames.append(sample(ds, sub.reset_index(drop=True)))
    df = pd.concat(frames, ignore_index=True)

    # единицы: K -> °C; м/сутки (среднее за месяц) -> мм/сутки
    out = pd.DataFrame({
        "territory_id": df["territory_id"].astype(int),
        "month": pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp(),
    })
    conv = {
        "t2m": ("t2m_c", lambda v: v - 273.15),
        "tp": ("precip_mm_day", lambda v: v * 1000),
        "sf": ("snowfall_mm_day", lambda v: v * 1000),
        "si10": ("wind_ms", lambda v: v),
    }
    found = []
    for src, (col, f) in conv.items():
        if src in df.columns:
            out[col] = f(df[src].values)
            found.append(col)
        else:
            print(f"[warn] в файле нет переменной {src} ({col}), пропускаю")
    assert found, "в файле не найдено ни одной нужной переменной"

    # аномалии относительно климата того же месяца за базовый период до начала панели
    out["m"] = out["month"].dt.month
    base = out[out["month"].dt.year.between(*BASELINE_YEARS)]
    clim = base.groupby(["territory_id", "m"])[found].mean()
    out = out.join(clim, on=["territory_id", "m"], rsuffix="_clim")
    for v in found:
        out[f"{v}_anom"] = out[v] - out[f"{v}_clim"]
        out = out.drop(columns=f"{v}_clim")
    out = out.drop(columns="m").sort_values(["territory_id", "month"])

    n_mo, n_m = out["territory_id"].nunique(), out["month"].nunique()
    print(f"[ok] МО: {n_mo}, месяцев: {n_m}, строк: {len(out)}, пропусков: {int(out.isna().sum().sum())}")
    assert n_mo == len(pts), "не все МО получили погоду"
    out.to_parquet(a.out, index=False)
    print(f"[ok] сохранено: {a.out}")


if __name__ == "__main__":
    main()
