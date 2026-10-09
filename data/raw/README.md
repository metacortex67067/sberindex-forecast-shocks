# Исходные данные (data/raw)

| файл | источник | нужен для |
|---|---|---|
| `sber/consumption.parquet`, `sber/market_access.parquet` | СберИндекс, набор хакатона | панель, статические признаки |
| `t_dict_municipal_districts.xlsx`, `t_dict_municipal_districts_poly.gpkg` | СберИндекс, справочник МО (`t_dict_municipal.rar`) | склейка МО; полигоны нужны только для газеттира новостей v1 |
| `sber/consumer-spending.csv` | СберИндекс: потребительские расходы РФ, мес. | STF, foundation-модели, проверка 2025 г. |
| `sber/ver-izmenenie-trat-po-kategoriyam.csv` | СберИндекс: изменение трат по категориям, нед. | рост маркетплейсов, здоровья, транспорта; проверка 2025 г. |
| `sber/real-key-interest-rate.csv` | СберИндекс, ЦМИ | внешние признаки |
| `rosstat/BUL_MO_2024.xlsx` (и 2023) | Росстат: численность населения МО | веса, признаки |
| `rosstat/urov_2010-2024.xlsx` | Росстат: доходы МО | признаки |
| `rosstat/den_dohod/*` (бюллетени «Денежные доходы и расходы населения», .doc) | Росстат | `region_income_spending.csv` (scripts/convert_income_bulletins.py, нужен LibreOffice) |
| `rosstat/cpi_regions_fedstat.xls` | ЕМИСС: ИПЦ по субъектам | признаки |
| `weather_mo_monthly.parquet`, `mo_points.csv` | ERA5 (Copernicus), scripts/collect_weather_era5.py | признаки |
| `inid_settlements.csv` | ИНИД: населённые пункты | газеттир новостей v1 |
| `news_json/*.json` | экспорт Telegram Desktop: ТАСС, РИА Новости, Интерфакс, МЧС России | только `FULL=1` (разметка новостей) |

Имена CSV СберИндекса указаны без суффикса выгрузки: `consumer-spending_ru_<id>.csv` сохраняется как `consumer-spending.csv`.

Файлы `t_dict_municipal_districts_poly.gpkg` и `inid_settlements.csv` в репозиторий не включены из-за размера. Они нужны только для пересборки газеттира новостей v1, готовый газеттир лежит в `data/processed/gazetteer.parquet`.
