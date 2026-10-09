#!/usr/bin/env bash
# Воспроизведение всех результатов: bash run_all.sh (около 1-1,5 ч на 2 ядрах, GPU не нужен).
# Долгие шаги по умолчанию не перезапускаются, берутся готовые файлы из репозитория:
#   results/bt_prophet_*.parquet                scripts/run_prophet.py               CPU, около 2 ч на вариант
#   results/bt_chronos2_*, bt_timesfm25_*       scripts/run_foundation.py            GPU
#   results/bt_arima_*.parquet                  scripts/run_classic.py --kinds arima CPU, около 4 ч на 4 ядрах
#   results/fc2025_*.parquet                    scripts/forecast_2025_foundation.py  GPU
#   data/processed/llm_full/*.parquet           scripts/run_news_llm.py              GPU, около 30 ч на T4
# Пересчитать и их: FULL=1 bash run_all.sh (нужен GPU и пакеты из requirements-gpu.txt;
# LLM-разметка идёт в отдельном окружении из requirements-llm.txt, путь к его python: PY_LLM=...).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
step() { echo; echo "=== $1"; }

step "1. Данные: панель, статические и внешние признаки"
$PY scripts/build_data.py

step "2. Новости v2 (LLM-разметка -> происшествия -> признаки МО × месяц)"
if [ "${FULL:-0}" = "1" ]; then
  ${PY_LLM:-$PY} scripts/run_news_llm.py --input-dir data/raw/news_json
  mkdir -p data/processed/llm_full && mv data/processed/news_llm_events_*.parquet data/processed/llm_full/
fi
$PY scripts/build_news_v2.py
$PY scripts/eval_news_accuracy.py

step "3. Бэктест моделей прогноза"
$PY scripts/run_backtest.py --models naive snaive snaive_growth stf
$PY scripts/run_members.py
$PY scripts/build_ensemble.py
$PY scripts/run_classic.py --kinds ets theta --modes raw stf_resid
if [ "${FULL:-0}" = "1" ]; then
  $PY scripts/run_classic.py --kinds arima --modes raw stf_resid
  $PY scripts/run_prophet.py --variant default && $PY scripts/run_prophet.py --variant tuned
  $PY scripts/run_foundation.py --backends chronos2 timesfm25 --modes raw stf_resid extended
fi
$PY scripts/build_final.py

step "4. Сравнение моделей: три режима оценки, абляция, MAE в месяцы шоков"
$PY scripts/final_tables.py
$PY scripts/ablation.py
$PY scripts/report_forecast.py

step "5. Детекция шоков: полусинтетика, реальные события, реестр, гибрид с новостями"
$PY scripts/run_changepoint.py
$PY scripts/run_realevents.py
$PY scripts/hybrid_registry.py
$PY scripts/shock_month_mae.py
for tag in v1 v2; do for mode in forecast nowcast; do
  NEWS_TAG=$tag EW_MODE=$mode $PY scripts/early_warning.py
done; done

step "6. Прогноз на 2025 год и проверка по фактическим данным СберИндекса"
if [ "${FULL:-0}" = "1" ]; then
  $PY scripts/forecast_2025_foundation.py --backends chronos2 timesfm25
fi
$PY scripts/check_fm_repro.py
$PY scripts/forecast_2025.py

step "7. Графики, лендинг, отчёт"
$PY scripts/fm_leak_compare.py
$PY scripts/make_figures.py
$PY scripts/build_landing_data.py && $PY scripts/build_landing.py
if command -v pandoc >/dev/null; then $PY scripts/render_report.py; fi
echo; echo "Готово: results/*.csv, reports/figures/*.png, docs/index.html, REPORT.md"
