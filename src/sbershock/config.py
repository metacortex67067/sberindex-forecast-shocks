"""Общие настройки проекта. Гиперпараметры моделей - в configs/*.yaml."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

CATEGORIES = ["Все категории", "Продовольствие", "Маркетплейсы", "Общественное питание", "Транспорт", "Здоровье"]
CAT_SHORT = {"Все категории": "total", "Продовольствие": "food", "Маркетплейсы": "marketplaces",
             "Общественное питание": "catering", "Транспорт": "transport", "Здоровье": "health"}

# национальные ряды СберИндекса (consumer-spending) для категорий панели;
# для категорий без прямого аналога взят ближайший агрегат
NATIONAL_MAP = {"Все категории": "Всего", "Продовольствие": "Продовольственные товары",
                "Общественное питание": "Общественное питание", "Маркетплейсы": "Непродовольственные товары",
                "Транспорт": "Услуги", "Здоровье": "Непродовольственные товары"}

PANEL_START, PANEL_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-01")
HORIZONS = [1, 3, 6, 12]
MIN_TRAIN_MONTHS = 12


def eval_origins(h: int):
    """Точки прогноза T (последний известный месяц): T >= 2023-12 (минимум 12 мес. истории) и T + h <= 2024-12."""
    first = PANEL_START + pd.DateOffset(months=MIN_TRAIN_MONTHS - 1)
    last = PANEL_END - pd.DateOffset(months=h)
    return list(pd.date_range(first, last, freq="MS"))


def load_yaml(name: str) -> dict:
    with open(ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)
