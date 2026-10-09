"""Проверки геопривязки и классификации на типичных фразах из новостей."""
import os, sys
import pandas as pd
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sbershock.news.events import EventClassifier
from sbershock.news.pipeline import GeoResolver

GZ = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "gazetteer.parquet")


@pytest.fixture(scope="module")
def res():
    return GeoResolver(pd.read_parquet(GZ))


def tids(hits):
    return {h["territory_id"] for h in hits if h["territory_id"] is not None}


def regs(hits):
    return {h["region_code"] for h in hits}


def test_city_genitive(res):  # «Администрация Орска»: NER видит организацию, город находит запасной путь
    assert 1673 in tids(res.resolve("Администрация Орска набирает добровольцев для эвакуации"))


def test_district_with_region(res):
    assert 662 in tids(res.resolve("В Туапсинском районе Краснодарского края четыре человека пропали"))


def test_region_only(res):
    h = res.resolve("Режим ЧС введен в Оренбургской области")
    assert 56 in regs(h) and not tids(h)


def test_moscow_is_region_not_district(res):
    h = res.resolve("Пожар на рынке в Москве потушен")
    assert 77 in regs(h) and not tids(h)


def test_foreign_not_resolved(res):
    assert res.resolve("Пожар в турецком порту Искендерун продолжается") == []


def test_risky_name_needs_region(res):  # Донецк (ДНР) не должен стать Донецком Ростовской области
    assert 1795 not in tids(res.resolve("В Донецке прозвучали взрывы"))


def test_classifier():
    clf = EventClassifier()
    assert clf.classify("В Орске объявлена эвакуация из-за паводка")[0] == ["flood", "evacuation"]
    assert clf.classify("Банк России сохранил ключевую ставку")[1] == ["key_rate"]
    assert clf.classify("МЧС провело учения по тушению пожара") == ([], [])
