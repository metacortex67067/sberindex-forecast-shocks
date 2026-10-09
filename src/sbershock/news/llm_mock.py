"""Заглушка LLM для проверки обвязки без GPU: тип по правилам v1, место - первый найденный регион/город."""
import re
from .events import EventClassifier
class MockLLM:
    def __init__(self):
        self.clf = EventClassifier()
    def json_batch(self, system, users, schema, max_tokens=300):
        out = []
        for u in users:
            if "Кандидаты" in u:
                out.append({"choice": 1 if "\n1." in u else 0}); continue
            loc, _ = self.clf.classify(u)
            m = re.search(r"(Оренбургск\w+ област\w+|Курганск\w+ област\w+|Москв\w+|Орск\w*)", u)
            pl = {"settlement": None, "district": None, "city": "Орск" if m and "Орск" in m.group(0) else None,
                  "region": "Оренбургская область" if m else None}
            out.append({"is_incident": bool(loc), "event_type": loc[0] if loc else "other", "severity": 1,
                        "in_russia": True, "places": [pl] if m else []})
        return out
