"""Лендинг: docs/landing_template.html + docs/landing_data.json -> docs/index.html (GitHub Pages) и docs/landing_artifact.html."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sbershock.config import ROOT
tpl = (ROOT / "docs" / "landing_template.html").read_text(encoding="utf-8")
data = (ROOT / "docs" / "landing_data.json").read_text(encoding="utf-8")
for ch in ("\u2014", "\u2013", "\u2212"):
    data = data.replace(ch, "-")
body = tpl.replace("/*__DATA__*/null", data)
(ROOT / "docs" / "landing_artifact.html").write_text(body, encoding="utf-8")
page = ('<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + body.replace("<div class=\"wrap\">", "</head>\n<body>\n<div class=\"wrap\">", 1) + "\n</body>\n</html>\n")
(ROOT / "docs" / "index.html").write_text(page, encoding="utf-8")
print("docs/index.html:", len(page) // 1024, "КБ")
