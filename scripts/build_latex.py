"""
LaTeX-версия отчёта: REPORT.md (собирается scripts/render_report.py) -> reports/latex/REPORT.tex + figures/ (+ проверочная сборка PDF).
Компиляция: XeLaTeX (на Overleaf: Menu -> Compiler -> XeLaTeX), шрифт DejaVu Sans (есть в TeX Live).
  python scripts/build_latex.py            # .tex + figures + REPORT_latex.pdf (если установлен xelatex)
"""
import os, re, shutil, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sbershock.config import ROOT

OUT = ROOT / "reports" / "latex"
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "figures").mkdir(parents=True)
md = (ROOT / "REPORT.md").read_text(encoding="utf-8")
md = "\n".join(l for l in md.split("\n") if not l.startswith("**Конкурс СберИндекса 2026") and l != "**Команда: Timofey Usenko MISIS**")
md = md.replace("reports/figures/", "figures/")
for f in re.findall(r"figures/([\w\-.]+\.png)", md):
    shutil.copy(ROOT / "reports" / "figures" / f, OUT / "figures" / f)
(OUT / "REPORT_src.md").write_text(md, encoding="utf-8")
header = r"""
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\footnotesize\setlength{\tabcolsep}{3pt}}
\usepackage{xcolor}
\definecolor{accent}{HTML}{14523A}
\usepackage{titlesec}
\titleformat{\section}{\Large\bfseries\color{accent}}{\thesection}{0.6em}{}
\titleformat{\subsection}{\large\bfseries\color{accent}}{\thesubsection}{0.6em}{}
\usepackage{float}
\let\origfigure\figure
\let\endorigfigure\endfigure
\renewenvironment{figure}[1][2] {\origfigure[H]} {\endorigfigure}
\setlength{\emergencystretch}{3em}
\usepackage{newunicodechar}
"""
(OUT / "header.tex").write_text(header, encoding="utf-8")
args = ["pandoc", "REPORT_src.md", "-s", "-o", "REPORT.tex", "--pdf-engine=xelatex", "--toc", "--toc-depth=2",
        "--include-in-header=header.tex",
        "-M", "title=Прогноз потребления в МО и раннее обнаружение шоков",
        "-M", "subtitle=Конкурс СберИндекса 2026, трек «Прогнозирование»",
        "-M", "author=Команда: Timofey Usenko MISIS", "-M", "date=",
        "-V", "lang=ru", "-V", "documentclass=article", "-V", "fontsize=10pt", "-V", "geometry:margin=1.8cm",
        "-V", "mainfont=DejaVuSans", "-V", "mainfontoptions=Extension=.ttf, BoldFont=*-Bold, ItalicFont=*-Oblique, BoldItalicFont=*-BoldOblique",
        "-V", "monofont=DejaVuSansMono", "-V", "monofontoptions=Extension=.ttf, Scale=0.85, BoldFont=*-Bold",
        "-V", "colorlinks=true", "-V", "linkcolor=accent", "-V", "urlcolor=accent", "-V", "toccolor=black"]
subprocess.run(args, check=True, cwd=OUT)
tex = (OUT / "REPORT.tex").read_text(encoding="utf-8")
tex = tex.replace("\\usepackage{lmodern}\n", "")
(OUT / "REPORT.tex").write_text(tex, encoding="utf-8")
(OUT / "README_LATEX.md").write_text(
    "# LaTeX-версия отчёта\n\n"
    "Файлы: `REPORT.tex` (основной), `header.tex` (оформление), `figures/` (графики).\n\n"
    "Сборка: XeLaTeX, два прохода (для оглавления):\n\n    xelatex REPORT.tex && xelatex REPORT.tex\n\n"
    "Overleaf: New Project, Upload Project, этот zip; затем Menu, Compiler: **XeLaTeX**, Recompile.\n\n"
    "Таблицы с цифрами в .tex являются снимком результатов. Основной источник отчёта: `reports/REPORT_template.md` "
    "(таблицы подставляются из `results/*.csv` скриптом `scripts/render_report.py`); после пересчёта результатов "
    "LaTeX пересобирается командой `python scripts/build_latex.py`.\n", encoding="utf-8")
if shutil.which("xelatex"):
    for _ in range(2):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "REPORT.tex"], cwd=OUT,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ok = r.returncode == 0
    print("xelatex:", "OK" if ok else "ОШИБКА, см. reports/latex/REPORT.log")
    if ok:
        shutil.copy(OUT / "REPORT.pdf", ROOT / "reports" / "REPORT_latex.pdf")
        for ext in ("aux", "log", "out", "toc", "pdf"):
            p = OUT / f"REPORT.{ext}"
            if p.exists():
                p.unlink()
(OUT / "REPORT_src.md").unlink()
shutil.make_archive(str(ROOT / "reports" / "report_latex"), "zip", OUT)
print("готово:", OUT, "и reports/report_latex.zip")
