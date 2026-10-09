"""
Сборка методологического отчёта: reports/REPORT_template.md + таблицы из results/ -> REPORT.md, reports/REPORT.html, reports/REPORT.pdf
Все числа в таблицах отчёта берутся из файлов results/*.csv, метки вида {{T_ИМЯ}} и {{V_ИМЯ}}.
PDF собирается через pandoc (HTML) и Chromium (печать в PDF); без них собирается только REPORT.md.
"""
import os, re, shutil, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from sbershock.config import RESULTS, ROOT

NAMES = {"final": "**Финал: 9 STF + Chronos-2 + TimesFM**", "ens_stf9": "Ансамбль 9 STF", "stf": "STF (одна модель)",
         "prophet_default": "Prophet (по умолчанию)", "prophet_tuned": "Prophet (настроенный)", "naive": "Наивный (последнее значение)",
         "snaive": "Сезонный наивный", "snaive_growth": "Сезонный наивный × рост",
         "chronos2_raw": "Chronos-2, raw", "chronos2_stf_resid": "Chronos-2, stf_resid", "chronos2_extended": "Chronos-2, extended",
         "timesfm25_raw": "TimesFM 2.5, raw", "timesfm25_stf_resid": "TimesFM 2.5, stf_resid", "timesfm25_extended": "TimesFM 2.5, extended",
         "ets_raw": "ETS, raw", "ets_stf_resid": "ETS, stf_resid", "theta_raw": "Theta, raw", "theta_stf_resid": "Theta, stf_resid",
         "arima_raw": "ARIMA, raw", "arima_stf_resid": "ARIMA, stf_resid"}
MODE_RU = {"rolling": "скользящий (все точки)", "holdout": "одна точка T = 2024-12 - h", "h2_2024": "цели 2024-07…12"}


def num(x, d=0):
    if pd.isna(x):
        return "-"
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


def t_leaderboard(mode):
    lb = pd.read_csv(RESULTS / f"leaderboard_{mode}.csv", index_col=0)
    lb = lb.sort_values("MAE h=1")
    rows = []
    for m, r in lb.iterrows():
        rows.append([NAMES.get(m, m)] + [num(r[f"MAE h={h}"]) for h in (1, 3, 6, 12)] +
                    [num(r.get(f"r2_yoy h={h}"), 2) for h in (1, 12)])
    return md(pd.DataFrame(rows, columns=["модель", "MAE h=1", "MAE h=3", "MAE h=6", "MAE h=12", "R² г/г h=1", "R² г/г h=12"]))


def t_vs_prophet():
    c = pd.read_csv(RESULTS / "final_vs_prophet_all_modes.csv")
    rows = [[MODE_RU[r["mode"]], int(r.h), num(r.n_forecasts), num(r.MAE_prophet), f"**{num(r.MAE_final)}**",
             num(r.delta_pct, 1) + " %", f"{num(r.ci_low)} … {num(r.ci_high)}", num(100 * r.win_share, 0) + " %"] for _, r in c.iterrows()]
    return md(pd.DataFrame(rows, columns=["режим оценки", "h", "прогнозов", "MAE Prophet", "MAE финал", "Δ MAE",
                                          "95 % ДИ Δ, руб.", "рядов, где финал точнее"]))


def t_metrics_final(mode="rolling"):
    lb = pd.read_csv(RESULTS / f"leaderboard_{mode}.csv", index_col=0)
    rows = []
    for m in ["prophet_default", "final"]:
        for met, nm, d in [("MAE", "MAE, руб.", 0), ("r2_level", "R² уровня", 4), ("r2_yoy", "R² годового прироста", 2),
                           ("MAPE", "MAPE, %", 1), ("MASE", "MASE", 2)]:
            rows.append([NAMES[m].replace("**", ""), nm] + [num(lb.at[m, f"{met} h={h}"], d) for h in (1, 3, 6, 12)])
    return md(pd.DataFrame(rows, columns=["модель", "метрика", "h=1", "h=3", "h=6", "h=12"]))


def t_ablation():
    A = pd.read_csv(RESULTS / "ablation.csv")
    rows = []
    for m in A.model.unique():
        a = A[A.model == m]
        r = [m]
        for mode in ("rolling", "h2_2024"):
            x = a[a["mode"] == mode].set_index("h").MAE
            r += [num(x.get(h)) for h in (1, 3, 6, 12)]
        rows.append(r)
    return md(pd.DataFrame(rows, columns=["шаг", "h=1", "h=3", "h=6", "h=12", "H2: h=1", "H2: h=3", "H2: h=6", "H2: h=12"]))


def t_category():
    c = pd.read_csv(RESULTS / "forecast_by_category.csv")
    c = c[c.model.isin(["final", "prophet_default"])].pivot_table(index=["category", "h"], columns="model", values="MAE")
    rows = []
    for cat in c.index.get_level_values(0).unique():
        r = [cat]
        for h in (1, 3, 6, 12):
            p, f = c.at[(cat, h), "prophet_default"], c.at[(cat, h), "final"]
            r.append(f"{num(p)} → **{num(f)}** ({num(100 * (f / p - 1), 0)} %)")
        rows.append(r)
    return md(pd.DataFrame(rows, columns=["категория", "h=1", "h=3", "h=6", "h=12"]))


def t_modes():
    lb = pd.read_csv(RESULTS / "leaderboard_rolling.csv", index_col=0)
    rows = []
    for fam, nm in [("chronos2", "Chronos-2"), ("timesfm25", "TimesFM 2.5"), ("ets", "ETS"), ("theta", "Theta"), ("arima", "ARIMA")]:
        for mode in ("raw", "stf_resid", "extended"):
            k = f"{fam}_{mode}"
            if k in lb.index:
                rows.append([nm, mode] + [num(lb.at[k, f"MAE h={h}"]) for h in (1, 3, 6, 12)])
    return md(pd.DataFrame(rows, columns=["модель", "режим", "h=1", "h=3", "h=6", "h=12"]))


def t_detectors(fname="cp_semisynthetic.csv", sort="F1"):
    d = pd.read_csv(RESULTS / fname).sort_values(sort, ascending=False)
    rows = [[r.detector, num(r.F1, 2), num(r.recall, 2), num(r.precision, 2), num(r.delay, 2), num(r.false_alarms_per100, 1),
             num(r.recall_step, 2), num(r.recall_dip, 2), num(r.recall_ramp, 2), num(r.recall_spike, 2)] for r in d.itertuples()]
    return md(pd.DataFrame(rows, columns=["детектор", "F1", "полнота", "точность", "задержка, мес.", "ложн. тревог / 100",
                                          "полнота: сдвиг", "провал", "рост-рампа", "всплеск"]))


def t_scorecard():
    d = pd.read_csv(RESULTS / "cp_scorecard.csv", index_col=0)
    rows = [[k, num(r["F1, местные шоки"], 2), num(r["задержка, мес."], 2), num(r["полнота, региональные шоки"], 2),
             f"{int(r['реестр 2024 (из 8)'])} из 8", num(100 * r["новости: тревога в месяц события"], 1) + " %",
             num(r["средний ранг"], 2)] for k, r in d.iterrows()]
    return md(pd.DataFrame(rows, columns=["детектор", "F1: местные шоки", "задержка, мес.", "полнота: региональные шоки",
                                          "реестр 2024", "новости: тревога в месяц события", "средний ранг (1-4)"]))


def t_registry():
    a = pd.read_csv(RESULTS / "cp_registry_hybrid.csv")
    b = pd.read_csv(RESULTS / "cp_registry_cases_v2.csv")
    t = a.merge(b, on="детектор")
    t["реестр"] = t["реестр при равном бюджете МО по месяцам"].astype(str) + " из 8"
    t = t.sort_values("реестр", ascending=False, kind="stable")
    cols = ["детектор", "реестр"] + list(b.columns[1:])
    return md(t[cols])


def t_realevents():
    d = pd.read_csv(RESULTS / "cp_realevents.csv")
    d = d[d.budget == "mo_month"].sort_values("hit_t+0", ascending=False)
    rows = [[r.detector, num(100 * r.base_rate_MO, 1) + " %", num(100 * r["hit_t+0"], 1) + " %", num(r.lift_t0, 2),
             num(100 * r.hit_window, 1) + " %"] for _, r in d.iterrows()]
    return md(pd.DataFrame(rows, columns=["детектор", "доля МО-месяцев с тревогой", "тревога в месяц события",
                                          "lift", "тревога в окне t-1…t+2"]))


def t_ew():
    rows = []
    for mode in ("forecast", "nowcast"):
        for tag in ("v1", "v2"):
            f = RESULTS / f"early_warning_{mode}_{tag}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            for _, r in d.iterrows():
                rows.append([{"forecast": "прогноз на t+1", "nowcast": "наукастинг t"}[mode], tag, r.features,
                             num(r.ROC_AUC, 3), num(r.PR_AUC, 3), num(r.lift_top5, 2)])
    return md(pd.DataFrame(rows, columns=["режим", "новости", "признаки", "ROC-AUC", "PR-AUC", "lift топ-5 %"]))


def t_news_acc():
    d = pd.read_csv(RESULTS / "news_accuracy_v1_v2.csv")
    rows = [[r["метрика"], f"{r['верно']} / {r['всего']}", num(100 * r["доля"], 1) + " %",
             f"{num(100 * r['ДИ95_низ'], 1)} … {num(100 * r['ДИ95_верх'], 1)} %"] for _, r in d.iterrows()]
    return md(pd.DataFrame(rows, columns=["показатель", "верно / всего", "доля", "95 % ДИ (Уилсон)"]))


def t_2025():
    s = pd.read_csv(RESULTS / "forecast_2025_national_summary.csv")
    rows = []
    for _, r in s.iterrows():
        note = r["сопоставимость"] if isinstance(r["сопоставимость"], str) and r["сопоставимость"] else ""
        rows.append([r.category, num(r["прогноз 2025, % г/г"], 1) + " %", num(r["страна факт 2025, % г/г"], 1) + " %",
                     ("не сопоставимо" if note else num(r["средняя ошибка 2025, п.п."], 1)),
                     ("-" if note else num(r["MAE прироста 2025 по месяцам, п.п."], 1)),
                     num(r["средний разрыв 2024, п.п."], 1),
                     ("-" if note else num(r["бенчмарк: MAE по месяцам, п.п."], 1)),
                     r.source + (f"; {note}" if note else "")])
    return md(pd.DataFrame(rows, columns=["категория", "прогноз 2025, г/г", "факт 2025 (СберИндекс), г/г", "ошибка, п.п.",
                                          "MAE по месяцам, п.п.", "разрыв 2024 между панелью и страной, п.п.",
                                          "бенчмарк «прирост 2024», MAE п.п.", "источник факта"]))


def t_coverage():
    d = pd.read_csv(RESULTS / "forecast_2025_intervals_coverage.csv")
    rows = [[int(r.h), int(r["точек калибровки"]), int(r["точек проверки"]), num(100 * r["покрытие 80 % (номинал 0,80)"], 1) + " %",
             num(100 * r["покрытие 90 % (номинал 0,90)"], 1) + " %"] for _, r in d.iterrows()]
    return md(pd.DataFrame(rows, columns=["h", "точек калибровки", "точек проверки", "покрытие 80 %-интервала", "покрытие 90 %-интервала"]))


def t_shock():
    d = pd.read_csv(RESULTS / "shock_month_mae.csv")
    rows = []
    for (m, s), g in d.groupby(["model", "shock"], sort=False):
        g = g.set_index("h")
        rows.append([m, "месяц шока" if s else "обычный месяц"] +
                    [f"{num(g.at[h, 'MAE'])} ({num(g.at[h, 'MAPE'], 1)} %)" for h in (1, 3, 6, 12)])
    return md(pd.DataFrame(rows, columns=["модель", "месяцы", "h=1: MAE (MAPE)", "h=3", "h=6", "h=12"]))


def t_members_2025():
    f = RESULTS / "forecast_2025.parquet"
    m = pd.read_parquet(f, columns=["members"]).members.iat[0].split(",")
    return ", ".join(m) + f" ({len(m)} членов)"


def v_2025(cat, col, d=1, signed=False):
    s = pd.read_csv(RESULTS / "forecast_2025_national_summary.csv").set_index("category")
    x = float(s.at[cat, col])
    return (("+" if x > 0 else "") if signed else "") + num(abs(x) if not signed else x, d)


def v_cov(h, col):
    d = pd.read_csv(RESULTS / "forecast_2025_intervals_coverage.csv").set_index("h")
    return num(100 * d.at[h, col], 0) + " %"


def v_classic_note():
    """Чувствительность классических моделей к недельным категориям (results/sensitivity/)."""
    def mae(f):
        r = pd.read_parquet(f); r = r[r.complete & r.y_true.notna()]
        return r.assign(ae=(r.y_true - r.y_pred).abs()).groupby("h").ae.mean()
    parts = []
    for k, nm in (("theta", "Theta"), ("ets", "ETS")):
        a, b = RESULTS / f"bt_{k}_stf_resid.parquet", RESULTS / "sensitivity" / f"bt_{k}_stf_resid.parquet"
        if a.exists() and b.exists():
            x, y = mae(a), mae(b)
            parts.append(f"{nm} без них {' / '.join(num(x[h]) for h in (1, 3, 6, 12))}, с ними {' / '.join(num(y[h]) for h in (1, 3, 6, 12))}")
    if not parts:
        return ""
    return ("**Оговорка о входных данных.** Классические модели и foundation-модели в режимах raw и stf_resid рассчитаны без "
            "недельных категорий СберИндекса, а STF и режим extended с ними. Недельные категории влияют только на точку прогноза "
            "2023-12, в которой у панели ещё нет года истории для оценки роста, то есть в основном на h = 12. Проверка "
            "чувствительности, MAE при h = 1 / 3 / 6 / 12: " + "; ".join(parts) +
            ". Вывод о том, что классические модели уступают ансамблю STF, от этого не меняется.")


def v_max_gap():
    gaps = []
    for mode in ("rolling", "holdout", "h2_2024"):
        lb = pd.read_csv(RESULTS / f"leaderboard_{mode}.csv", index_col=0)
        for h in (1, 3, 6, 12):
            c = f"MAE h={h}"
            gaps.append(100 * (lb.at["final", c] / lb[c].min() - 1))
    return num(max(gaps), 1) + " %"


VALUES = {"V_TOT_ERR": lambda: v_2025("Все категории", "средняя ошибка 2025, п.п.", 2, signed=True),
          "V_TOT_PRED": lambda: v_2025("Все категории", "прогноз 2025, % г/г", 2),
          "V_TOT_FACT": lambda: v_2025("Все категории", "страна факт 2025, % г/г", 2),
          "V_TOT_BENCH_MAE": lambda: v_2025("Все категории", "бенчмарк: MAE по месяцам, п.п."),
          "V_TOT_MAE": lambda: v_2025("Все категории", "MAE прироста 2025 по месяцам, п.п."),
          "V_MAX_GAP": v_max_gap,
          "V_CLASSIC_NOTE": lambda: v_classic_note(),
          "V_TOT_BENCH": lambda: v_2025("Все категории", "бенчмарк «прирост 2024 без изменений»: средняя ошибка, п.п."),
          "V_FOOD_ERR": lambda: v_2025("Продовольствие", "средняя ошибка 2025, п.п.", signed=True),
          "V_FOOD_GAP": lambda: v_2025("Продовольствие", "средний разрыв 2024, п.п.", signed=True),
          "V_MP_PRED": lambda: v_2025("Маркетплейсы", "прогноз 2025, % г/г", 0),
          "V_MP_FACT": lambda: v_2025("Маркетплейсы", "страна факт 2025, % г/г", 0),
          "V_MP_BENCH": lambda: v_2025("Маркетплейсы", "бенчмарк «прирост 2024 без изменений»: средняя ошибка, п.п.", 0, signed=True),
          "V_COV1_80": lambda: v_cov(1, "покрытие 80 % (номинал 0,80)"), "V_COV6_80": lambda: v_cov(6, "покрытие 80 % (номинал 0,80)")}

def t_fm_leak():
    d = pd.read_csv(RESULTS / "fm_leak_compare.csv")
    d["модель"] = d["модель"].map(lambda m: NAMES.get(m, m).replace("**", ""))
    for c in [c for c in d.columns if c.startswith("MAE")]:
        d[c] = d[c].map(lambda x: num(x))
    return md(d)


def t_summary():
    c = pd.read_csv(RESULTS / "final_vs_prophet_all_modes.csv")
    rows = []
    for mode, nm in [("rolling", "скользящая точка прогноза (основной)"), ("holdout", "одна точка T = 2024-12 - h"), ("h2_2024", "целевые месяцы 2024-07…2024-12")]:
        x = c[c["mode"] == mode].set_index("h")
        rows.append([nm] + [(f"**{num(x.at[h, 'delta_pct'], 0)} %**" if mode == "rolling" else f"{num(x.at[h, 'delta_pct'], 0)} %") for h in (1, 3, 6, 12)])
    return md(pd.DataFrame(rows, columns=["режим оценки", "h = 1", "h = 3", "h = 6", "h = 12"]))


def t_vs_prophet_tuned():
    f = RESULTS / "final_vs_prophet_tuned_all_modes.csv"
    c = pd.read_csv(f)
    rows = [[MODE_RU[r["mode"]], int(r.h), num(r.MAE_prophet_tuned), f"**{num(r.MAE_final)}**", num(r.delta_pct, 1) + " %",
             f"{num(r.ci_low)} … {num(r.ci_high)}", num(100 * r.win_share, 0) + " %"] for _, r in c.iterrows()]
    return md(pd.DataFrame(rows, columns=["режим оценки", "h", "MAE Prophet (настроенный)", "MAE финал", "Δ MAE",
                                          "95 % ДИ Δ, руб.", "рядов, где финал точнее"]))


TABLES = {"T_DETECTORS_REGIONAL": lambda: t_detectors("cp_semisynthetic_regional.csv", "recall"), "T_SCORECARD": t_scorecard,
          "T_VS_PROPHET_TUNED": t_vs_prophet_tuned, "T_SUMMARY": t_summary, "T_FM_LEAK": t_fm_leak, "T_LB_ROLLING": lambda: t_leaderboard("rolling"), "T_LB_HOLDOUT": lambda: t_leaderboard("holdout"),
          "T_LB_H2": lambda: t_leaderboard("h2_2024"), "T_VS_PROPHET": t_vs_prophet, "T_METRICS_FINAL": t_metrics_final,
          "T_ABLATION": t_ablation, "T_CATEGORY": t_category, "T_MODES": t_modes, "T_DETECTORS": t_detectors,
          "T_REGISTRY": t_registry, "T_REALEVENTS": t_realevents, "T_EW": t_ew, "T_NEWS_ACC": t_news_acc,
          "T_2025": t_2025, "T_SHOCK": t_shock, "T_COVERAGE": t_coverage, "T_MEMBERS_2025": t_members_2025}

if __name__ == "__main__":
    tpl = (ROOT / "reports" / "REPORT_template.md").read_text(encoding="utf-8")
    def sub(m):
        k = m.group(1)
        try:
            return TABLES[k]()
        except Exception as e:
            print(f"!! таблица {k}: {e}")
            return f"*(таблица {k} недоступна: {e})*"
    out = re.sub(r"\{\{(T_[A-Z0-9_]+)\}\}", sub, tpl)
    out = re.sub(r"\{\{(V_[A-Z0-9_]+)\}\}", lambda m: VALUES[m.group(1)](), out)
    out = out.replace("\u2014", "-").replace("\u2013", "-").replace("\u2212", "-")
    (ROOT / "REPORT.md").write_text(out, encoding="utf-8")
    print("REPORT.md:", len(out), "символов")
    if shutil.which("pandoc"):
        css = ROOT / "reports" / "report.css"
        html = ROOT / "reports" / "REPORT.html"
        head = "**Конкурс СберИндекса 2026"
        team = "**Команда: Timofey Usenko MISIS**"
        body = "\n".join(l for l in out.split("\n") if not l.startswith(head) and l != team)
        src = ROOT / "reports" / "_report_pdf.md"
        src.write_text(body, encoding="utf-8")
        subprocess.run(["pandoc", str(src), "-o", str(html), "--standalone", "--toc", "--toc-depth=2",
                        "--metadata", "title=Прогноз потребления в МО и раннее обнаружение шоков",
                        "--metadata", "subtitle=Конкурс СберИндекса 2026, трек «Прогнозирование»",
                        "--metadata", "author=Команда: Timofey Usenko MISIS",
                        "--css", "report.css", "--resource-path", str(ROOT), "--embed-resources"],
                       check=True, cwd=ROOT / "reports")
        src.unlink()
        print("HTML:", html)
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                b = p.chromium.launch()
                pg = b.new_page()
                pg.goto(html.as_uri()); pg.wait_for_timeout(500)
                pg.pdf(path=str(ROOT / "reports" / "REPORT.pdf"), format="A4", print_background=True,
                       margin={"top": "14mm", "bottom": "14mm", "left": "12mm", "right": "12mm"},
                       display_header_footer=True, header_template="<span></span>",
                       footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#888'><span class='pageNumber'></span> / <span class='totalPages'></span></div>")
                b.close()
            print("PDF:", ROOT / "reports" / "REPORT.pdf")
        except Exception as e:
            print("!! PDF не собран:", e)
