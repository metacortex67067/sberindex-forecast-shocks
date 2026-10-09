"""Графики для отчёта и лендинга -> reports/figures/*.png (данные из results/ и data/processed/)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sbershock.config import CATEGORIES, PROCESSED, RESULTS, ROOT
from sbershock.evaluation.backtest import PanelMatrix
from sbershock.changepoint import detectors as D

OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130, "savefig.bbox": "tight", "axes.unicode_minus": False})
C_FINAL, C_PROPHET, C_GREY = "#1f6f50", "#c0392b", "#8c8c8c"
NAMES = {"final": "Финал (9 STF + Chronos-2 + TimesFM)", "ens_stf9": "Ансамбль 9 STF", "prophet_default": "Prophet",
         "naive": "Наивный", "snaive_growth": "Сезонный наивный × рост", "chronos2_extended": "Chronos-2 (extended)",
         "timesfm25_extended": "TimesFM 2.5 (extended)", "ets_stf_resid": "ETS (stf_resid)", "theta_stf_resid": "Theta (stf_resid)",
         "arima_stf_resid": "ARIMA (stf_resid)", "ets_raw": "ETS", "theta_raw": "Theta", "arima_raw": "ARIMA"}
pm = PanelMatrix.from_panel(pd.read_parquet(PROCESSED / "panel.parquet"))


def save(fig, name):
    fig.savefig(OUT / name); plt.close(fig); print("  ", name)


for mode in ("rolling", "holdout", "h2_2024"):
    f = RESULTS / f"leaderboard_{mode}.csv"
    if not f.exists():
        continue
    lb = pd.read_csv(f, index_col=0)
    models = [m for m in ["prophet_default", "naive", "snaive_growth", "ets_raw", "theta_raw", "arima_raw",
                          "theta_stf_resid", "ets_stf_resid", "arima_stf_resid", "chronos2_extended",
                          "timesfm25_extended", "ens_stf9", "final"] if m in lb.index]
    hs = [1, 3, 6, 12]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        col = C_FINAL if m == "final" else C_PROPHET if m.startswith("prophet") else plt.cm.tab20(i % 20)
        ax.bar(np.arange(4) + i * w - 0.4 + w / 2, [lb.at[m, f"MAE h={h}"] for h in hs], w, label=NAMES.get(m, m), color=col,
               edgecolor="black" if m == "final" else "none", linewidth=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"h = {h}" for h in hs])
    ax.set_ylabel("MAE, руб./жителя в месяц")
    ttl = {"rolling": "скользящая точка прогноза 2023-12…2024-11", "holdout": "одна точка T = 2024-12 - h",
           "h2_2024": "целевые месяцы 2024-07…2024-12"}[mode]
    ax.set_title(f"Ошибка прогноза по горизонтам: {ttl} (12 096 рядов)")
    ax.set_ylim(0, min(ax.get_ylim()[1], 2000))
    ax.legend(ncol=3, fontsize=7.5, frameon=False, loc="upper left")
    save(fig, f"mae_by_h_{mode}.png")

A = pd.read_csv(RESULTS / "ablation.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, mode in zip(axes, ["rolling", "h2_2024"]):
    x = A[A["mode"] == mode].pivot(index="model", columns="h", values="MAE")
    for h, mk in zip([1, 3, 6, 12], ["o", "s", "^", "D"]):
        ax.plot(range(len(x)), x[h].values, marker=mk, label=f"h = {h}")
    ax.set_xticks(range(len(x))); ax.set_xticklabels([s.split(".")[0] for s in x.index])
    ax.set_title({"rolling": "Абляция: скользящий бэктест", "h2_2024": "Абляция: целевые месяцы 2024-07…12"}[mode])
    ax.set_xlabel("шаг (0: Prophet, 6: финал)")
axes[0].set_ylabel("MAE, руб./жителя в месяц"); axes[0].legend(frameon=False)
save(fig, "ablation.png")

lb = pd.read_csv(RESULTS / "leaderboard_rolling.csv", index_col=0)
fams = [("chronos2", "Chronos-2"), ("timesfm25", "TimesFM 2.5"), ("ets", "ETS"), ("theta", "Theta"), ("arima", "ARIMA")]
modes = [("raw", "raw: ряд как есть"), ("stf_resid", "stf_resid: без сезонности STF"), ("extended", "extended: + синт. предыстория")]
fams = [f for f in fams if any(f"{f[0]}_{m}" in lb.index for m, _ in modes)]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, h in zip(axes, [1, 12]):
    for j, (m, lab) in enumerate(modes):
        vals = [lb.at[f"{f}_{m}", f"MAE h={h}"] if f"{f}_{m}" in lb.index else np.nan for f, _ in fams]
        ax.bar(np.arange(len(fams)) + (j - 1) * 0.27, vals, 0.27, label=lab)
    ax.axhline(lb.at["final", f"MAE h={h}"], color=C_FINAL, ls="--", lw=1.5, label="финал")
    ax.axhline(lb.at["prophet_default", f"MAE h={h}"], color=C_PROPHET, ls=":", lw=1.5, label="Prophet")
    ax.set_xticks(range(len(fams))); ax.set_xticklabels([n for _, n in fams]); ax.set_title(f"Режимы подачи данных, h = {h}")
axes[0].set_ylabel("MAE (скользящий бэктест)")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
save(fig, "fm_modes.png")

cp = pd.read_csv(RESULTS / "cp_semisynthetic.csv", index_col=0)
cr = pd.read_csv(RESULTS / "cp_semisynthetic_regional.csv", index_col=0)
order = pd.read_csv(RESULTS / "cp_scorecard.csv", index_col=0).index.tolist() if (RESULTS / "cp_scorecard.csv").exists() else cp.index.tolist()
order = [d for d in order if d in cp.index][::-1]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
y = np.arange(len(order))
axes[0].barh(y - 0.2, cp.loc[order, "F1"], 0.4, label="F1", color=C_FINAL)
axes[0].barh(y + 0.2, cp.loc[order, "recall"], 0.4, label="полнота", color="#5dade2")
axes[0].set_title("Местные шоки (шок в одном ряду)", fontsize=10); axes[0].legend(frameon=False, loc="lower right")
axes[1].barh(y, cr.loc[order, "recall"], 0.6, color="#c49a3a")
axes[1].set_title("Региональные шоки: полнота", fontsize=10)
axes[0].set_yticks(y); axes[0].set_yticklabels(order)
for ax in axes:
    ax.set_xlim(0, 1)
fig.suptitle("Детекторы на полусинтетике при 2 ложных тревогах на 100 ряд-мес. (сверху лучший по среднему рангу)", fontsize=10)
save(fig, "detectors.png")

E = np.load(RESULTS / "cp_residuals_clean.npy")
keys = pm.keys[pm.keys.complete.values].reset_index(drop=True)
S = D.panel_score(E, keys.cat_idx.values, keys.region_code.values)
thr = np.nanquantile(S[:, 12:], 0.98)
inc = pd.read_parquet(PROCESSED / "incidents_v2.parquet")
for tid, nm in [(1673, "Орск"), (2192, "Ишим")]:
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    m = keys.territory_id.values == tid
    for i in np.where(m)[0]:
        axes[0].plot(pm.months, S[i], marker=".", label=keys.category.iat[i])
    axes[0].axhline(thr, color="red", ls="--", lw=1, label=f"порог ({thr:.1f})")
    axes[0].set_ylabel("оценка панельного детектора"); axes[0].legend(ncol=4, fontsize=7.5, frameon=False)
    axes[0].set_title(f"{nm}: детектор расходов и новости v2")
    x = inc[inc.territory_id == tid].groupby("month").agg(n=("msgs", "size"), msgs=("msgs", "sum"), sev=("severity", "max"))
    x = x.reindex(pm.months).fillna(0)
    axes[1].bar(x.index, x.msgs, width=20, color=["#c0392b" if s >= 3 else "#f39c12" if s >= 2 else "#95a5a6" for s in x.sev])
    axes[1].set_ylabel("сообщений о\nпроисшествиях"); axes[1].set_yscale("symlog")
    save(fig, f"case_{tid}.png")

fin = pd.read_parquet(RESULTS / "bt_final.parquet"); pro = pd.read_parquet(RESULTS / "bt_prophet_default.parquet")
fc25 = pd.read_parquet(RESULTS / "forecast_2025.parquet") if (RESULTS / "forecast_2025.parquet").exists() else None
ex = [(1673, "Все категории", "Орск, все категории"), (1673, "Маркетплейсы", "Орск, маркетплейсы"),
      (2192, "Продовольствие", "Ишим, продовольствие")]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, (tid, cat, title) in zip(axes, ex):
    i = int(pm.keys[(pm.keys.territory_id == tid) & (pm.keys.category == cat)].index[0])
    ax.plot(pm.months, pm.Y[i], color="black", lw=1.6, label="факт")
    for res, col, lab in [(fin, C_FINAL, "финал"), (pro, C_PROPHET, "Prophet")]:
        r = res[(res.series == i)]
        r1 = r[r.origin == "2023-12-01"].sort_values("target")
        ax.plot(r1.target, r1.y_pred, color=col, ls="--", marker="o", ms=3, label=f"{lab}, T = 2023-12 (h = 1, 3, 6, 12)")
    if fc25 is not None:
        f = fc25[fc25.series == i].sort_values("target")
        ax.plot(f.target, f.y_pred, color=C_FINAL, lw=1.4, label="прогноз 2025")
        ax.fill_between(f.target, f.lo80, f.hi80, color=C_FINAL, alpha=0.18, label="80 %-интервал")
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%y"))
axes[0].set_ylabel("руб./жителя в месяц"); axes[0].legend(fontsize=7, frameon=False)
save(fig, "examples.png")

na = pd.read_csv(RESULTS / "news_accuracy_v1_v2.csv")
sel = ["v1: место (все 200)", "v2: место", "v1: тип (все 200)", "v2: тип (строго)", "v1: оба (все 200)", "v2: оба (строго)"]
x = na.set_index("метрика").loc[sel]
fig, ax = plt.subplots(figsize=(7.5, 3.4))
cols = ["#95a5a6", C_FINAL] * 3
ax.bar(range(6), x["доля"], color=cols, yerr=[x["доля"] - x["ДИ95_низ"], x["ДИ95_верх"] - x["доля"]], capsize=4)
ax.set_xticks(range(6)); ax.set_xticklabels(["место\nv1", "место\nv2", "тип\nv1", "тип\nv2", "оба\nv1", "оба\nv2"])
ax.set_ylim(0.5, 1.0); ax.set_ylabel("доля верных (95 % ДИ)")
ax.set_title("Новости: v1 (правила + Natasha) против v2 (Qwen2.5-7B + выбор МО)", fontsize=10)
for i, v in enumerate(x["доля"]):
    ax.text(i, v + 0.035, f"{100 * v:.0f} %", ha="center", fontsize=9)
save(fig, "news_accuracy.png")

f = RESULTS / "forecast_2025_national_check.csv"
if f.exists():
    chk = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    c = chk[chk.category == "Все категории"]
    axes[0].plot(c.month, 100 * c["страна факт 2025 г/г"], "k-o", ms=4, label="СберИндекс, факт 2025 (страна)")
    axes[0].plot(c.month, 100 * c["панель прогноз 2025 г/г"], "-o", color=C_FINAL, ms=4, label="прогноз 2025 (агрегат МО)")
    axes[0].plot(c.month, 100 * c["панель факт 2024 г/г"], ":", color=C_GREY, label="бенчмарк: прирост 2024 без изменений")
    axes[0].set_xticks(range(1, 13)); axes[0].set_xlabel("месяц 2025 г."); axes[0].set_ylabel("% г/г")
    axes[0].set_title("Все категории: прирост расходов 2025 г."); axes[0].legend(fontsize=8, frameon=False)
    s = pd.read_csv(RESULTS / "forecast_2025_national_summary.csv", index_col=0)
    s = s[s["сопоставимость"].isna() | (s["сопоставимость"] == "")]
    xx = np.arange(len(s))
    axes[1].bar(xx - 0.2, s["прогноз 2025, % г/г"], 0.4, color=C_FINAL, label="прогноз 2025")
    axes[1].bar(xx + 0.2, s["страна факт 2025, % г/г"], 0.4, color="#34495e", label="факт 2025 (СберИндекс)")
    axes[1].set_xticks(xx); axes[1].set_xticklabels([t.replace(" ", "\n") for t in s.index], fontsize=8)
    axes[1].set_title("Средний прирост 2025 г. по категориям, % г/г"); axes[1].legend(fontsize=8, frameon=False)
    save(fig, "check_2025.png")

rows = []
for mode in ("forecast", "nowcast"):
    for tag in ("v1", "v2"):
        f = RESULTS / f"early_warning_{mode}_{tag}.csv"
        if f.exists():
            d = pd.read_csv(f); d["tag"] = tag; d["mode"] = mode; rows.append(d)
if rows:
    ew = pd.concat(rows)
    ew = ew[ew.features.isin(["инерция (тревога и сила в t)", "только новости", "+ новости"])]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    piv = ew.pivot_table(index=["mode", "tag"], columns="features", values="ROC_AUC")
    piv.plot.bar(ax=ax, rot=0, color=["#5dade2", C_FINAL, "#f5b041"])
    ax.axhline(0.5, color="black", lw=0.8); ax.set_ylim(0.45, 0.7); ax.set_ylabel("ROC-AUC (проверка 2024-06…11)")
    ax.set_title("Раннее предупреждение: новости почти не предсказывают шоки расходов", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.set_xlabel("")
    save(fig, "early_warning.png")

from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(13, 6.4)); ax.set_xlim(0, 13); ax.set_ylim(0, 6.6); ax.axis("off"); ax.grid(False)
def box(x, y, w, h, text, fc, fs=7.8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12", fc=fc, ec="#34495e", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", linespacing=1.35)
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#34495e", lw=1.2))
D1, D2, M1, OUT_ = "#eaf2f8", "#fef5e7", "#e8f6f3", "#fdedec"
for x, t in [(1.55, "ДАННЫЕ"), (5.2, "МОДЕЛИ"), (8.35, "АНСАМБЛЬ / ДЕТЕКЦИЯ"), (11.45, "РЕЗУЛЬТАТ")]:
    ax.text(x, 6.35, t, ha="center", weight="bold")
box(0.1, 5.0, 2.9, 1.1, "СберИндекс: расходы МО\n2 016 МО × 6 категорий × 24 мес.\n(склейка объединённых МО)", D1)
box(0.1, 3.7, 2.9, 1.1, "СберИндекс: национальные ряды\n(месячные с 2018 г.)\nи недельные категории (% г/г)", D1)
box(0.1, 2.4, 2.9, 1.1, "Росстат, ЦБ, ERA5: население (веса),\nдоходы, ИПЦ, ставка, погода\n(проверены как признаки,\nв финал не вошли)", D1)
box(0.1, 0.5, 2.9, 1.6, "Новости: Telegram ТАСС, РИА,\nИнтерфакс, МЧС (2022-2024)\nQwen2.5-7B: происшествие?\nтип, тяжесть 0-3, место → МО", D2)
box(3.6, 4.4, 3.2, 1.7, "STF, перенос сезонности:\nуровень + затухающий рост +\nпрофиль «категория → регион → МО»;\n9 вариантов (φ × окно роста)", M1)
box(3.6, 2.6, 3.2, 1.5, "Foundation-модели:\nChronos-2, TimesFM 2.5\nрежим extended (длинный\nконтекст по нац. ряду)", M1)
box(3.6, 0.5, 3.2, 1.6, "Детекторы шоков (14 вариантов)\nпо остатку прогноза STF на 1 мес.:\nлучший: BOCPD; панельный\n(МО против региона) и региональный", M1)
box(7.3, 3.4, 2.1, 1.7, "Ансамбль:\nсреднее логарифмов\n11 членов,\nравные веса", M1, bold=True)
box(7.3, 0.5, 2.1, 1.6, "Порог по месяцу\n(10,5 % МО) +\nобъяснение тревоги\nновостями (тип,\nтяжесть, источники)", M1)
box(9.9, 4.4, 3.0, 1.7, "Прогноз МО на 1, 3, 6, 12 мес.\n+ интервалы по ошибкам бэктеста;\nпрогноз на 2025 г. с проверкой\nпо факту СберИндекса", OUT_)
box(9.9, 2.6, 3.0, 1.5, "Сравнение с Prophet, ETS, Theta,\nARIMA, наивными:\nMAE, R², MASE, бутстреп-ДИ", OUT_)
box(9.9, 0.5, 3.0, 1.6, "Тревоги «МО × месяц»\n+ объяснение новостями\n(тип, тяжесть, число сообщений)", OUT_)
arrow(3.0, 5.55, 3.6, 5.3); arrow(3.0, 4.25, 3.6, 5.0); arrow(3.0, 4.1, 3.6, 3.4)
arrow(6.8, 5.2, 7.3, 4.6); arrow(6.8, 3.35, 7.3, 3.9)
arrow(9.4, 4.5, 9.9, 5.2); arrow(9.4, 4.0, 9.9, 3.35)
arrow(5.2, 4.4, 5.2, 4.12)
ax.annotate("", xy=(3.6, 1.3), xytext=(1.55, 5.0), arrowprops=dict(arrowstyle="-|>", color="#34495e", lw=1.0, connectionstyle="angle3,angleA=0,angleB=90", alpha=0.0))
arrow(6.8, 1.3, 7.3, 1.3); arrow(9.4, 1.3, 9.9, 1.3)
ax.annotate("", xy=(7.3, 0.75), xytext=(3.0, 0.75), arrowprops=dict(arrowstyle="-|>", color="#b9770e", lw=1.3))
ax.text(5.2, 0.25, "новости → объяснение тревог (и проверенный гибрид)", ha="center", fontsize=7.5, color="#b9770e")
save(fig, "architecture.png")
print("готово:", OUT)
