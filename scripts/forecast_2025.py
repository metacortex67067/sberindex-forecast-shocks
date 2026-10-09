"""
Прогноз потребления МО на 2025 год финальной моделью (T = 2024-12, h = 1…12) + интервалы + проверка.

1. Члены ансамбля: 9 вариантов STF (затухание × окно роста, configs/models.yaml → ensemble), Chronos-2 и
   TimesFM 2.5 extended (results/fc2025_<backend>_extended.parquet; если файлов нет - ансамбль из 9 STF).
   Строго по времени: национальный ряд - по 2024-12, недельные категории - по 31.12.2024.
2. Интервалы: эмпирические квантили ошибки финальной модели в бэктесте, log(y/ŷ), по горизонту и категории
   (h = 1, 3, 6, 12 - из бэктеста, промежуточные h - линейная интерполяция). Покрытие проверяется
   «вперёд по времени»: квантили по ранним точкам прогноза -> доля попаданий на поздних.
3. Проверка на фактических данных 2025 г.: на уровне МО фактов за 2025 г. нет, поэтому прогноз агрегируется
   (взвешивание по населению 2024 г., полные ряды) и его годовой прирост сравнивается с фактическим приростом
   национальных рядов СберИндекса (consumer-spending - месячный, до 2026-08; недельные категории - для
   маркетплейсов, здоровья, транспорта). Базовая линия сопоставимости - тот же разрыв «панель - страна» в 2024 г.
Выход: results/forecast_2025.parquet, results/forecast_2025_intervals_coverage.csv,
       results/forecast_2025_national_check.csv, results/forecast_2025_national_summary.csv
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from sbershock.config import CATEGORIES, PROCESSED, RESULTS, load_yaml
from sbershock.data import loaders as L
from sbershock.evaluation.backtest import PanelMatrix
from sbershock.models.stf import STF, WEEKLY_MAP

HS = list(range(1, 13))
QS = {"lo80": 0.10, "hi80": 0.90, "lo90": 0.05, "hi90": 0.95}
NAT = {"Все категории": "Всего", "Продовольствие": "Продовольственные товары", "Общественное питание": "Общественное питание"}

pm = PanelMatrix.from_panel(pd.read_parquet(PROCESSED / "panel.parquet"))
ti = len(pm.months) - 1
T = pm.months[ti]
nat_full, wk_full = L.load_national_spending(), L.load_weekly_categories()
ctx = {"national": nat_full[nat_full.index <= T], "weekly": wk_full[wk_full.index <= T + pd.offsets.MonthEnd(0)]}

cfg, base = load_yaml("models.yaml")["ensemble"], load_yaml("models.yaml")["stf"]
base = {k: v for k, v in base.items() if k in ("lam_region", "lam_mo", "k_level")}
logs, members = [], []
for d in cfg["damp"]:
    for kg in cfg["k_growth"]:
        m = STF(**base, k_growth=kg, damp=d, anchor_years=cfg["anchor_years"], lam_growth_region=cfg["lam_growth_region"],
                damp_direct_only=cfg.get("damp_direct_only", False), name=f"stf_d{d}_kg{kg}")
        logs.append(np.log(m.predict(pm, ti, 12, ctx))); members.append(m.name)
for fm in cfg["foundation_members"]:
    f = RESULTS / f"fc2025_{fm}.parquet"
    if f.exists():
        x = pd.read_parquet(f).pivot(index="series", columns="h", values="y_pred").reindex(range(len(pm.keys)))[HS].values
        logs.append(np.log(x)); members.append(fm)
    else:
        print(f"!! нет {f.name}, в прогноз 2025 г. войдут только члены STF")
P = np.exp(np.nanmean(np.stack(logs), axis=0))
model_name = "final" if len(members) == 11 else "ens_stf9"
print(f"членов: {len(members)} -> модель {model_name}")

bt = pd.read_parquet(RESULTS / f"bt_{model_name}.parquet")
bt = bt[bt.complete & bt.y_true.notna()].copy()
bt["r"] = np.log(bt.y_true / bt.y_pred)


def quantiles(b):
    q = b.groupby(["category", "h"]).r.quantile(list(QS.values())).unstack()
    q.columns = list(QS.keys())
    return q


def interp_h(q):
    """квантили на сетке h = 1, 3, 6, 12 -> все h = 1…12 (линейно)"""
    out = []
    for c, g in q.groupby(level=0):
        g = g.droplevel(0)
        gi = g.reindex(HS).interpolate(method="index", limit_direction="both")
        gi["category"] = c
        out.append(gi.rename_axis("h").reset_index())
    return pd.concat(out, ignore_index=True)


Q = interp_h(quantiles(bt))
# проверка покрытия вперёд по времени: квантили по ранней половине точек, покрытие на поздней
cov = []
for h, g in bt.groupby("h"):
    orig = sorted(g.origin.unique())
    if len(orig) < 2:
        continue
    early, late = orig[: len(orig) // 2], orig[len(orig) // 2:]
    q = quantiles(g[g.origin.isin(early)])
    gl = g[g.origin.isin(late)].join(q, on=["category", "h"])
    cov.append({"h": h, "точек калибровки": len(early), "точек проверки": len(late),
                "покрытие 80 % (номинал 0,80)": ((gl.r >= gl.lo80) & (gl.r <= gl.hi80)).mean(),
                "покрытие 90 % (номинал 0,90)": ((gl.r >= gl.lo90) & (gl.r <= gl.hi90)).mean()})
cov = pd.DataFrame(cov)
cov.to_csv(RESULTS / "forecast_2025_intervals_coverage.csv", index=False)
print(cov.round(3).to_string(index=False))

rows = []
for h in HS:
    rows.append(pd.DataFrame({"series": np.arange(len(pm.keys)), "h": h, "target": T + pd.DateOffset(months=h),
                              "y_pred": P[:, h - 1]}))
fc = pd.concat(rows, ignore_index=True).join(pm.keys[["territory_id", "category", "region_code", "complete"]], on="series")
fc = fc.merge(Q, on=["category", "h"], how="left")
for k in QS:
    fc[k] = fc.y_pred * np.exp(fc[k])
fc["model"] = model_name
fc["members"] = ",".join(members)
fc.to_parquet(RESULTS / "forecast_2025.parquet", index=False)

pop = pd.read_parquet(PROCESSED / "static_features.parquet").set_index("territory_id").population
k = pm.keys.assign(pop=pm.keys.territory_id.map(pop).values)
ok = k.complete.values & np.isfinite(k["pop"].values)
y24 = pm.Y[:, 12:24]; y23 = pm.Y[:, 0:12]
wk_month = wk_full.copy(); wk_month.index = wk_month.index.to_period("M").to_timestamp()
wk_month = wk_month.groupby(level=0).mean()


def agg(Y, cat):
    m = ok & (k.category.values == cat)
    return (Y[m] * k["pop"].values[m, None]).sum(0)


def national_yoy(cat, year):
    months = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
    if cat in NAT:
        n = nat_full[NAT[cat]]
        return (n.reindex(months).values / n.reindex(months - pd.DateOffset(years=1)).values - 1), "consumer-spending (месячный)"
    w = wk_month[WEEKLY_MAP[cat]].mean(axis=1).reindex(months).values / 100
    return w, "недельные категории (среднее % г/г за месяц)"


rows = []
for cat in CATEGORIES:
    pred25 = agg(P, cat) / agg(y24, cat) - 1
    act24 = agg(y24, cat) / agg(y23, cat) - 1
    n25, src = national_yoy(cat, 2025)
    n24, _ = national_yoy(cat, 2024)
    for i in range(12):
        rows.append({"category": cat, "month": i + 1, "source": src,
                     "панель факт 2024 г/г": act24[i], "страна 2024 г/г": n24[i],
                     "панель прогноз 2025 г/г": pred25[i], "страна факт 2025 г/г": n25[i]})
chk = pd.DataFrame(rows)
chk["ошибка 2025 при приросте 2024 г. без изменений"] = chk["панель факт 2024 г/г"] - chk["страна факт 2025 г/г"]
chk["разрыв 2024 (панель минус страна)"] = chk["панель факт 2024 г/г"] - chk["страна 2024 г/г"]
chk["ошибка 2025 (прогноз минус страна)"] = chk["панель прогноз 2025 г/г"] - chk["страна факт 2025 г/г"]
chk.to_csv(RESULTS / "forecast_2025_national_check.csv", index=False)
summ = chk.groupby("category", sort=False).agg(
    **{"панель факт 2024, % г/г": ("панель факт 2024 г/г", "mean"), "страна 2024, % г/г": ("страна 2024 г/г", "mean"),
       "прогноз 2025, % г/г": ("панель прогноз 2025 г/г", "mean"), "страна факт 2025, % г/г": ("страна факт 2025 г/г", "mean"),
       "средний разрыв 2024, п.п.": ("разрыв 2024 (панель минус страна)", "mean"),
       "средняя ошибка 2025, п.п.": ("ошибка 2025 (прогноз минус страна)", "mean"),
       "MAE прироста 2025 по месяцам, п.п.": ("ошибка 2025 (прогноз минус страна)", lambda s: s.abs().mean()),
       "бенчмарк «прирост 2024 без изменений»: средняя ошибка, п.п.": ("ошибка 2025 при приросте 2024 г. без изменений", "mean"),
       "бенчмарк: MAE по месяцам, п.п.": ("ошибка 2025 при приросте 2024 г. без изменений", lambda s: s.abs().mean())})
summ.iloc[:, :] = summ.values * 100
summ["source"] = chk.groupby("category", sort=False).source.first()
# методологический разрыв национального ряда: январь/декабрь 2025 против медианы 2019-2024
brk = {}
for cat, col in NAT.items():
    n = nat_full[col]
    r = {y: n.get(pd.Timestamp(f"{y}-01-01")) / n.get(pd.Timestamp(f"{y - 1}-12-01")) for y in range(2019, 2026)}
    med = np.median([r[y] for y in range(2019, 2025)])
    brk[cat] = f"разрыв ряда: январь/декабрь 2025 = {r[2025]:.3f} против медианы {med:.3f}" if abs(r[2025] / med - 1) > 0.1 else ""
summ["сопоставимость"] = summ.index.map(lambda c: brk.get(c, ""))
summ.to_csv(RESULTS / "forecast_2025_national_summary.csv")
pd.set_option("display.width", 250)
print(summ.round(1).to_string())
