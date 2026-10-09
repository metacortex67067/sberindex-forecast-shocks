"""
Детекторы на РЕЕСТРЕ реальных событий 2024 г. и гибрид «детектор расходов + новости v2».

Реестр (внешний источник: известные бедствия 2024 г. в МО, которые есть в панели):
  паводки апреля-мая 2024 г. (Орск, Новотроицк, г. Оренбург, Кувандык, г. Курган, Ишим, Абатский р-н)
  и теракт в «Крокус Сити Холле» (Красногорск, 03.2024).
Попадание - тревога детектора по любой категории МО в месяце события t или в t+1.

Гибрид: оценка панельного детектора × (1 + γ·[в МО в месяце t есть происшествие тяжести ≥ 2 по новостям v2]).
Порог любого варианта пересчитывается так, чтобы доля тревог была одинаковой (2 % ряд-месяцев 2024 г.) -
сравнение при равном бюджете тревог, т. е. без «покупки» попаданий лишними тревогами.
Цена гибрида: на полусинтетике (шоки не связаны с новостями) - доля потерянных обнаружений при том же бюджете.
Выход: results/cp_registry_hybrid.csv, results/cp_registry_cases_v2.csv
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from sbershock.config import PROCESSED, RESULTS
from sbershock.evaluation.backtest import PanelMatrix
from sbershock.changepoint import detectors as D

FAR = 0.02
REGISTRY = [(1673, "Орск", "паводок, прорыв дамбы", "2024-04"), (1672, "Новотроицк", "паводок", "2024-04"),
            (1665, "г. Оренбург", "паводок", "2024-04"), (1670, "Кувандык", "паводок", "2024-04"),
            (1333, "г. Курган", "паводок на Тоболе", "2024-04"), (2192, "Ишим", "паводок на Ишиме", "2024-04"),
            (2195, "Абатский р-н", "паводок", "2024-05"), (1459, "Красногорск", "теракт в «Крокус Сити Холле»", "2024-03")]

pm = PanelMatrix.from_panel(pd.read_parquet(PROCESSED / "panel.parquet"))
keys = pm.keys[pm.keys.complete.values].reset_index(drop=True)
E = np.load(RESULTS / "cp_residuals_clean.npy")
_npz = RESULTS / "cp_scores_real.npz"
if _npz.exists():
    base = dict(np.load(_npz))
else:
    from sbershock.config import load_yaml
    base = D.all_scores(E, keys, load_yaml("changepoint.yaml"))
ex = pd.read_parquet(PROCESSED / "exogenous_v2.parquet")
sev = ex.pivot_table(index="territory_id", columns="month", values="news2_mo_max_sev").reindex(columns=pm.months)
I = (sev.reindex(keys.territory_id.values).fillna(0).values >= 2).astype(float)
print(f"доля МО-месяцев 2024 г. с происшествием тяжести ≥ 2: {I[:, 12:].mean():.3f}")


def alarms(score):
    thr = np.nanquantile(score[:, 12:], 1 - FAR)
    A = pd.DataFrame(np.nan_to_num(score) > thr, columns=pm.months).groupby(keys.territory_id.values).max()
    return A, thr


def registry_hits(score):
    A, _ = alarms(score)
    out = []
    for tid, name, what, m in REGISTRY:
        t = pd.Timestamp(m + "-01")
        hit = bool(A.at[tid, t] or A.at[tid, t + pd.DateOffset(months=1)]) if tid in A.index else None
        out.append(hit)
    return out


def registry_hits_mo(score, budget):
    """Равный бюджет на уровне МО В КАЖДОМ МЕСЯЦЕ: оценка МО-месяца = максимум по категориям, в каждом месяце 2024 г.
    тревога у доли budget МО с наибольшей оценкой (порог по срезу месяца - работает в реальном времени)."""
    M = pd.DataFrame(score, columns=pm.months).groupby(keys.territory_id.values).max()
    A = pd.DataFrame(False, index=M.index, columns=M.columns)
    for m in M.columns[12:]:
        A[m] = M[m].fillna(-np.inf) > np.nanquantile(M[m].values, 1 - budget)
    return [bool(A.at[tid, pd.Timestamp(m + "-01")] or A.at[tid, pd.Timestamp(m + "-01") + pd.DateOffset(months=1)]) for tid, _, _, m in REGISTRY]


variants = {k: v for k, v in base.items()}
for g in (0.25, 0.5, 1.0):
    variants[f"panel+news_v2 (γ={g})"] = base["panel"] * (1 + g * I)
variants["bocpd+news_v2 (γ=0.5)"] = base["bocpd"] * (1 + 0.5 * I)  # γ взят у панельного гибрида, не подбирался
BUDGET_MO = float(alarms(base["panel"])[0].iloc[:, 12:].values.mean())
print(f"равный бюджет на уровне МО: {BUDGET_MO:.3f}")
rows, cases = [], []
for name, sc in variants.items():
    h = registry_hits(sc)
    A, thr = alarms(sc)
    share_news = (np.nan_to_num(sc[:, 12:]) > thr)[I[:, 12:] > 0].sum() / max((np.nan_to_num(sc[:, 12:]) > thr).sum(), 1)
    rows.append({"детектор": name, "реестр: попаданий": sum(bool(x) for x in h), "реестр: всего": len(h),
                 "реестр при равном бюджете МО по месяцам": sum(registry_hits_mo(sc, BUDGET_MO)),
                 "доля тревог в МО-месяцах с тяжёлым происшествием": round(float(share_news), 3)})
    cases.append([name] + ["✓" if x else "-" for x in registry_hits_mo(sc, BUDGET_MO)])

# у вставленных шоков нет новостей, поэтому при том же бюджете гибрид тратит часть тревог на МО-месяцы
# с новостями; цена гибрида = доля тревог панельного детектора на реальных данных, вытесненных гибридом
Ap, _ = alarms(base["panel"])
for g in (0.25, 0.5, 1.0):
    Ah, _ = alarms(variants[f"panel+news_v2 (γ={g})"])
    lost = (Ap.values[:, 12:] & ~Ah.values[:, 12:]).sum() / Ap.values[:, 12:].sum()
    for r in rows:
        if r["детектор"] == f"panel+news_v2 (γ={g})":
            r["тревог панельного детектора вытеснено"] = round(float(lost), 3)
res = pd.DataFrame(rows)
res.to_csv(RESULTS / "cp_registry_hybrid.csv", index=False)
cases = pd.DataFrame(cases, columns=["детектор"] + [f"{n} {m}" for _, n, _, m in REGISTRY])
cases.to_csv(RESULTS / "cp_registry_cases_v2.csv", index=False)
pd.set_option("display.width", 250)
print(res.to_string(index=False)); print(cases.to_string(index=False))
print("\nтяжесть происшествий (v2) в МО реестра в месяц события:")
for tid, name, what, m in REGISTRY:
    print(f"  {name:14s} {m}: max_sev={sev.at[tid, pd.Timestamp(m + '-01')] if tid in sev.index else None}")

A24 = alarms(base["panel"])[0].iloc[:, 12:]
sev24 = sev.reindex(index=A24.index, columns=A24.columns).fillna(0)
pop = pd.read_parquet(PROCESSED / "static_features.parquet").set_index("territory_id").population.reindex(A24.index)
q = pd.qcut(pop.rank(method="first"), 4, labels=["Q1 (малые)", "Q2", "Q3", "Q4 (крупные)"])
rows = []
for lab in list(q.cat.categories) + ["все"]:
    m = np.ones(len(q), bool) if lab == "все" else (q == lab).values
    a, s = A24.values[m].ravel(), (sev24.values[m] >= 2).ravel()
    rows.append({"квартиль населения": lab, "МО-месяцев с тяжёлым происшествием": int(s.sum()),
                 "доля тревог при происшествии": a[s].mean(), "доля тревог без": a[~s].mean(), "lift": a[s].mean() / a[~s].mean()})
lift = pd.DataFrame(rows)
lift.to_csv(RESULTS / "news_severe_incident_lift.csv", index=False)
print("\nтяжёлое происшествие -> тревога в расходах (lift по квартилям населения):\n", lift.round(3).to_string(index=False))

def alarms_monthly(score):
    A = np.zeros(score.shape, bool)
    for t in range(12, score.shape[1]):
        A[:, t] = np.nan_to_num(score[:, t]) > np.nanquantile(score[:, t], 1 - FAR)
    return pd.DataFrame(A, columns=pm.months).groupby(keys.territory_id.values).max()
rows = []
for name, A in [("общий порог", alarms(base["panel"])[0]), ("порог по месяцу", alarms_monthly(base["panel"]))]:
    h = sum(bool(A.at[tid, pd.Timestamp(m + "-01")] or A.at[tid, pd.Timestamp(m + "-01") + pd.DateOffset(months=1)]) for tid, _, _, m in REGISTRY)
    by = A.iloc[:, 12:].sum(0).values
    rows.append({"вариант": name, "реестр: попаданий из 8": h, "тревог МО в январе 2024": int(by[0]),
                 "тревог МО в месяц: мин": int(by.min()), "макс": int(by.max())})
mt = pd.DataFrame(rows); mt.to_csv(RESULTS / "cp_month_threshold.csv", index=False)
print("\nпорог детектора:\n", mt.to_string(index=False))


# сводный ранг по 4 проверкам: F1 на местных шоках, полнота на региональных при 2 ложных тревогах на 100,
# попадания в реестр 2024 г. и доля новостных МО-месяцев с тревогой (равный бюджет тревог на уровне МО)
loc = pd.read_csv(RESULTS / "cp_semisynthetic.csv", index_col=0)
regn = pd.read_csv(RESULTS / "cp_semisynthetic_regional.csv", index_col=0)
rev = pd.read_csv(RESULTS / "cp_realevents.csv"); rev = rev[rev.budget == "mo_month"].set_index("detector")
regmo = pd.read_csv(RESULTS / "cp_registry_hybrid.csv").set_index("детектор")["реестр при равном бюджете МО по месяцам"]
dets = [d for d in loc.index if d in regn.index and d in rev.index and d in regmo.index]
sc = pd.DataFrame({"F1, местные шоки": loc.loc[dets, "F1"], "задержка, мес.": loc.loc[dets, "delay"],
                   "полнота, региональные шоки": regn.loc[dets, "recall"],
                   "реестр 2024 (из 8)": regmo.loc[dets].astype(int),
                   "новости: тревога в месяц события": rev.loc[dets, "hit_t+0"]})
main = ["F1, местные шоки", "полнота, региональные шоки", "реестр 2024 (из 8)", "новости: тревога в месяц события"]
sc["средний ранг"] = sc[main].rank(ascending=False, method="average").mean(axis=1)
sc = sc.sort_values("средний ранг")
sc.to_csv(RESULTS / "cp_scorecard.csv")
print("\nсводная таблица детекторов:\n", sc.round(3).to_string())
