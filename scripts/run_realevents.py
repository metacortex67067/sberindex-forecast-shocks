"""
Детекторы на реальных данных 2024 г. Независимая разметка - новости v1 (детекторы расходов их НЕ используют):
событие = МО-месяц 2024 г. с ≥ N местных новостей о бедствиях/безопасности/инфраструктуре и всплеском ≥ 2
(в e² раз выше нормы). Метрика: доля таких МО-месяцев с тревогой детектора (по любой категории) в месяц события
и в окне t-1 … t+2 против базовой доли тревог; lift = во сколько раз чаще.

Два способа задать бюджет тревог:
  series - порог на 2 % ряд-месяцев 2024 г. (как в полусинтетике). У детекторов, которые дают одну оценку на
           весь МО (многомерный), доля МО-месяцев с тревогой при этом меньше, чем у порядовых (панельный ≈ 10 %).
  mo     - РАВНЫЙ бюджет на уровне МО: оценка МО-месяца = максимум по категориям, порог - так, чтобы доля
           МО-месяцев с тревогой была одинаковой у всех детекторов (как у панельного при 2 % ряд-месяцев).
  mo_month - то же, но доля одинакова в КАЖДОМ месяце (основной режим: порог по срезу месяца работает в реальном
           времени и не даёт детектору выигрывать за счёт месяцев, где у него вообще много тревог - например,
           PELT/BinSeg с дискретной оценкой дают почти все тревоги в мае и октябре).
Все 11 детекторов считаются на всех рядах. Оценки сохраняются в results/cp_scores_real.npz (для hybrid_registry.py).
Выход: results/cp_realevents.csv, results/cp_realevents_list.csv
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from sbershock.config import PROCESSED, RESULTS, load_yaml
from sbershock.evaluation.backtest import PanelMatrix
from sbershock.changepoint import detectors as D

MIN_NEWS, MIN_SURGE, FAR = 10, 2.0, 0.02
cfg = load_yaml("changepoint.yaml")
p = pd.read_parquet(PROCESSED / "panel.parquet"); pm = PanelMatrix.from_panel(p)
keys = pm.keys[pm.keys.complete.values].reset_index(drop=True)
E = np.load(RESULTS / "cp_residuals_clean.npy")
t0 = time.time()
S = D.all_scores(E, keys, cfg)
print(f"детекторы на всех {len(keys)} рядах: {time.time() - t0:.0f} c")
np.savez_compressed(RESULTS / "cp_scores_real.npz", **S)

ex = pd.read_parquet(PROCESSED / "exogenous.parquet")
ex["news_local"] = ex.news_mo_security + ex.news_mo_disaster + ex.news_mo_infra
base = ex.groupby("territory_id").news_local.transform(lambda s: s.shift(1).rolling(12, min_periods=3).mean())
ex["surge"] = np.log1p(ex.news_local) - np.log1p(base)
evs = ex[(ex.month >= "2024-01-01") & (ex.month <= "2024-11-01") & (ex.news_local >= MIN_NEWS) & (ex.surge >= MIN_SURGE)
         & ex.territory_id.isin(keys.territory_id)][["territory_id", "month", "news_local"]]
print(f"событий (МО-месяцев) по новостям: {len(evs)}, МО: {evs.territory_id.nunique()}")


def mo_alarms(sc, budget, per_month=False):
    """budget='series' - 2 % ряд-месяцев за 2024 г.; число - доля МО-месяцев с тревогой (равный бюджет на уровне МО);
    per_month=True - та же доля в КАЖДОМ месяце (порог по срезу месяца: работает в реальном времени и не даёт
    детектору «выиграть» за счёт месяцев, где у него вообще много тревог)."""
    if budget == "series":
        thr = np.nanquantile(sc[:, 12:], 1 - FAR)
        A = pd.DataFrame(np.nan_to_num(sc) > thr).groupby(keys.territory_id.values).max()
        A.columns = pm.months
        return A
    M = pd.DataFrame(sc, columns=pm.months).groupby(keys.territory_id.values).max()
    if per_month:
        A = pd.DataFrame(False, index=M.index, columns=M.columns)
        for m in M.columns[12:]:
            A[m] = M[m].fillna(-np.inf) > np.nanquantile(M[m].values, 1 - budget)
        return A
    thr = np.nanquantile(M.iloc[:, 12:].values, 1 - budget)
    return M.fillna(-np.inf) > thr


budget_mo = float(mo_alarms(S["panel"], "series").iloc[:, 12:].values.mean())
rows = []
for bname, budget, pmn in [("series", "series", False), ("mo", budget_mo, False), ("mo_month", budget_mo, True)]:
    for name, sc in S.items():
        A = mo_alarms(sc, budget, pmn)
        base_rate = A.iloc[:, 12:].values.mean()
        r = {"budget": bname, "detector": name, "base_rate_MO": base_rate}
        for lag in (-1, 0, 1, 2):
            hits = [A.at[e.territory_id, e.month + pd.DateOffset(months=lag)] for e in evs.itertuples()
                    if e.month + pd.DateOffset(months=lag) <= pm.months[-1]]
            r[f"hit_t{lag:+d}"] = np.mean(hits)
        win = [A.loc[e.territory_id, [m for m in pm.months if e.month - pd.DateOffset(months=1) <= m <= e.month + pd.DateOffset(months=2)]].any()
               for e in evs.itertuples()]
        r["hit_window"] = np.mean(win)
        r["lift_t0"] = r["hit_t+0"] / base_rate
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(RESULTS / "cp_realevents.csv", index=False)
evs.to_csv(RESULTS / "cp_realevents_list.csv", index=False)
pd.set_option("display.width", 220)
print(f"равный бюджет на уровне МО: {budget_mo:.3f} МО-месяцев с тревогой")
print(res.round(3).to_string(index=False))
