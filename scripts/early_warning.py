"""
Раннее предупреждение: P(шок в расходах МО в месяце t+1 | информация на конец месяца t).

Метка: тревога панельного детектора хотя бы в одной из 6 категорий МО в месяце t+1 (порог - 2 % тревог на ряд).
Признаки на конец месяца t (без заглядывания вперёд):
  persistence   - была ли тревога в МО в месяце t; сила отклонения (макс. оценка детектора)
  contagion     - доля МО региона с тревогой в месяце t
  news          - местные/региональные события (по группам), всплеск относительно нормы
  weather       - погодные аномалии месяца t
  static        - население, доступность рынков, доход
Честная схема по времени: обучение на t ≤ 2024-05 (метки до 2024-06), проверка на t = 2024-06 … 2024-11.
Сравнение наборов признаков (абляция): только инерция → + соседи → + погода → + новости.
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sbershock.config import PROCESSED, RESULTS
from sbershock.evaluation.backtest import PanelMatrix
from sbershock.changepoint import detectors as D

NEWS_TAG = os.environ.get("NEWS_TAG", "v1")
pm = PanelMatrix.from_panel(pd.read_parquet(PROCESSED / "panel.parquet"))
comp = np.where(pm.keys.complete.values)[0]; keys = pm.keys.iloc[comp].reset_index(drop=True)
E = np.load(RESULTS / "cp_residuals_clean.npy")
S = D.panel_score(E, keys.cat_idx.values, keys.region_code.values)
thr = np.nanquantile(S[:, 12:], 0.98)
A = (np.nan_to_num(S) > thr).astype(float)

months = pm.months
df_s = pd.DataFrame(np.nan_to_num(S), columns=months); df_s["tid"] = keys.territory_id.values
df_a = pd.DataFrame(A, columns=months); df_a["tid"] = keys.territory_id.values
smax = df_s.groupby("tid").max().stack().rename("score")
amax = df_a.groupby("tid").max().stack().rename("alarm")
P = pd.concat([smax, amax], axis=1).reset_index().rename(columns={"level_1": "month"})
reg = keys.drop_duplicates("territory_id").set_index("territory_id").region_code
P["region"] = P.tid.map(reg)
P["reg_alarm_share"] = P.groupby(["region", "month"]).alarm.transform("mean")
P["reg_alarm_share_other"] = (P.reg_alarm_share * P.groupby(["region", "month"]).alarm.transform("size") - P.alarm) / \
                             (P.groupby(["region", "month"]).alarm.transform("size") - 1).clip(lower=1)
P = P.sort_values(["tid", "month"])
P["y_next"] = P.groupby("tid").alarm.shift(-1)
MODE = os.environ.get("EW_MODE", "forecast")  # forecast: шок в t+1 по данным t; nowcast: шок в t по новостям t и расходам t-1
if MODE == "nowcast":
    # расходы месяца t ещё не опубликованы: инерция и соседи берутся из t-1, новости и погода из t
    for c in ["alarm", "score", "reg_alarm_share_other"]:
        P[c] = P.groupby("tid")[c].shift(1)
    P["y_next"] = P.groupby("tid").alarm.shift(-1).groupby(P.tid).shift(1)
    P["y_next"] = amax.reindex(pd.MultiIndex.from_arrays([P.tid, P.month])).values

ex = pd.read_parquet(PROCESSED / ("exogenous.parquet" if NEWS_TAG == "v1" else f"exogenous_{NEWS_TAG}.parquet"))
news_cols = [c for c in ex.columns if c.startswith("news")]
wcols = ["t2m_c_anom", "precip_mm_day_anom", "snowfall_mm_day_anom", "wind_ms_anom"]
X = P.merge(ex[["territory_id", "month"] + news_cols + wcols].rename(columns={"territory_id": "tid"}), on=["tid", "month"], how="left")
st = pd.read_parquet(PROCESSED / "static_features.parquet").set_index("territory_id")[["log_pop", "market_access", "income_pc_best_2023"]]
X = X.join(st, on="tid")
X = X[(X.month >= "2023-07-01") & (X.month <= "2024-11-01") & X.y_next.notna()]

SETS = {
    "инерция (тревога и сила в t)": ["alarm", "score"],
    "+ соседи по региону": ["alarm", "score", "reg_alarm_share_other"],
    "+ погода": ["alarm", "score", "reg_alarm_share_other"] + wcols,
    "+ новости": ["alarm", "score", "reg_alarm_share_other"] + wcols + news_cols,
    "+ новости + свойства МО": ["alarm", "score", "reg_alarm_share_other"] + wcols + news_cols + list(st.columns),
    "только новости": news_cols,
}
train = X.month <= "2024-05-01"; test = X.month >= "2024-06-01"
import lightgbm as lgb
rows = []
base_rate = X.loc[test, "y_next"].mean()
for name, cols in SETS.items():
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    m.fit(X.loc[train, cols], X.loc[train, "y_next"])
    p = m.predict_proba(X.loc[test, cols])[:, 1]
    y = X.loc[test, "y_next"].values
    top = p >= np.quantile(p, 0.95)  # 5 % самых тревожных МО-месяцев
    rows.append(dict(features=name, ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p),
                     precision_top5=y[top].mean(), recall_top5=y[top].sum() / y.sum(), lift_top5=y[top].mean() / base_rate))
R = pd.DataFrame(rows); R["base_rate"] = base_rate
R["mode"] = MODE
R.to_csv(RESULTS / f"early_warning_{MODE}_{NEWS_TAG}.csv", index=False)
pd.set_option("display.width", 200)
print(f"наблюдений: обучение {train.sum()}, проверка {test.sum()}; доля шоков в проверке {base_rate:.3f}")
print(R.round(3).to_string(index=False))
