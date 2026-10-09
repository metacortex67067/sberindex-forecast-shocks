"""Данные для интерактивного лендинга -> docs/landing_data.json (числа из results/ и data/processed/)."""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from sbershock.config import CATEGORIES, PROCESSED, RESULTS, ROOT
from sbershock.data.loaders import territory_table
from sbershock.evaluation.backtest import PanelMatrix
from sbershock.changepoint import detectors as D

R = lambda x, d=0: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d) if d else int(round(float(x)))
pm = PanelMatrix.from_panel(pd.read_parquet(PROCESSED / "panel.parquet"))
tt = territory_table()
months = [m.strftime("%Y-%m") for m in pm.months]
out = {"months": months, "categories": CATEGORIES}
NAMES = {"final": "Финал (9 STF + Chronos-2 + TimesFM)", "ens_stf9": "Ансамбль 9 STF", "stf": "STF (одна модель)",
         "prophet_default": "Prophet", "prophet_tuned": "Prophet (настроенный)", "naive": "Наивный", "snaive": "Сезонный наивный",
         "snaive_growth": "Сезонный наивный × рост", "chronos2_raw": "Chronos-2 · raw", "chronos2_stf_resid": "Chronos-2 · stf_resid",
         "chronos2_extended": "Chronos-2 · extended", "timesfm25_raw": "TimesFM 2.5 · raw", "timesfm25_stf_resid": "TimesFM 2.5 · stf_resid",
         "timesfm25_extended": "TimesFM 2.5 · extended", "ets_raw": "ETS · raw", "ets_stf_resid": "ETS · stf_resid",
         "theta_raw": "Theta · raw", "theta_stf_resid": "Theta · stf_resid", "arima_raw": "ARIMA · raw", "arima_stf_resid": "ARIMA · stf_resid"}
FAMILY = lambda m: ("final" if m == "final" else "prophet" if m.startswith("prophet") else "stf" if "stf" in m and not any(
    m.startswith(f) for f in ("chronos", "timesfm", "ets", "theta", "arima")) else "fm" if m.startswith(("chronos", "timesfm"))
    else "classic" if m.startswith(("ets", "theta", "arima")) else "naive")

out["lb"] = {}
for mode in ("rolling", "holdout", "h2_2024"):
    lb = pd.read_csv(RESULTS / f"leaderboard_{mode}.csv", index_col=0)
    out["lb"][mode] = [{"id": m, "name": NAMES.get(m, m), "family": FAMILY(m),
                        "mae": [R(r[f"MAE h={h}"]) for h in (1, 3, 6, 12)],
                        "r2yoy": [R(r[f"r2_yoy h={h}"], 2) for h in (1, 3, 6, 12)],
                        "mase": [R(r[f"MASE h={h}"], 2) for h in (1, 3, 6, 12)]} for m, r in lb.iterrows()]
c = pd.read_csv(RESULTS / "final_vs_prophet_all_modes.csv")
out["vsProphet"] = [{"mode": r["mode"], "h": int(r.h), "prophet": R(r.MAE_prophet), "final": R(r.MAE_final), "delta": R(r.delta_pct, 1),
                     "lo": R(r.ci_low), "hi": R(r.ci_high), "win": R(100 * r.win_share, 1), "n": int(r.n_forecasts)} for _, r in c.iterrows()]
A = pd.read_csv(RESULTS / "ablation.csv")
steps = list(A.model.unique())
out["ablation"] = {"steps": steps, **{mode: [[R(A[(A.model == s) & (A["mode"] == mode) & (A.h == h)].MAE.iat[0]) for h in (1, 3, 6, 12)]
                                            for s in steps] for mode in ("rolling", "h2_2024")}}
cat = pd.read_csv(RESULTS / "forecast_by_category.csv")
cat = cat[cat.model.isin(["final", "prophet_default"])].pivot_table(index=["category", "h"], columns="model", values="MAE")
out["byCategory"] = [{"cat": k, "prophet": [R(cat.at[(k, h), "prophet_default"]) for h in (1, 3, 6, 12)],
                      "final": [R(cat.at[(k, h), "final"]) for h in (1, 3, 6, 12)]} for k in CATEGORIES]
d = pd.read_csv(RESULTS / "cp_semisynthetic.csv").sort_values("F1", ascending=False)
out["detectors"] = [{"name": r.detector, "f1": R(r.F1, 3), "recall": R(r.recall, 3), "precision": R(r.precision, 3),
                     "delay": R(r.delay, 2), "far": R(r.false_alarms_per100, 2)} for r in d.itertuples()]
dr = pd.read_csv(RESULTS / "cp_semisynthetic_regional.csv", index_col=0)
for d_ in out["detectors"]:
    d_["recall_regional"] = R(dr.at[d_["name"], "recall"], 3) if d_["name"] in dr.index else None
scd = pd.read_csv(RESULTS / "cp_scorecard.csv", index_col=0)
out["scorecard"] = [{"name": k, "f1": R(r["F1, местные шоки"], 2), "delay": R(r["задержка, мес."], 2),
                     "regional": R(r["полнота, региональные шоки"], 2), "registry": int(r["реестр 2024 (из 8)"]),
                     "news": R(100 * r["новости: тревога в месяц события"], 1), "rank": R(r["средний ранг"], 2)} for k, r in scd.iterrows()]
reg = pd.read_csv(RESULTS / "cp_registry_cases_v2.csv")
out["registry"] = {"events": list(reg.columns[1:]), "rows": [{"det": r.iloc[0], "hits": [v == "✓" for v in r.iloc[1:]]} for _, r in reg.iterrows()]}
ew = []
for mode in ("forecast", "nowcast"):
    f = RESULTS / f"early_warning_{mode}_v2.csv"
    if f.exists():
        for _, r in pd.read_csv(f).iterrows():
            ew.append({"mode": mode, "features": r.features, "auc": R(r.ROC_AUC, 3), "lift": R(r.lift_top5, 2)})
out["earlyWarning"] = ew
na = pd.read_csv(RESULTS / "news_accuracy_v1_v2.csv")
out["newsAcc"] = [{"metric": r["метрика"], "k": int(r["верно"]), "n": int(r["всего"]), "p": R(r["доля"], 3),
                   "lo": R(r["ДИ95_низ"], 3), "hi": R(r["ДИ95_верх"], 3)} for _, r in na.iterrows()]
chk = pd.read_csv(RESULTS / "forecast_2025_national_check.csv")
summ = pd.read_csv(RESULTS / "forecast_2025_national_summary.csv")
out["check2025"] = {
    "monthly": {k: {"nat25": [R(100 * v, 1) for v in g["страна факт 2025 г/г"]], "pred25": [R(100 * v, 1) for v in g["панель прогноз 2025 г/г"]],
                    "act24": [R(100 * v, 1) for v in g["панель факт 2024 г/г"]]} for k, g in chk.groupby("category", sort=False)},
    "summary": [{"cat": r.category, "pred": R(r["прогноз 2025, % г/г"], 2), "fact": R(r["страна факт 2025, % г/г"], 2),
                 "err": R(r["средняя ошибка 2025, п.п."], 2), "gap24": R(r["средний разрыв 2024, п.п."], 1),
                 "bench": R(r["бенчмарк «прирост 2024 без изменений»: средняя ошибка, п.п."], 1), "source": r.source,
                 "note": r["сопоставимость"] if isinstance(r["сопоставимость"], str) else ""} for _, r in summ.iterrows()]}
EX = {2541: "Пресненский район", 2543: "Тверской район", 2544: "Хамовники", 2535: "Арбат", 2536: "Басманный район",
      2537: "Замоскворечье", 2545: "Якиманка", 2486: "Раменки", 2353: "Адмиралтейский округ", 2358: "Васильевский остров",
      2456: "Дворцовый округ", 1597: "Новосибирск", 1971: "Екатеринбург", 354: "Казань", 1523: "Нижний Новгород",
      2275: "Челябинск", 1874: "Самара", 1632: "Омск", 21: "Уфа", 666: "Красноярск", 727: "Пермь", 1007: "Волгоград",
      2190: "Тюмень", 775: "Владивосток", 1134: "Иркутск", 842: "Хабаровск", 1176: "Калининград", 2318: "Ярославль",
      309: "Якутск", 1673: "Орск"}
fin = pd.read_parquet(RESULTS / "bt_final.parquet"); pro = pd.read_parquet(RESULTS / "bt_prophet_default.parquet")
fc = pd.read_parquet(RESULTS / "forecast_2025.parquet")
mi = {m: i for i, m in enumerate(pm.months)}
series = []
for tid in EX:
    cats = {}
    for cname in CATEGORIES:
        k = pm.keys[(pm.keys.territory_id == tid) & (pm.keys.category == cname)]
        if not len(k):
            continue
        i = int(k.index[0])
        ent = {"y": [R(v) for v in pm.Y[i]]}
        for lab, res in (("fin", fin), ("pro", pro)):
            r = res[res.series == i]
            ent[lab] = {str(h): [[mi[t], R(v)] for t, v in zip(g.target, g.y_pred)] for h, g in r.sort_values("target").groupby("h") if h in (1, 3, 6)}
        f = fc[fc.series == i].sort_values("h")
        ent["f25"] = [R(v) for v in f.y_pred]; ent["lo"] = [R(v) for v in f.lo80]; ent["hi"] = [R(v) for v in f.hi80]
        cats[cname] = ent
    series.append({"tid": tid, "name": EX[tid], "region": tt.at[tid, "region_name"], "cats": cats})
out["series"] = series
E = np.load(RESULTS / "cp_residuals_clean.npy")
keys = pm.keys[pm.keys.complete.values].reset_index(drop=True)
S = D.panel_score(E, keys.cat_idx.values, keys.region_code.values)
thr = float(np.nanquantile(S[:, 12:], 0.98))
inc = pd.read_parquet(PROCESSED / "incidents_v2.parquet")
TYPES = {"flood": "паводок", "fire": "пожар", "wildfire": "природный пожар", "emergency_regime": "режим ЧС", "drone_attack": "атака БПЛА",
         "shelling": "обстрел", "evacuation": "эвакуация", "utility_outage": "отключение ЖКУ", "industrial_accident": "авария/взрыв",
         "terror": "теракт", "transport_disruption": "перекрытие транспорта", "weather_extreme": "аномальная погода",
         "epidemic": "эпидемия/отравление", "enterprise_shock": "проблемы предприятия"}
CASES = [(1673, "Орск, 04.2024: паводок"), (1874, "Самара, 01.2024: пожар"), (1007, "Волгоград, 08.2024: теракт"),
         (1971, "Екатеринбург, 09.2024: пожар"), (345, "Владикавказ, 12.2024: пожар")]
cases = {}
for tid, label in CASES:
    m = keys.territory_id.values == tid
    x = inc[inc.territory_id == tid].groupby("month").agg(msgs=("msgs", "sum"), sev=("severity", "max"))
    x = x.reindex(pm.months).fillna(0)
    cases[str(tid)] = {"name": label,
                       "scores": {keys.category.iat[i]: [R(v, 2) for v in S[i]] for i in np.where(m)[0]},
                       "msgs": [int(v) for v in x.msgs], "sev": [int(v) for v in x.sev]}
out["cases"] = cases; out["caseOrder"] = [str(k) for k, _ in CASES]; out["thr"] = round(thr, 2)
al = []
Sdf = pd.DataFrame(S[:, 12:], columns=pm.months[12:])
Sdf["tid"] = keys.territory_id.values; Sdf["cat"] = keys.category.values
lng = Sdf.melt(id_vars=["tid", "cat"], var_name="month", value_name="score").dropna()
lng["month"] = pd.to_datetime(lng.month)
lng = lng[lng.score > thr].sort_values("score", ascending=False).drop_duplicates(["tid", "month"])
mo_inc = inc[inc.territory_id.notna()].astype({"territory_id": int})
# крупное происшествие: тяжесть >= 2, не меньше 5 сообщений, группы «бедствия», «инфраструктура», «безопасность»
big = mo_inc[(mo_inc.severity >= 2) & (mo_inc.msgs >= 5) & mo_inc.group.isin(["disaster", "infra", "security"])]
explained = lng.merge(big[["territory_id", "month"]].drop_duplicates(), left_on=["tid", "month"], right_on=["territory_id", "month"])
for r in explained.itertuples():
    ii = mo_inc[(mo_inc.territory_id == r.tid) & (mo_inc.month == r.month)].sort_values(["severity", "msgs"], ascending=False)
    al.append({"name": tt.at[r.tid, "municipal_district_name"].replace("городской округ ", "").replace("муниципальный район", "р-н"),
               "region": tt.at[r.tid, "region_name"], "month": pd.Timestamp(r.month).strftime("%Y-%m"), "cat": r.cat, "score": R(r.score, 1),
               "incidents": [{"type": TYPES.get(t, t), "sev": int(s), "msgs": int(n)} for t, s, n in zip(ii.event_type, ii.severity, ii.msgs)][:4]})
out["alarms"] = al
out["alarmStats"] = {"alarms2024": int(len(lng)), "withMajorNews": int(len(explained)), "thr": round(thr, 2),
                     "byMonth": [int(v) for v in lng.groupby("month").size().reindex(pm.months[12:]).fillna(0).values]}
ev = pd.concat([pd.read_parquet(f) for f in sorted((PROCESSED / "llm_full").glob("news_llm_events_*.parquet"))], ignore_index=True)
picks = [("tass", 244175), ("mchs", 12236), ("ria", 253553), ("mchs", 14057), ("tass", 194588)]
acc_rows = pd.read_csv(RESULTS / "news_accuracy_rows.csv")
v = pd.read_csv(ROOT / "data" / "validation" / "news_validation_labeled.csv", sep=";", encoding="utf-8-sig").set_index(["channel", "msg_id"])
ex = []
for ch, mid in picks:
    e = ev[(ev.channel == ch) & (ev.msg_id == mid)].drop_duplicates(["event_type", "territory_id", "region_code"])
    vv = v.loc[(ch, mid)]
    ar = acc_rows[(acc_rows.channel == ch) & (acc_rows.msg_id == mid)]
    v2ok = bool((ar.p.min() == 1) and (ar.t.min() == 1)) if len(ar) else bool(vv.type_final == 0)
    ex.append({"channel": ch, "date": str(vv.published_msk)[:10], "text": str(vv.snippet)[:320], "v2ok": v2ok,
               "v1": {"type": vv.event_type, "place": vv.place, "region": vv.region, "ok": bool(vv.place_final == 1 and vv.type_final == 1)},
               "v2": [{"type": TYPES.get(t, t), "sev": int(s), "level": l,
                       "place": (tt.municipal_district_name.get(int(tid)) if pd.notna(tid) else None),
                       "region": tt.drop_duplicates("region_code").set_index("region_code").region_name.get(int(rc)) if pd.notna(rc) else None}
                      for t, s, l, tid, rc in zip(e.event_type, e.severity, e.level, e.territory_id, e.region_code)]})
out["newsExamples"] = ex
from sbershock.features.news_v2 import load_events
out["newsStats"] = {"messages": 355751, "events": int(len(load_events(PROCESSED / "llm_full"))), "incidents": int(len(inc))}
(ROOT / "docs").mkdir(exist_ok=True)
js = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
(ROOT / "docs" / "landing_data.json").write_text(js, encoding="utf-8")
print("landing_data.json:", len(js) // 1024, "КБ; тревог 2024:", len(lng), "с тяжёлыми новостями:", len(explained))
