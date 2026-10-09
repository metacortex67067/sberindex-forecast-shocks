"""
STF + LightGBM: глобальная модель градиентного бустинга исправляет ошибки STF.

Простыми словами: STF даёт «структурный» прогноз (уровень + рост + сезонность). LightGBM, обученный
сразу на всех МО, учится предсказывать, НАСКОЛЬКО STF ошибётся в конкретном МО - по признакам:
недавние ошибки STF по этому ряду, импульс расходов, доходы и размер МО, цены в регионе, погода, новости.

Цель:   r = log(y[T'+h]) - log(STF_{T'}[h])      (относительная поправка)
Вес:    уровень ряда (так ошибка в рублях, т.е. MAE, учитывается правильно)
Без утечки: для точки прогноза T обучаемся только на парах (T', h) с T'+h ≤ T.
Переобучение раз в квартал (2023-12, 2024-03, 2024-06, 2024-09), между ними модель переиспользуется.
Горизонт 12 при T = 2023-12 обучить нельзя (нет ни одного известного исхода) -> поправка 0, прогноз = STF.
"""
import numpy as np
import pandas as pd

from ..config import CATEGORIES

STATIC_COLS = ["log_pop", "market_access", "income_pc_best_2023", "reg_income_pc_2023", "reg_spending_pc_2023",
               "lat", "lon", "is_city", "is_federal_city_district", "is_regional_center", "is_zato"]
LGB_PARAMS = dict(objective="l1", learning_rate=0.05, num_leaves=31, min_child_samples=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  n_estimators=400, verbose=-1, seed=0)


class STFLGBM:
    def __init__(self, base, static: pd.DataFrame, exog: pd.DataFrame, first_origin_idx=5,
                 retrain_idx=(11, 14, 17, 20), use_exog=True, params=None, train_frac=0.5, name=None):
        self.base, self.static, self.exog = base, static, exog
        self.t0, self.retrain = first_origin_idx, sorted(retrain_idx)
        self.use_exog, self.params, self.train_frac = use_exog, {**LGB_PARAMS, **(params or {})}, train_frac
        self.name = name or ("stf_lgbm" if use_exog else "stf_lgbm_noexog")
        self._cache, self._models = {}, {}

    def _base_fc(self, pm, ctx):
        if "F" not in self._cache:
            F = {}
            for t in range(self.t0 - 1, len(pm.months) - 1):
                F[t] = self.base.predict(pm, t, 12, ctx)
            self._cache["F"] = F
        return self._cache["F"]

    def _base_feats(self, pm, F, t):
        """Признаки точки t, не зависящие от горизонта: float32-матрица n_series × k (кэшируется)."""
        key = ("bf", t)
        if key in self._cache:
            return self._cache[key]
        Y, keys = pm.Y, pm.keys
        n = len(Y)
        logy = np.log(Y)
        yt = logy[:, t]
        nan = np.full(n, np.nan)
        cols = {
            "cat": keys.cat_idx.values, "region": keys.region_code.values, "log_y": yt,
            "mom1": yt - logy[:, t - 1], "mom3": yt - logy[:, t - 3] if t >= 3 else nan,
            "yoy": yt - logy[:, t - 12] if t >= 12 else nan,
            "res1": yt - np.log(F[t - 1][:, 0]) if (t - 1) in F else nan, "hist": np.full(n, t + 1),
        }
        r_prev = [logy[:, t - j] - np.log(F[t - j - 1][:, 0]) for j in range(3) if (t - j - 1) in F]
        cols["res1_m3"] = np.nanmean(np.vstack(r_prev), 0) if r_prev else nan
        g = pd.Series(cols["mom1"]).groupby([cols["cat"], cols["region"]]).transform("median").values
        cols["mom1_rel"] = cols["mom1"] - g
        tids = keys.territory_id.values
        st = self.static.reindex(tids)[STATIC_COLS]
        for c in STATIC_COLS:
            cols[c] = st[c].values
        if self.use_exog:
            ex = self._exog_at(pm, t).reindex(tids)
            for c in ex.columns:
                cols[c] = ex[c].values
        names = list(cols)
        M = np.column_stack([np.asarray(cols[c], dtype=np.float32) for c in names])
        self._cache[key] = (M, names)
        return M, names

    def _rows(self, pm, F, t, hs, series=None):
        """Строки (ряд × горизонт) для точки t: numpy float32 + имена признаков + индексы."""
        M, names = self._base_feats(pm, F, t)
        idx = np.arange(len(M)) if series is None else series
        yt = np.log(pm.Y[idx, t])
        blocks, sers, hh = [], [], []
        for h in hs:
            extra = np.column_stack([np.full(len(idx), h), np.full(len(idx), (pm.months[t].month - 1 + h) % 12 + 1),
                                     np.log(F[t][idx, h - 1]) - yt]).astype(np.float32)
            blocks.append(np.hstack([M[idx], extra]))
            sers.append(idx); hh.append(np.full(len(idx), h))
        return np.vstack(blocks), names + ["h", "moy_target", "stf_growth"], np.concatenate(sers), np.concatenate(hh)

    def _exog_at(self, pm, t):
        key = ("ex", t)
        if key not in self._cache:
            m = pm.months[t]
            e = self.exog[self.exog.month == m].set_index("territory_id").drop(columns="month")
            e3 = self.exog[(self.exog.month > m - pd.DateOffset(months=3)) & (self.exog.month <= m)]
            ncols = [c for c in e.columns if c.startswith("news_") and not c.endswith("_surge")]
            s3 = e3.groupby("territory_id")[ncols].sum().add_suffix("_3m")
            self._cache[key] = e.join(s3)
        return self._cache[key]

    def _train(self, pm, F, T_idx):
        import lightgbm as lgb
        rng = np.random.default_rng(0)
        Xs, ys, ws, names = [], [], [], None
        for t in range(max(self.t0, T_idx - 12), T_idx):
            hs = [h for h in range(1, 12) if t + h <= T_idx]
            if not hs:
                continue
            ser = np.where(rng.random(len(pm.Y)) < self.train_frac)[0]
            X, names, sr, hh = self._rows(pm, F, t, hs, ser)
            y_true = pm.Y[sr, t + hh]
            pred = F[t][sr, hh - 1]
            ok = np.isfinite(y_true) & np.isfinite(pred) & np.isfinite(X[:, names.index("log_y")])
            r = np.log(y_true[ok] / pred[ok])
            # общий для категории эффект месяца (медиана по МО в той же точке и горизонте) вычитается:
            # модель учит только индивидуальную поправку МО, которая переносится на другие месяцы
            key = pd.Series(r).groupby([X[ok][:, names.index("cat")], hh[ok]]).transform("median").values
            Xs.append(X[ok]); ys.append(r - key); ws.append(y_true[ok])
        if not Xs:
            return None
        X, y, w = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
        m = lgb.LGBMRegressor(**self.params)
        m.fit(X, y, sample_weight=w, feature_name=names, categorical_feature=["cat", "region"])
        m.feats_, m.max_h_, m.n_rows_ = names, int(X[:, names.index("h")].max()), len(X)
        return m

    def predict(self, pm, ti, h_max, ctx):
        F = self._base_fc(pm, ctx)
        T_train = max([r for r in self.retrain if r <= ti], default=None)
        P = F[ti][:, :h_max].copy()
        if T_train is None:
            return P
        if T_train not in self._models:
            self._models[T_train] = self._train(pm, F, T_train)
            for k in [k for k in self._cache if isinstance(k, tuple) and k[0] == "bf" and k[1] < T_train - 13]:
                del self._cache[k]
        m = self._models[T_train]
        if m is None:
            return P
        hs = list(range(1, min(h_max, m.max_h_) + 1))
        X, _, sr, hh = self._rows(pm, F, ti, hs)
        corr = m.predict(X)
        P[sr, hh - 1] = F[ti][sr, hh - 1] * np.exp(corr)
        return P


def F_lookup(F, ts, series, hs):
    out = np.empty(len(ts))
    for t in np.unique(ts):
        m = ts == t
        out[m] = F[t][series[m], hs[m] - 1]
    return out
