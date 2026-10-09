"""
Базовые модели и семейство STF (Seasonality Transfer Forecaster - «перенос сезонности»).

Идея STF простыми словами: у отдельного МО всего 1-2 года истории, и по нему одному сезонность
не оценить. Но у 2 000 МО сезонность очень похожа, а у страны в целом есть ряды с 2018 г.
Поэтому прогноз МО собирается из трёх частей:
    уровень МО (последние месяцы, очищенные от сезонности)
  + темп роста (годовой прирост категории: по панели, а в начале - по национальному ряду СберИндекса)
  + сезонный профиль (общий для категории → уточнённый для региона → чуть-чуть индивидуальный для МО).
Все вычисления векторные по всем рядам сразу.
"""
import numpy as np
import pandas as pd

from ..config import CATEGORIES, NATIONAL_MAP


def last_valid(Y, ti):
    """Последнее известное значение каждого ряда на момент ti (для рядов с пропусками)."""
    sub = Y[:, : ti + 1]
    idx = np.where(~np.isnan(sub), np.arange(sub.shape[1]), -1).max(1)
    out = np.full(len(Y), np.nan)
    ok = idx >= 0
    out[ok] = sub[np.where(ok)[0], idx[ok]]
    return out


class Naive:
    name = "naive"

    def predict(self, pm, ti, h_max, ctx):
        return np.repeat(last_valid(pm.Y, ti)[:, None], h_max, axis=1)


class SeasonalNaive:
    name = "snaive"

    def predict(self, pm, ti, h_max, ctx):
        P = np.full((len(pm.Y), h_max), np.nan)
        for h in range(1, h_max + 1):
            j = ti + h - 12
            P[:, h - 1] = pm.Y[:, j] if j >= 0 else np.nan
        return P


# недельные категории СберИндекса для категорий панели без прямого месячного аналога
WEEKLY_MAP = {"Маркетплейсы": ["Маркетплейсы"],
              "Здоровье": ["Лекарства и медицинские товары", "Медицинские услуги"],
              # состав по определению СберИндекса: общественный транспорт, такси и каршеринг, топливо, ремонт авто,
              # перевозка и доставка отправлений (без авиа и ж/д)
              "Транспорт": ["Топливо", "Такси, каршеринг, аренда автомобилей", "Локальный транспорт",
                            "Сервис и обслуживание автомобилей", "Хранение и доставка грузов и корреспонденции"]}


def national_yoy(ctx, category, T, k=3):
    """Годовой прирост национального ряда СберИндекса (среднее за k последних месяцев до T).
    Для маркетплейсов/здоровья/транспорта при наличии - недельный ряд именно этой категории
    (последние 4 недели, закончившиеся не позже конца месяца T)."""
    W = ctx.get("weekly")
    if W is not None and category in WEEKLY_MAP:
        end = T + pd.offsets.MonthEnd(0)
        w = W.loc[W.index <= end, WEEKLY_MAP[category]].tail(4)
        if len(w) >= 2:
            return float(1 + w.mean(axis=1).mean() / 100)
    N = ctx["national"][NATIONAL_MAP[category]]
    vals = [N.get(T - pd.DateOffset(months=j)) / N.get(T - pd.DateOffset(months=j + 12)) for j in range(k)]
    return float(np.nanmean(vals))


def panel_yoy(pm, ti, k=3):
    """Годовой прирост каждого ряда (среднее за до k последних месяцев) - NaN, если истории < 13 мес."""
    vals = []
    for j in range(k):
        t = ti - j
        if t - 12 >= 0:
            vals.append(pm.Y[:, t] / pm.Y[:, t - 12])
    return np.nanmean(np.vstack(vals), axis=0) if vals else np.full(len(pm.Y), np.nan)


def category_growth(pm, ti, ctx, k=3):
    """Годовой прирост по категориям: медиана панели, если доступна, иначе национальный ряд."""
    T = pm.months[ti]
    g_ser = panel_yoy(pm, ti, k)
    G = {}
    for ci, c in enumerate(CATEGORIES):
        m = (pm.keys.cat_idx.values == ci) & np.isfinite(g_ser)
        G[c] = float(np.median(g_ser[m])) if m.sum() > 50 else national_yoy(ctx, c, T, k)
    return G, g_ser


class SeasonalGrowth:
    """Сезонный наивный прогноз × годовой прирост: ŷ(T+h) = y(T+h-12) · G_i.
    G_i = G_категории · (собственный прирост МО / G_категории)^alpha - частичное доверие истории МО."""

    def __init__(self, alpha=0.5, k=3, name=None):
        self.alpha, self.k = alpha, k
        self.name = name or f"snaive_growth_a{alpha}"

    def predict(self, pm, ti, h_max, ctx):
        G, g_ser = category_growth(pm, ti, ctx, self.k)
        Gc = pm.keys.category.map(G).values
        rel = np.where(np.isfinite(g_ser), g_ser / Gc, 1.0)
        Gi = Gc * np.clip(rel, 0.5, 2.0) ** self.alpha
        return SeasonalNaive().predict(pm, ti, h_max, ctx) * Gi[:, None]


def national_profile(ctx, category, T, years=5):
    """Сезонный профиль национального ряда (лог-отклонения по месяцам года), оценённый по данным до T."""
    N = ctx["national"][NATIONAL_MAP[category]]
    N = N[(N.index <= T) & (N.index > T - pd.DateOffset(years=years))]
    L = np.log(N)
    yoy = (L - L.shift(12)).dropna()
    g = yoy.mean() / 12 if len(yoy) else 0.0
    Lt = L - g * np.arange(len(L))
    prof = Lt.groupby(Lt.index.month).mean()
    prof = prof.reindex(range(1, 13))
    return (prof - prof.mean()).values


DIRECT_NATIONAL = {"Все категории", "Продовольствие", "Общественное питание"}


class STF:
    """Перенос сезонности с иерархическим сглаживанием профиля: категория → регион → МО.

    lam_region, lam_mo - доля «собственного» профиля региона/МО (0 - полностью общий профиль, 1 - свой).
    k_level - по скольким последним месяцам оценивается уровень ряда.
    """

    def __init__(self, lam_region=0.5, lam_mo=0.3, k_level=3, k_growth=3, beta_trend=0.0, w_nat=0.0,
                 damp=None, anchor_years=3, robust_level=False, lam_growth_region=0.0, damp_direct_only=False, name=None):
        self.lr, self.lm, self.kl, self.kg, self.bt, self.wn = lam_region, lam_mo, k_level, k_growth, beta_trend, w_nat
        self.damp, self.anchor_years, self.robust = damp, anchor_years, robust_level
        self.lgr, self.damp_direct_only = lam_growth_region, damp_direct_only
        self.name = name or f"stf_r{lam_region}_m{lam_mo}_k{k_level}_kg{k_growth}_b{beta_trend}_n{w_nat}"

    def profiles(self, pm, ti, g_month, ctx=None):
        """Сезонные отклонения (в логарифмах) по месяцам года: возвращает S (n_series × 12)."""
        L = np.log(pm.Y[:, : ti + 1])
        pos = np.arange(ti + 1)
        Lt = L - g_month[:, None] * pos[None, :]
        moy = pm.months[: ti + 1].month.values - 1
        D = np.full((len(L), 12), np.nan)
        for m in range(12):
            cols = np.where(moy == m)[0]
            if len(cols):
                D[:, m] = np.nanmean(Lt[:, cols], axis=1)
        D = D - np.nanmean(D, axis=1, keepdims=True)
        keys = pm.keys
        S = np.full_like(D, np.nan)
        for ci in range(len(CATEGORIES)):
            mc = keys.cat_idx.values == ci
            full = mc & np.isfinite(D).all(1)
            s_cat = np.nanmedian(D[full], axis=0) if full.sum() > 50 else np.full(12, np.nan)
            if ctx is not None and "national" in ctx:
                s_nat = national_profile(ctx, CATEGORIES[ci], pm.months[ti])
                # доля национального профиля: wn для категорий с прямым аналогом; если панельного профиля нет, 1
                w = self.wn if CATEGORIES[ci] in DIRECT_NATIONAL else 0.0
                s_cat = np.where(np.isfinite(s_cat), (1 - w) * s_cat + w * s_nat, s_nat)
            for rc in np.unique(keys.region_code.values[mc]):
                mr = mc & (keys.region_code.values == rc)
                fr = mr & np.isfinite(D).all(1)
                s_reg = s_cat + self.lr * (np.nanmedian(D[fr], axis=0) - s_cat) if fr.sum() >= 3 else s_cat
                own = np.where(np.isfinite(D[mr]), D[mr], s_reg)
                S[mr] = s_reg + self.lm * (own - s_reg)
        return S

    def predict(self, pm, ti, h_max, ctx):
        G, g_ser = category_growth(pm, ti, ctx, self.kg)
        Gc = pm.keys.category.map(G).values
        if self.lgr > 0 and np.isfinite(g_ser).sum() > 1000:
            # прирост региона (медиана по МО региона в категории), частично: G_rc = G_c·(G_reg/G_c)^λ
            df = pd.DataFrame({"g": np.log(g_ser), "c": pm.keys.cat_idx.values, "r": pm.keys.region_code.values})
            med = df.groupby(["c", "r"]).g.transform(lambda x: x.median() if x.notna().sum() >= 3 else np.nan).values
            adj = np.where(np.isfinite(med), med - np.log(Gc), 0.0)
            Gc = np.exp(np.log(Gc) + self.lgr * adj)
        g_month = np.log(Gc) / 12
        S = self.profiles(pm, ti, g_month, ctx)
        moy = pm.months.month.values - 1
        if self.bt > 0:
            # собственный тренд МО: наклон очищенного от сезонности ряда за последние 12 мес., сжатый к категории
            w = np.arange(max(0, ti - 11), ti + 1)
            Z = np.log(pm.Y[:, w]) - S[:, moy[w]]
            x = w - w.mean()
            ok = np.isfinite(Z)
            xm = np.where(ok, x, np.nan)
            xc = xm - np.nanmean(xm, axis=1, keepdims=True)
            zc = Z - np.nanmean(Z, axis=1, keepdims=True)
            slope = np.nansum(xc * zc, axis=1) / np.nansum(xc ** 2, axis=1)
            slope = np.where(np.isfinite(slope) & (ok.sum(1) >= 9), slope, g_month)
            g_month = g_month + self.bt * (np.clip(slope, g_month - 0.03, g_month + 0.03) - g_month)
        # уровень: среднее последних k_level очищенных от сезонности значений, приведённое к моменту T
        lv = []
        for j in range(self.kl):
            t = ti - j
            lv.append(np.log(pm.Y[:, t]) - S[np.arange(len(S)), moy[t]] + g_month * j)
        level = (np.nanmedian if self.robust else np.nanmean)(np.vstack(lv), axis=0)
        level = np.where(np.isfinite(level), level, np.log(last_valid(pm.Y, ti)))
        P = np.full((len(pm.Y), h_max), np.nan)
        cum = np.zeros(len(pm.Y))
        g_anchor = self._anchor(pm, ti, ctx) if self.damp else None
        for h in range(1, h_max + 1):
            m = (pm.months[ti].month - 1 + h) % 12
            if self.damp:  # рост стягивается к долгосрочному темпу национального ряда: g_h = a + (g - a)·φ^h
                a = np.where(np.isfinite(g_anchor), g_anchor, g_month)
                cum = cum + a + (g_month - a) * self.damp ** h
            else:
                cum = g_month * h
            P[:, h - 1] = np.exp(level + cum + S[:, m])
        return P

    def _anchor(self, pm, ti, ctx):
        """Долгосрочный месячный темп роста категории: средний годовой прирост нац. ряда за anchor_years лет до T."""
        T = pm.months[ti]
        a = {}
        for c in CATEGORIES:
            N = ctx["national"][NATIONAL_MAP[c]]
            N = N[N.index <= T]
            L = np.log(N)
            yoy = (L - L.shift(12)).dropna()
            a[c] = yoy[yoy.index > T - pd.DateOffset(years=self.anchor_years)].mean() / 12
        anchor = pm.keys.category.map(a).values
        if self.damp_direct_only:
            # у маркетплейсов, транспорта и здоровья нет прямого национального аналога, якорь = собственный текущий рост
            direct = pm.keys.category.isin(DIRECT_NATIONAL).values
            anchor = np.where(direct, anchor, np.nan)
        return anchor


class PerDay:
    """Обёртка: модель работает с расходами В ДЕНЬ (учёт длины месяца, в т.ч. високосного февраля 2024 г.)."""

    def __init__(self, model, name=None):
        self.m = model
        self.name = name or f"{model.name}_perday"

    def predict(self, pm, ti, h_max, ctx):
        from ..evaluation.backtest import PanelMatrix
        days = pm.months.days_in_month.values.astype(float)
        pm_d = PanelMatrix(Y=pm.Y / days[None, :], keys=pm.keys, months=pm.months)
        P = self.m.predict(pm_d, ti, h_max, ctx)
        tgt = pd.date_range(pm.months[ti] + pd.offsets.MonthBegin(1), periods=h_max, freq="MS").days_in_month.values
        return P * tgt[None, :]
