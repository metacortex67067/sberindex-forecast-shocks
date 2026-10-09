"""
Детекторы структурных изменений (шоков) в рядах расходов МО.

Все детекторы работают ОНЛАЙН-совместимо: решение о месяце t принимается по данным ≤ t
(кроме офлайн-методов ruptures, которые запускаются на «растущем окне» [0..t] - тоже без будущего).
Вход у большинства - остатки прогноза STF на 1 месяц вперёд: e_t = log y_t - log ŷ_{t|t-1}.
Это и есть «совмещение прогноза с обнаружением»: шок = то, что модель не смогла предсказать.

Семейства:
  A. Офлайн (ruptures): PELT, BinSeg - классика литературы (Killick et al. 2012; Truong et al. 2020).
  B. Онлайн: CUSUM (Page 1954), Page-Hinkley, BOCPD (Adams & MacKay 2007).
  C. Прогнозный: конформный порог на остаток (Vovk et al.) - тревога, если остаток вне интервала.
  D. Панельный: остаток МО относительно соседей по категории и региону (убирает общие шоки - ставку, сезон).
  E. Многомерный: сумма квадратов стандартизованных остатков по 6 категориям сразу.
  F. Гибрид с новостями: порог снижается, если в МО всплеск новостей (априорная вероятность шока выше).
Каждый детектор возвращает матрицу оценок score (n_series × n_months); тревога = score > порога.
Порог подбирается так, чтобы на «чистых» рядах было ≈ 2 ложные тревоги на 100 ряд-месяцев.
"""
import numpy as np
import pandas as pd


def robust_z(E, axis=None):
    med = np.nanmedian(E, axis=axis, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(E - med), axis=axis, keepdims=True)
    return (E - med) / np.where(mad > 1e-9, mad, np.nan)


def expanding_scale(E, min_obs=4, floor=0.01):
    """Масштаб остатков каждого ряда по прошлому (MAD за месяцы < t) - без заглядывания вперёд.
    Нужны ≥ min_obs прошлых остатков; нижняя граница floor защищает от деления на ~0."""
    S = np.full_like(E, np.nan)
    for t in range(E.shape[1]):
        past = E[:, :t]
        cnt = np.isfinite(past).sum(1)
        if (cnt >= min_obs).any():
            med = np.nanmedian(past, 1)
            mad = 1.4826 * np.nanmedian(np.abs(past - med[:, None]), 1)
            S[:, t] = np.where(cnt >= min_obs, np.maximum(mad, floor), np.nan)
    return S


def conformal_score(E, cat_idx, t_cal=6):
    """|e_t| / квантиль |e| по категории за прошлые месяцы (калибровка на t' < t)."""
    A = np.abs(E)
    Sc = np.full_like(E, np.nan)
    for t in range(t_cal, E.shape[1]):
        for c in np.unique(cat_idx):
            m = cat_idx == c
            q = np.nanquantile(A[m, :t], 0.9)
            Sc[m, t] = A[m, t] / q
    return Sc


def panel_score(E, cat_idx, region):
    """Остаток минус медиана остатков той же категории в регионе (в тот же месяц), в единицах MAD."""
    df_idx = pd.Series(list(zip(cat_idx, region)))
    groups = df_idx.groupby(df_idx).indices
    R = np.full_like(E, np.nan)
    for _, ix in groups.items():
        if len(ix) >= 4:
            R[ix] = E[ix] - np.nanmedian(E[ix], axis=0, keepdims=True)
        else:  # мало соседей: сравнение со всей категорией
            c = cat_idx[ix[0]]
            m = cat_idx == c
            R[ix] = E[ix] - np.nanmedian(E[m], axis=0, keepdims=True)
    S = expanding_scale(R)
    return np.abs(R) / S


def cusum_score(E, k=0.5, scale=None):
    """Двусторонний CUSUM по стандартизованным остаткам (накопление отклонений сверх k сигм)."""
    Z = E / (scale if scale is not None else expanding_scale(E))
    Sp = np.zeros(len(E)); Sn = np.zeros(len(E))
    out = np.full_like(E, np.nan)
    for t in range(E.shape[1]):
        z = np.nan_to_num(Z[:, t])
        Sp = np.maximum(0, Sp + z - k); Sn = np.maximum(0, Sn - z - k)
        out[:, t] = np.maximum(Sp, Sn)
        out[~np.isfinite(Z[:, t]), t] = np.nan
    return out


def cusum_reset_score(E, k=0.5, h=4.0, scale=None):
    """Классический двусторонний CUSUM с «решающим интервалом» h (Page 1954; Montgomery): после превышения h
    статистика обнуляется (перезапуск после тревоги), поэтому один шок не «заражает» все следующие месяцы.
    Оценка месяца t - значение статистики до обнуления; порог тревоги калибруется так же, как у остальных."""
    Z = E / (scale if scale is not None else expanding_scale(E))
    Sp = np.zeros(len(E)); Sn = np.zeros(len(E))
    out = np.full_like(E, np.nan)
    for t in range(E.shape[1]):
        ok = np.isfinite(Z[:, t]); z = np.nan_to_num(Z[:, t])
        Sp = np.where(ok, np.maximum(0, Sp + z - k), Sp); Sn = np.where(ok, np.maximum(0, Sn - z - k), Sn)
        st = np.maximum(Sp, Sn)
        out[:, t] = np.where(ok, st, np.nan)
        hit = st > h
        Sp[hit] = 0.0; Sn[hit] = 0.0
    return out


def expanding_median_norm(S):
    """S[:, t] / медиана S по всем рядам за месяцы ≤ t (нормировка без заглядывания вперёд)."""
    out = np.full_like(S, np.nan)
    for t in range(S.shape[1]):
        med = np.nanmedian(S[:, : t + 1]) if np.isfinite(S[:, : t + 1]).any() else np.nan
        out[:, t] = S[:, t] / med if med and np.isfinite(med) else np.nan
    return out


def regional_score(E_raw, cat_idx, region, min_mo=4):
    """Региональный уровень: медиана остатков категории в регионе (после вычета общего эффекта страны) -
    насколько регион в этом месяце отклонился от страны; в единицах собственного масштаба ряда региона
    (MAD по прошлым месяцам). Оценка присваивается всем МО региона в этой категории.
    Ловит шоки, общие для всего региона (паводок на весь регион), которые панельный детектор не видит по построению."""
    E = remove_common(E_raw, cat_idx)
    out = np.full_like(E, np.nan)
    key = pd.Series(list(zip(cat_idx, region)))
    for _, ix in key.groupby(key).indices.items():
        if len(ix) < min_mo:
            continue
        m = np.nanmedian(E[ix], axis=0, keepdims=True)
        sc = expanding_scale(m)
        out[ix] = np.abs(m) / sc
    return out


def expanding_rank(S):
    """Перцентильный ранг оценки месяца t среди всех оценок за месяцы ≤ t (без заглядывания вперёд):
    приводит разные детекторы к одной шкале, чтобы их можно было объединять."""
    out = np.full_like(S, np.nan)
    for t in range(S.shape[1]):
        past = S[:, : t + 1].ravel()
        past = np.sort(past[np.isfinite(past)])
        if not len(past):
            continue
        v = S[:, t]; ok = np.isfinite(v)
        out[ok, t] = np.searchsorted(past, v[ok], side="right") / len(past)
    return out


def all_scores(E_raw, keys, cfg, slow_rows=None):
    """Оценки всех детекторов на одной матрице остатков (n_series × месяцы).
    slow_rows - подвыборка рядов для медленных офлайн-детекторов (BOCPD, PELT, BinSeg); None - все ряды."""
    cat, reg, tid = keys.cat_idx.values, keys.region_code.values, keys.territory_id.values
    S = {"conformal_raw": conformal_score(E_raw, cat)}  # абляция: без удаления общего эффекта месяца
    E = remove_common(E_raw, cat)
    S["conformal"] = conformal_score(E, cat)
    S["panel"] = panel_score(E_raw, cat, reg)  # сам вычитает медиану региона
    S["cusum"] = cusum_score(E, k=cfg["cusum_k"])
    S["cusum_reset"] = cusum_reset_score(E, k=cfg["cusum_k"], h=cfg.get("cusum_h", 4.0))
    S["page_hinkley"] = page_hinkley_score(E, delta=cfg["ph_delta"])
    S["multivariate"] = multivariate_score(E, tid)
    S["panel_multivariate"] = 0.5 * (expanding_median_norm(S["panel"]) + expanding_median_norm(S["multivariate"]))
    rows = np.arange(len(E)) if slow_rows is None else slow_rows
    for name in ("bocpd", "pelt", "binseg"):
        S[name] = np.full(E.shape, np.nan)
    S["bocpd"][rows] = bocpd_score(E[rows], hazard=cfg["bocpd_hazard"])
    S["pelt"][rows] = ruptures_score(np.nan_to_num(E[rows]), "pelt")
    S["binseg"][rows] = ruptures_score(np.nan_to_num(E[rows]), "binseg")
    # комбинированный: тревога, если МО отклонился от соседей по региону (panel) или есть слом в собственной
    # истории ряда (BOCPD); второй ловит шоки, общие для всего региона
    S["panel_bocpd"] = np.fmax(expanding_rank(S["panel"]), expanding_rank(S["bocpd"]))
    # двухуровневый: МО необычно отличается от своего региона (panel) или весь регион в категории
    # необычно отличается от страны (regional)
    S["regional"] = regional_score(E_raw, cat, reg)
    S["panel_regional"] = np.fmax(expanding_rank(S["panel"]), expanding_rank(S["regional"]))
    return S


def page_hinkley_score(E, delta=0.1):
    Z = E / expanding_scale(E)
    out = np.full_like(E, np.nan)
    cum = np.zeros(len(E)); mn = np.zeros(len(E)); mx = np.zeros(len(E)); mean = np.zeros(len(E)); n = np.zeros(len(E))
    for t in range(E.shape[1]):
        z = Z[:, t]; ok = np.isfinite(z); z = np.nan_to_num(z)
        n += ok; mean = np.where(ok, mean + (z - mean) / np.maximum(n, 1), mean)
        cum = cum + np.where(ok, z - mean - delta, 0)
        mn = np.minimum(mn, cum); mx = np.maximum(mx, cum)
        out[:, t] = np.where(ok, np.maximum(cum - mn, mx - cum), np.nan)
    return out


def bocpd_score(E, hazard=1 / 24, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
    """BOCPD (Adams & MacKay, 2007), модель Normal-Gamma. Оценка = P(длина серии ≤ 1 | данные ≤ t)."""
    from scipy.stats import t as student
    Z = E / expanding_scale(E)
    n, T = Z.shape
    out = np.full((n, T), np.nan)
    for i in range(n):
        R = np.array([1.0]); mu = np.array([mu0]); ka = np.array([kappa0]); al = np.array([alpha0]); be = np.array([beta0])
        for t in range(T):
            x = Z[i, t]
            if not np.isfinite(x):
                continue
            scale = np.sqrt(be * (ka + 1) / (al * ka))
            pred = student.pdf(x, 2 * al, loc=mu, scale=scale)
            growth = R * pred * (1 - hazard)
            cp = (R * pred * hazard).sum()
            R = np.append(cp, growth); R /= R.sum()
            out[i, t] = R[:2].sum()
            mu_n = (ka * mu + x) / (ka + 1); ka_n = ka + 1; al_n = al + 0.5
            be_n = be + ka * (x - mu) ** 2 / (2 * (ka + 1))
            mu = np.append(mu0, mu_n); ka = np.append(kappa0, ka_n); al = np.append(alpha0, al_n); be = np.append(beta0, be_n)
    return out


def ruptures_score(L, method="pelt", pens=(0.5, 1, 2, 4, 8, 16, 32), min_size=2):
    """Растущее окно [0..t]; оценка = наибольший штраф (в единицах дисперсии разностей), при котором
    детектор всё ещё ставит точку смены в t-1 или t. Непрерывная оценка позволяет честно калибровать порог."""
    import ruptures as rpt
    n, T = L.shape
    out = np.zeros((n, T))
    for i in range(n):
        x = L[i]
        for t in range(6, T):
            seg = x[: t + 1]
            if not np.isfinite(seg).all():
                continue
            v = np.var(np.diff(seg)) or 1e-6
            algo = (rpt.Pelt(model="l2", min_size=min_size) if method == "pelt"
                    else rpt.Binseg(model="l2", min_size=min_size)).fit(seg.reshape(-1, 1))
            best = 0.0
            for p in pens:
                bk = [b for b in algo.predict(pen=p * v) if b < len(seg)]
                if any(b >= t - 1 for b in bk):
                    best = p
                else:
                    break
            out[i, t] = best
    return out


def remove_common(E, cat_idx):
    """Вычитает общий для категории эффект месяца (медиана остатков всех МО категории в этом месяце):
    остаётся только то, что произошло ИМЕННО в этом МО."""
    R = E.copy()
    for c in np.unique(cat_idx):
        m = cat_idx == c
        R[m] = E[m] - np.nanmedian(E[m], axis=0, keepdims=True)
    return R


def multivariate_score(E, territory_ids):
    """Для каждого МО: сумма квадратов стандартизованных остатков по всем категориям (≈ χ²). Оценка
    присваивается всем рядам МО."""
    Z = E / expanding_scale(E)
    df = pd.DataFrame(Z ** 2)
    df["tid"] = territory_ids
    agg = df.groupby("tid").transform("mean")
    return np.sqrt(agg.values)


def news_adjusted(score, surge, gamma=0.3):
    """Оценка детектора × (1 + gamma·max(всплеск новостей, 0)): при всплеске новостей в МО меньшее
    отклонение расходов уже считается шоком (байесовская логика: выше априорная вероятность)."""
    return score * (1 + gamma * np.clip(np.nan_to_num(surge), 0, None))
