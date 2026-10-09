"""
Оценка детекторов шоков.

Проблема: размеченных шоков в данных нет. Решение - два вида проверки:
1) Полусинтетика: в РЕАЛЬНЫЕ ряды вставляем шоки известного типа, размера и даты; детектор не знает где.
   Типы: step (сдвиг уровня, навсегда), dip (провал на 1-2 мес. и восстановление - паводок, отключения),
         ramp (плавный сдвиг за 3 мес.), spike (разовый всплеск).
   Метрики: precision / recall / F1 с допуском ±1 мес., средняя задержка обнаружения (мес.),
            ложные тревоги на 100 ряд-месяцев (на «чистой» половине рядов).
   Порог каждого детектора подбирается на ОТДЕЛЬНОЙ калибровочной выборке рядов под одинаковый уровень
   ложных тревог (2 на 100 ряд-месяцев) - так детекторы сравниваются честно.
2) Реальные события (реестр с первоисточниками): совпадение тревог с известными шоками и опережение (lead time).
"""
import numpy as np
import pandas as pd

SHOCK_TYPES = ("step", "dip", "ramp", "spike")


def inject(Y, rng, frac=0.5, t_min=12, t_max=22, size=(0.1, 0.3)):
    """Возвращает (Y_shocked, labels): labels - DataFrame(series, t, type, size) для рядов с шоком."""
    Y2 = Y.copy()
    n, T = Y.shape
    sel = rng.random(n) < frac
    rows = []
    for i in np.where(sel)[0]:
        t = int(rng.integers(t_min, t_max + 1))
        typ = SHOCK_TYPES[int(rng.integers(len(SHOCK_TYPES)))]
        s = float(rng.uniform(*size)) * (1 if rng.random() < 0.3 else -1)  # падения чаще ростов
        f = np.ones(T)
        if typ == "step":
            f[t:] = 1 + s
        elif typ == "dip":
            d = int(rng.integers(1, 3)); f[t:t + d] = 1 + s
        elif typ == "ramp":
            for j in range(3):
                if t + j < T:
                    f[t + j:] = 1 + s * (j + 1) / 3
        else:
            f[t] = 1 + s
        Y2[i] = Y[i] * f
        rows.append((i, t, typ, s))
    return Y2, pd.DataFrame(rows, columns=["series", "t", "type", "size"]), sel


def inject_regional(Y, keys, rng, frac_regions=0.3, t_min=12, t_max=22, size=(0.1, 0.3)):
    """Региональные шоки: в выбранных регионах один и тот же шок (тип, размер, месяц) получают ВСЕ МО региона
    в одной категории - как паводок, затронувший весь регион, или региональные ограничения.
    Такой шок не виден детектору, который сравнивает МО с соседями по региону (панельному), - проверяем это."""
    Y2 = Y.copy()
    T = Y.shape[1]
    rows = []
    sel = np.zeros(len(Y), bool)
    for rc in np.unique(keys.region_code.values):
        if rng.random() >= frac_regions:
            continue
        c = int(rng.integers(keys.cat_idx.max() + 1))
        idx = np.where((keys.region_code.values == rc) & (keys.cat_idx.values == c))[0]
        if not len(idx):
            continue
        t = int(rng.integers(t_min, t_max + 1))
        typ = SHOCK_TYPES[int(rng.integers(len(SHOCK_TYPES)))]
        s = float(rng.uniform(*size)) * (1 if rng.random() < 0.3 else -1)
        f = np.ones(T)
        if typ == "step":
            f[t:] = 1 + s
        elif typ == "dip":
            d = int(rng.integers(1, 3)); f[t:t + d] = 1 + s
        elif typ == "ramp":
            for j in range(3):
                if t + j < T:
                    f[t + j:] = 1 + s * (j + 1) / 3
        else:
            f[t] = 1 + s
        Y2[idx] = Y[idx] * f[None, :]
        sel[idx] = True
        rows += [(i, t, typ, s) for i in idx]
    return Y2, pd.DataFrame(rows, columns=["series", "t", "type", "size"]), sel


def threshold_for_far(score, clean_mask, months, far=0.02):
    """Порог, при котором доля тревог на чистых рядах в месяцах months равна far."""
    v = score[clean_mask][:, months].ravel()
    v = v[np.isfinite(v)]
    return np.quantile(v, 1 - far) if len(v) else np.inf


def evaluate(score, labels, shocked_mask, thr, months, tol=1):
    """Первая тревога после начала шока (t-tol … t+2) = обнаружение. Также задержка и ложные тревоги."""
    alarm = np.nan_to_num(score) > thr
    hits, delays = 0, []
    for r in labels.itertuples():
        win = [m for m in range(r.t - tol, min(r.t + 3, alarm.shape[1])) if m in months]
        a = [m for m in win if alarm[r.series, m]]
        if a:
            hits += 1
            delays.append(max(0, a[0] - r.t))
    recall = hits / max(len(labels), 1)
    clean = ~shocked_mask
    fa_rate = alarm[clean][:, months].mean()
    # precision: тревоги на рядах с шоком вне окна шока считаются ложными
    tp_alarms, all_alarms = 0, alarm[:, months].sum()
    lab = labels.set_index("series")
    for i in np.where(shocked_mask)[0]:
        t = lab.at[i, "t"] if i in lab.index else -99
        for m in months:
            if alarm[i, m] and (t - tol) <= m <= (t + 2):
                tp_alarms += 1
    precision = tp_alarms / max(all_alarms, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return dict(recall=recall, precision=precision, F1=f1, delay=np.mean(delays) if delays else np.nan,
                false_alarms_per100=100 * fa_rate, n_shocks=len(labels))
