"""
Ансамбль с весами, подобранными ТОЛЬКО по прошлому (online stacking).

Для точки прогноза T, горизонта h и категории c берём ошибки каждого члена ансамбля на парах (T', h),
исход которых уже известен к моменту T (T' + h ≤ T). Если для этого h таких пар ещё нет - используем
ближайший меньший горизонт, по которому они есть; если нет ни одного - равные веса.
Берём top_k лучших членов, вес ∝ 1 / MAE². Никакой информации из будущего относительно T.
"""
import numpy as np
import pandas as pd

KEY = ["series", "origin", "h"]


def online_ensemble(members: dict, top_k=3, by_category=True, name="ens_online"):
    names = list(members)
    base = members[names[0]][KEY + ["target", "y_true", "territory_id", "category", "complete", "y_lag12"]].copy()
    idx = pd.MultiIndex.from_frame(base[KEY])
    P = np.vstack([members[n].set_index(KEY).loc[idx].y_pred.values for n in names])
    ae = np.abs(P - base.y_true.values[None, :])
    ok = base.complete.values & np.isfinite(base.y_true.values)
    origin, h, cat = base.origin.values, base.h.values, base.category.values
    tgt = base.target.values
    out = np.full(len(base), np.nan)
    weights_log = []
    for (T, hh, c), g in base.groupby(["origin", "h", "category"] if by_category else ["origin", "h"]).groups.items() \
            if by_category else [((k[0], k[1], None), v) for k, v in base.groupby(["origin", "h"]).groups.items()]:
        rows = np.asarray(g)
        w = None
        for h_use in range(hh, 0, -1):  # точный горизонт, иначе ближайший меньший
            past = ok & (h == h_use) & (tgt <= T) & ((cat == c) if c is not None else True)
            if past.sum() > 100:
                mae = ae[:, past].mean(1)
                best = np.argsort(mae)[:top_k]
                w = np.zeros(len(names)); w[best] = 1 / mae[best] ** 2; w /= w.sum()
                break
        if w is None:
            w = np.full(len(names), 1 / len(names))
        out[rows] = np.exp((np.log(P[:, rows]) * w[:, None]).sum(0))
        weights_log.append({"origin": T, "h": hh, "category": c, **dict(zip(names, w))})
    base["y_pred"] = out
    base["model"] = name
    return base, pd.DataFrame(weights_log)
