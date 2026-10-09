"""
Фундаментальные (предобученные) модели временных рядов в том же интерфейсе, что и остальные модели.

Проблема: у МО 12-23 точки истории - для foundation-моделей это очень мало (им нужен контекст с сезонностью).
Поэтому сравниваем три режима подачи данных:
  raw       - исходный ряд как есть (так обычно делают; ожидаемо слабо).
  stf_resid - ряд, очищенный от сезонности профилем STF (в логарифмах); модель прогнозирует уровень/тренд,
              сезонность возвращается обратно. Модель решает только ту часть задачи, где она сильна.
  extended  - «удлинённый контекст»: к истории МО добавляется синтетическая предыстория 2019-2022 гг.,
              построенная как уровень МО × динамика национального ряда СберИндекса × сезонный профиль МО.
              Предыстория синтетическая (это честно оговаривается); проверяем, помогает ли длинный контекст.
Модели: Chronos-2 (Amazon, 2025), TimesFM 2.5 (Google, 2025), TiRex (NX-AI, 2025). Нужен GPU.
"""
import numpy as np
import pandas as pd

from ..config import NATIONAL_MAP
from .stf import STF, last_valid


class _Chronos2:
    def __init__(self, device="cuda"):
        from chronos import BaseChronosPipeline
        self.p = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=device)

    def forecast(self, ctxs, h):
        import torch
        out = []
        for i in range(0, len(ctxs), 1024):
            batch = [torch.tensor(c, dtype=torch.float32) for c in ctxs[i:i + 1024]]
            q = self.p.predict(batch, prediction_length=h)
            q = [x if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in q]
            for x in q:
                x = x.squeeze(0)
                out.append(x[x.shape[0] // 2].cpu().numpy())
        return np.vstack(out)


class _TimesFM25:
    def __init__(self, device="cuda"):
        import timesfm
        self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        # per_core_batch_size по умолчанию 1 (очень медленно); пакетная обработка не меняет прогноз ряда
        self.m.compile(timesfm.ForecastConfig(max_context=512, max_horizon=64, normalize_inputs=True,
                                              use_continuous_quantile_head=True, fix_quantile_crossing=True,
                                              per_core_batch_size=128))

    def forecast(self, ctxs, h):
        point, _ = self.m.forecast(horizon=h, inputs=[np.asarray(c, dtype=np.float32) for c in ctxs])
        return np.asarray(point)[:, :h]


class _TiRex:
    def __init__(self, device="cuda"):
        from tirex import load_model
        self.m = load_model("NX-AI/TiRex", device=device)

    @staticmethod
    def _to_np(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        elif isinstance(x, (list, tuple)) and len(x) and hasattr(x[0], "detach"):
            x = np.stack([np.asarray(v.detach().cpu().numpy()) for v in x])
        return np.asarray(x, dtype=float)

    def forecast(self, ctxs, h):
        """Устойчивый разбор ответа: (квантили [B,h,Q], среднее [B,h]) в любом порядке/формате;
        если пришёл только тензор квантилей - берём медиану (средний квантиль)."""
        import torch
        out = []
        for i in range(0, len(ctxs), 512):
            batch = ctxs[i:i + 512]
            L = max(len(c) for c in batch)
            arr = np.full((len(batch), L), np.nan, dtype=np.float32)
            for j, c in enumerate(batch):
                arr[j, L - len(c):] = c
            res = self.m.forecast(context=torch.tensor(arr), prediction_length=h)
            parts = list(res) if isinstance(res, (tuple, list)) else [res]
            parts = [self._to_np(p) for p in parts]
            point = None
            for p_ in parts:
                if p_.ndim == 2 and p_.shape[0] == len(batch):
                    point = p_
            if point is None:
                for p_ in parts:
                    if p_.ndim == 3 and p_.shape[0] == len(batch):
                        point = p_[:, :, p_.shape[2] // 2]
            if point is None:
                raise ValueError(f"неожиданный формат ответа TiRex: {[p_.shape for p_ in parts]}")
            out.append(point[:, :h])
        return np.vstack(out)


BACKENDS = {"chronos2": _Chronos2, "timesfm25": _TimesFM25, "tirex": _TiRex}


class FoundationModel:
    def __init__(self, backend: str, mode: str = "stf_resid", stf_kwargs=None, device="cuda"):
        self.backend_name, self.mode = backend, mode
        self.name = f"{backend}_{mode}"
        self.stf = STF(**(stf_kwargs or {}))
        self._b, self.device = None, device

    @property
    def b(self):
        if self._b is None:
            self._b = BACKENDS[self.backend_name](self.device)
        return self._b

    @staticmethod
    def _clean(x):
        s = pd.Series(x).interpolate(limit_direction="both")
        return s.values

    def predict(self, pm, ti, h_max, ctx):
        Y = pm.Y[:, : ti + 1]
        moy = pm.months.month.values - 1
        tgt_moy = [(pm.months[ti].month - 1 + h) % 12 for h in range(1, h_max + 1)]
        if self.mode == "raw":
            ctxs = [self._clean(y[np.argmax(np.isfinite(y)):]) for y in Y]
            P = self.b.forecast(ctxs, h_max)
            return np.where(np.isfinite(P), P, last_valid(pm.Y, ti)[:, None])
        # профиль сезонности STF на момент ti (только прошлое)
        from .stf import category_growth
        G, _ = category_growth(pm, ti, ctx, self.stf.kg)
        g_month = np.log(pm.keys.category.map(G).values) / 12
        S = self.stf.profiles(pm, ti, g_month, ctx)
        Z = np.log(Y) - S[:, moy[: ti + 1]]
        ctxs = []
        for i, z in enumerate(Z):
            z = self._clean(z[np.argmax(np.isfinite(z)):])
            if self.mode == "extended":
                z = np.concatenate([self._backcast(pm, ctx, i, z, pm.months[ti]), z])
            ctxs.append(z)
        Zf = self.b.forecast(ctxs, h_max)
        P = np.exp(Zf + S[:, tgt_moy])
        return np.where(np.isfinite(P), P, last_valid(pm.Y, ti)[:, None])

    _bc_cache: dict = {}

    @classmethod
    def _national_deseason(cls, ctx, category, T):
        """Национальный ряд категории (лог), очищенный от сезонности, ТОЛЬКО по данным до T включительно:
        сезонный профиль = средние по месяцам остатков после линейного тренда (без заглядывания вперёд)."""
        key = (category, pd.Timestamp(T), id(ctx["national"]))
        if key not in cls._bc_cache:
            N = ctx["national"][NATIONAL_MAP[category]]
            L = np.log(N[N.index <= T])
            t = np.arange(len(L), dtype=float)
            slope, icpt = np.polyfit(t, L.values, 1)
            resid = L - (icpt + slope * t)
            prof = resid.groupby(resid.index.month).transform("mean")
            cls._bc_cache[key] = L - prof
        return cls._bc_cache[key]

    @classmethod
    def _backcast(cls, pm, ctx, i, z, T, start="2019-01-01"):
        """Синтетическая предыстория (очищенная от сезонности, в логарифмах) до начала панели:
        уровень МО в первые 6 мес. панели + динамика национального ряда соответствующей категории.
        Используются только данные национального ряда до точки прогноза T (исправлено 08.10.2026:
        раньше сезонный профиль считался по всему ряду до 2026-08 и без удаления тренда)."""
        Ld = cls._national_deseason(ctx, pm.keys.category.iat[i], T)
        hist = Ld[(Ld.index >= start) & (Ld.index < pm.months[0])]
        anchor = Ld[(Ld.index >= pm.months[0]) & (Ld.index < pm.months[0] + pd.DateOffset(months=6))].mean()
        return (np.nanmean(z[:6]) + (hist - anchor)).values
