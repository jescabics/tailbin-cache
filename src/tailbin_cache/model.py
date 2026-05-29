from __future__ import annotations

from .grid import ParameterPoint
from readsampled_cdf.distribution import ModelParams


def make_model_params(point: ParameterPoint, alpha: float) -> ModelParams:
    return ModelParams(
        lam=float(point.lam),
        delta=float(point.delta),
        u=float(point.u),
        T=float(point.T),
        N_const=float(point.effective_N),
        alpha=float(alpha),
        depth=int(point.depth),
        theta_f=float(point.theta_f),
        condition_on_survival=bool(point.condition_on_survival),
    )
