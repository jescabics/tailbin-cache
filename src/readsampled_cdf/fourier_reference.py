"""Reference-quality Fourier inversion for finite-depth read-sampled tail-bin CDFs.

This module evaluates the same finite-volume PGF target used by the production
finite-depth CGF engine, but recovers the CDF by numerical Fourier inversion
rather than any saddlepoint approximation.  It is used by this package as
the coefficient/PGF target for cache generation and validation.

For an integer-valued nonnegative count Y with characteristic function
    phi(t) = E[exp(i t Y)],
the exact lattice inversion identity is
    P(Y <= k) = (1/pi) int_0^pi Re[ phi(t) D_k(t) ] dt,
where
    D_k(t) = sum_{n=0}^k exp(-i t n)
           = (1 - exp(-i t (k+1))) / (1 - exp(-i t)).
As the quadrature tolerance is tightened and the finite-volume grid/RK step is
refined, this converges to the finite-depth grid target.  This is not a
saddlepoint approximation and does not assume a negative-binomial shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math, time
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.stats import norm

from .distribution import ModelParams, extinction_probability
from .frequency_volume import AdaptiveFrequencyVolumeEngine
from .fast_frequency import FastFrequencyConfig, make_fast_grid_config


@dataclass(frozen=True)
class FourierCDFConfig:
    n_bins: int = 24
    steps_per_time: float = 1.4
    min_steps: int = 32
    max_steps: int = 120
    quad_epsabs: float = 1e-8
    quad_epsrel: float = 1e-8
    quad_limit: int = 500
    clip_eps: float = 1e-12
    addition_rule: str = "linear"
    representative: str = "geometric"
    ode_method: str = "DOP853"
    ode_rtol: float = 1e-7
    ode_atol: float = 1e-9
    use_solve_ivp: bool = True


def _clip_cdf(x: float, eps: float) -> float:
    return float(np.clip(float(x), float(eps), 1.0 - float(eps))) if np.isfinite(x) else 0.5


def _z_from_cdf(x: float, eps: float) -> float:
    return float(norm.ppf(_clip_cdf(x, eps)))


class ComplexPGFGridEngine:
    """Complex-PGF evaluator on the same finite-volume frequency grid."""

    def __init__(self, params: ModelParams, config: Optional[FourierCDFConfig] = None):
        self.params = params
        self.config = config or FourierCDFConfig()
        fcfg = FastFrequencyConfig(
            n_bins=int(self.config.n_bins),
            steps_per_time=float(self.config.steps_per_time),
            min_steps=int(self.config.min_steps),
            max_steps=int(self.config.max_steps),
            addition_rule=str(self.config.addition_rule),
            representative=str(self.config.representative),
        )
        self.grid_engine = AdaptiveFrequencyVolumeEngine(params, config=make_fast_grid_config(fcfg))
        if self.grid_engine._uniform_i is None:
            raise RuntimeError("Fourier backend requires sparse linear/uniform addition arrays")
        self.rho = np.asarray(self.grid_engine.rho, dtype=float)
        self.ii = np.asarray(self.grid_engine._uniform_i, dtype=np.int32)
        self.jj = np.asarray(self.grid_engine._uniform_j, dtype=np.int32)
        self.kk = np.asarray(self.grid_engine._uniform_k, dtype=np.int32)
        self.ww = np.asarray(self.grid_engine._uniform_w, dtype=float)
        self.Q0 = np.asarray(self.grid_engine._initial_Q(), dtype=np.complex128)
        self.ext_prob = float(extinction_probability(params.lam, params.delta, params.T))
        self.surv = max(1.0 - self.ext_prob, np.finfo(float).tiny)
        self.n_steps = int(min(max(int(math.ceil(float(params.T) * float(self.config.steps_per_time))), int(self.config.min_steps)), int(self.config.max_steps)))
        self._pgf_cache: Dict[Tuple[float, float], complex] = {}

    @property
    def n_classes(self) -> int:
        return int(self.Q0.size)

    def _conv(self, Q: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_classes, dtype=np.complex128)
        # np.add.at supports complex accumulation and keeps implementation simple.
        np.add.at(out, self.kk, self.ww * Q[self.ii] * Q[self.jj])
        return out

    def _rhs(self, Q: np.ndarray, z: complex) -> np.ndarray:
        p = self.params
        c = z - 1.0
        dQ = float(p.lam) * (self._conv(Q) - Q) - float(p.delta) * Q + float(p.u) * c * self.rho * Q
        dQ[0] += float(p.delta)
        return dQ

    def pgf(self, z: complex) -> complex:
        # Cache rounded real/imag coordinates; quadrature frequently revisits nodes.
        key = (round(float(np.real(z)), 14), round(float(np.imag(z)), 14))
        if key in self._pgf_cache:
            return self._pgf_cache[key]
        Q = self.Q0.copy()
        T = float(self.params.T)
        if self.n_steps <= 0 or T == 0.0:
            pass
        elif bool(self.config.use_solve_ivp):
            def rhs_time(_t, y):
                return self._rhs(y, z)
            sol = solve_ivp(
                rhs_time, (0.0, T), Q, method=str(self.config.ode_method),
                rtol=float(self.config.ode_rtol), atol=float(self.config.ode_atol)
            )
            if not sol.success or not np.isfinite(sol.y[:, -1]).all():
                self._pgf_cache[key] = complex(np.nan, np.nan)
                return self._pgf_cache[key]
            Q = sol.y[:, -1]
        else:
            dt = T / float(self.n_steps)
            for _ in range(self.n_steps):
                k1 = self._rhs(Q, z)
                k2 = self._rhs(Q + 0.5 * dt * k1, z)
                k3 = self._rhs(Q + 0.5 * dt * k2, z)
                k4 = self._rhs(Q + dt * k3, z)
                Q = Q + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        factor = np.exp(float(self.params.theta_f) * (z - 1.0) * self.rho)
        G = complex(np.dot(Q, factor))
        if self.params.condition_on_survival:
            G = (G - self.ext_prob) / self.surv
        self._pgf_cache[key] = G
        return G

    def characteristic(self, t: float) -> complex:
        return self.pgf(np.exp(1j * float(t)))

    @staticmethod
    def _dirichlet(k: int, t: float) -> complex:
        # D_k(t)=sum_{n=0}^k exp(-itn).  Use stable sine form away from zero.
        t = float(t)
        if abs(t) < 1e-10:
            return complex(int(k) + 1, 0.0)
        return (1.0 - np.exp(-1j * t * (int(k) + 1))) / (1.0 - np.exp(-1j * t))

    def cdf_fourier(self, k: int) -> Tuple[float, Dict[str, float | str]]:
        k = int(k)
        t0 = time.perf_counter()
        evals = {"n": 0}
        def integrand(t: float) -> float:
            evals["n"] += 1
            return float(np.real(self.characteristic(t) * self._dirichlet(k, t)))
        val, err = quad(
            integrand, 0.0, math.pi,
            epsabs=float(self.config.quad_epsabs),
            epsrel=float(self.config.quad_epsrel),
            limit=int(self.config.quad_limit),
            points=[0.0, math.pi],
        )
        F_raw = float(val / math.pi)
        F = _clip_cdf(F_raw, float(self.config.clip_eps))
        return F, {
            "method": "fourier_lattice_inversion",
            "cdf_raw": float(F_raw),
            "cdf": float(F),
            "z": _z_from_cdf(F, float(self.config.clip_eps)),
            "quad_abs_error_estimate": float(err / math.pi),
            "quad_evaluations": float(evals["n"]),
            "n_bins": float(self.config.n_bins),
            "n_steps": float(self.n_steps),
            "steps_per_time": float(self.config.steps_per_time),
            "seconds": float(time.perf_counter() - t0),
            "finite_depth_grid_target": 1.0,
            "convergent_quadrature_reference": 1.0,
        }
