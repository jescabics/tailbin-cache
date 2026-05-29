from __future__ import annotations

"""Batched complex-PGF backend for dense cache generation.

The The initial cache builder evaluated one complex PGF node at a time.  That was
simple but not feasible for large Cauchy/FFT tables because Kmax=50k requires
O(1e5) complex nodes per table when base/refinement are included.  This module
keeps the same finite-volume target but integrates many complex nodes in one
compiled RK4 call.

This is an engineering acceleration, not a mathematical change: for a node
z=r exp(i theta), pgf_many returns the same finite-grid PGF as the scalar
ComplexPGFGridEngine configured with fixed-step RK4.
"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco

from readsampled_cdf.distribution import ModelParams, extinction_probability
from readsampled_cdf.frequency_volume import AdaptiveFrequencyVolumeEngine
from readsampled_cdf.fast_frequency import FastFrequencyConfig, make_fast_grid_config
from readsampled_cdf.fourier_reference import FourierCDFConfig


@njit(cache=True)
def _conv_batch(Q, ii, jj, kk, ww):
    B = Q.shape[0]
    J = Q.shape[1]
    E = kk.size
    out = np.zeros((B, J), dtype=np.complex128)
    for e in range(E):
        i = ii[e]
        j = jj[e]
        k = kk[e]
        w = ww[e]
        for b in range(B):
            out[b, k] += w * Q[b, i] * Q[b, j]
    return out


@njit(cache=True)
def _rhs_batch(Q, z, ii, jj, kk, ww, rho, lam, delta, u):
    B = Q.shape[0]
    J = Q.shape[1]
    conv = _conv_batch(Q, ii, jj, kk, ww)
    out = np.empty((B, J), dtype=np.complex128)
    for b in range(B):
        c = z[b] - 1.0
        for j in range(J):
            out[b, j] = lam * (conv[b, j] - Q[b, j]) - delta * Q[b, j] + u * c * rho[j] * Q[b, j]
        out[b, 0] += delta
    return out


@njit(cache=True)
def _rk4_batch(Q0, z, ii, jj, kk, ww, rho, lam, delta, u, T, n_steps):
    B = z.size
    J = Q0.size
    Q = np.empty((B, J), dtype=np.complex128)
    for b in range(B):
        for j in range(J):
            Q[b, j] = Q0[j]
    if n_steps <= 0 or T == 0.0:
        return Q
    dt = T / n_steps
    for _ in range(n_steps):
        k1 = _rhs_batch(Q, z, ii, jj, kk, ww, rho, lam, delta, u)
        k2 = _rhs_batch(Q + 0.5 * dt * k1, z, ii, jj, kk, ww, rho, lam, delta, u)
        k3 = _rhs_batch(Q + 0.5 * dt * k2, z, ii, jj, kk, ww, rho, lam, delta, u)
        k4 = _rhs_batch(Q + dt * k3, z, ii, jj, kk, ww, rho, lam, delta, u)
        Q = Q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return Q




@njit(cache=True)
def _rhs_batch_inplace(Q, z, ii, jj, kk, ww, rho, lam, delta, u, out, conv):
    B = Q.shape[0]
    J = Q.shape[1]
    E = kk.size
    # zero conv
    for b in range(B):
        for j in range(J):
            conv[b, j] = 0.0 + 0.0j
    for e in range(E):
        i = ii[e]
        j = jj[e]
        k = kk[e]
        w = ww[e]
        for b in range(B):
            conv[b, k] += w * Q[b, i] * Q[b, j]
    for b in range(B):
        c = z[b] - 1.0
        for j in range(J):
            out[b, j] = lam * (conv[b, j] - Q[b, j]) - delta * Q[b, j] + u * c * rho[j] * Q[b, j]
        out[b, 0] += delta


@njit(cache=True)
def _rk4_batch_inplace(Q0, z, ii, jj, kk, ww, rho, lam, delta, u, T, n_steps):
    B = z.size
    J = Q0.size
    Q = np.empty((B, J), dtype=np.complex128)
    for b in range(B):
        for j in range(J):
            Q[b, j] = Q0[j]
    if n_steps <= 0 or T == 0.0:
        return Q
    dt = T / n_steps
    k1 = np.empty((B, J), dtype=np.complex128)
    k2 = np.empty((B, J), dtype=np.complex128)
    k3 = np.empty((B, J), dtype=np.complex128)
    k4 = np.empty((B, J), dtype=np.complex128)
    tmp = np.empty((B, J), dtype=np.complex128)
    conv = np.empty((B, J), dtype=np.complex128)
    for _ in range(n_steps):
        _rhs_batch_inplace(Q, z, ii, jj, kk, ww, rho, lam, delta, u, k1, conv)
        for b in range(B):
            for j in range(J):
                tmp[b, j] = Q[b, j] + 0.5 * dt * k1[b, j]
        _rhs_batch_inplace(tmp, z, ii, jj, kk, ww, rho, lam, delta, u, k2, conv)
        for b in range(B):
            for j in range(J):
                tmp[b, j] = Q[b, j] + 0.5 * dt * k2[b, j]
        _rhs_batch_inplace(tmp, z, ii, jj, kk, ww, rho, lam, delta, u, k3, conv)
        for b in range(B):
            for j in range(J):
                tmp[b, j] = Q[b, j] + dt * k3[b, j]
        _rhs_batch_inplace(tmp, z, ii, jj, kk, ww, rho, lam, delta, u, k4, conv)
        c = dt / 6.0
        for b in range(B):
            for j in range(J):
                Q[b, j] = Q[b, j] + c * (k1[b, j] + 2.0 * k2[b, j] + 2.0 * k3[b, j] + k4[b, j])
    return Q


@njit(cache=True)
def _founder_dot_batch(Q, z, rho, theta_f, ext_prob, surv, condition_on_survival):
    B = z.size
    J = Q.shape[1]
    out = np.empty(B, dtype=np.complex128)
    for b in range(B):
        total = 0.0 + 0.0j
        for j in range(J):
            factor = np.exp(theta_f * (z[b] - 1.0) * rho[j])
            total += Q[b, j] * factor
        if condition_on_survival:
            total = (total - ext_prob) / surv
        out[b] = total
    return out


@njit(cache=True)
def _founder_dot_multi_theta_batch(Q, z, rho, theta_values, ext_prob, surv, condition_on_survival):
    B = z.size
    J = Q.shape[1]
    H = theta_values.size
    out = np.empty((H, B), dtype=np.complex128)
    for h in range(H):
        theta_f = theta_values[h]
        for b in range(B):
            total = 0.0 + 0.0j
            for j in range(J):
                factor = np.exp(theta_f * (z[b] - 1.0) * rho[j])
                total += Q[b, j] * factor
            if condition_on_survival:
                total = (total - ext_prob) / surv
            out[h, b] = total
    return out




class StableSolveIVPThetaPGFGridEngine:
    """Stable scalar solve_ivp PGF evaluator with theta bundling.

    This backend is intentionally slower than BatchedComplexPGFGridEngine, but it
    is much more stable on damped complex Cauchy nodes where fixed-step RK4 can
    blow up.  The ODE state Q(z) is integrated once per node and reused across
    all founder loads theta_f, preserving the exact theta-bundling identity.
    """

    def __init__(self, params: ModelParams, config: Optional[FourierCDFConfig] = None):
        self.params = params
        cfg = config or FourierCDFConfig(use_solve_ivp=True)
        self.config = FourierCDFConfig(
            n_bins=int(cfg.n_bins),
            steps_per_time=float(cfg.steps_per_time),
            min_steps=int(cfg.min_steps),
            max_steps=int(cfg.max_steps),
            quad_epsabs=float(cfg.quad_epsabs),
            quad_epsrel=float(cfg.quad_epsrel),
            quad_limit=int(cfg.quad_limit),
            clip_eps=float(cfg.clip_eps),
            addition_rule=str(cfg.addition_rule),
            representative=str(cfg.representative),
            ode_method=str(cfg.ode_method),
            ode_rtol=float(cfg.ode_rtol),
            ode_atol=float(cfg.ode_atol),
            use_solve_ivp=True,
        )
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
            raise RuntimeError("stable solve_ivp backend requires sparse linear/uniform addition arrays")
        self.rho = np.asarray(self.grid_engine.rho, dtype=np.float64)
        self.ii = np.asarray(self.grid_engine._uniform_i, dtype=np.int32)
        self.jj = np.asarray(self.grid_engine._uniform_j, dtype=np.int32)
        self.kk = np.asarray(self.grid_engine._uniform_k, dtype=np.int32)
        self.ww = np.asarray(self.grid_engine._uniform_w, dtype=np.float64)
        self.Q0 = np.asarray(self.grid_engine._initial_Q(), dtype=np.complex128)
        self.ext_prob = float(extinction_probability(params.lam, params.delta, params.T))
        self.surv = max(1.0 - self.ext_prob, np.finfo(float).tiny)
        self._Q_cache: dict[tuple[float, float], np.ndarray] = {}

    @property
    def n_classes(self) -> int:
        return int(self.Q0.size)

    def _conv(self, Q: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_classes, dtype=np.complex128)
        np.add.at(out, self.kk, self.ww * Q[self.ii] * Q[self.jj])
        return out

    def _rhs(self, Q: np.ndarray, z: complex) -> np.ndarray:
        p = self.params
        c = z - 1.0
        dQ = float(p.lam) * (self._conv(Q) - Q) - float(p.delta) * Q + float(p.u) * c * self.rho * Q
        dQ[0] += float(p.delta)
        return dQ

    def integrate_Q(self, z: complex) -> np.ndarray:
        key = (round(float(np.real(z)), 14), round(float(np.imag(z)), 14))
        if key in self._Q_cache:
            return self._Q_cache[key]
        Q0 = self.Q0.copy()
        T = float(self.params.T)
        if T == 0.0:
            Q = Q0
        else:
            def rhs_time(_t, y):
                return self._rhs(y, complex(z))
            sol = solve_ivp(
                rhs_time, (0.0, T), Q0, method=str(self.config.ode_method),
                rtol=float(self.config.ode_rtol), atol=float(self.config.ode_atol)
            )
            if (not sol.success) or (not np.isfinite(sol.y[:, -1]).all()):
                Q = np.full_like(Q0, complex(np.nan, np.nan))
            else:
                Q = np.asarray(sol.y[:, -1], dtype=np.complex128)
        self._Q_cache[key] = Q
        return Q

    def pgf_many_theta(self, z_values: np.ndarray, theta_values: np.ndarray) -> np.ndarray:
        z_values = np.asarray(z_values, dtype=np.complex128)
        theta_values = np.asarray(theta_values, dtype=np.float64)
        out = np.empty((theta_values.size, z_values.size), dtype=np.complex128)
        for b, z in enumerate(z_values):
            Q = self.integrate_Q(complex(z))
            if not np.isfinite(Q).all():
                out[:, b] = complex(np.nan, np.nan)
                continue
            for h, theta_f in enumerate(theta_values):
                factor = np.exp(float(theta_f) * (complex(z) - 1.0) * self.rho)
                total = complex(np.dot(Q, factor))
                if self.params.condition_on_survival:
                    total = (total - self.ext_prob) / self.surv
                out[h, b] = total
        return out

    def pgf_many(self, z_values: np.ndarray) -> np.ndarray:
        vals = self.pgf_many_theta(z_values, np.array([float(self.params.theta_f)], dtype=np.float64))
        return vals[0]


@dataclass(frozen=True)
class BatchedPGFConfig:
    batch_size: int = 128


class BatchedComplexPGFGridEngine:
    """Evaluate PGF nodes in batches using compiled fixed-step RK4."""

    def __init__(self, params: ModelParams, config: Optional[FourierCDFConfig] = None, *, batch_config: Optional[BatchedPGFConfig] = None):
        self.params = params
        self.config = config or FourierCDFConfig(use_solve_ivp=False)
        self.batch_config = batch_config or BatchedPGFConfig()
        if bool(self.config.use_solve_ivp):
            raise ValueError("BatchedComplexPGFGridEngine requires use_solve_ivp=False")
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
            raise RuntimeError("batched backend requires sparse linear/uniform addition arrays")
        self.rho = np.asarray(self.grid_engine.rho, dtype=np.float64)
        self.ii = np.asarray(self.grid_engine._uniform_i, dtype=np.int32)
        self.jj = np.asarray(self.grid_engine._uniform_j, dtype=np.int32)
        self.kk = np.asarray(self.grid_engine._uniform_k, dtype=np.int32)
        self.ww = np.asarray(self.grid_engine._uniform_w, dtype=np.float64)
        self.Q0 = np.asarray(self.grid_engine._initial_Q(), dtype=np.complex128)
        self.ext_prob = float(extinction_probability(params.lam, params.delta, params.T))
        self.surv = max(1.0 - self.ext_prob, np.finfo(float).tiny)
        self.n_steps = int(min(max(int(math.ceil(float(params.T) * float(self.config.steps_per_time))), int(self.config.min_steps)), int(self.config.max_steps)))

    @property
    def n_classes(self) -> int:
        return int(self.Q0.size)

    def integrate_Q_many(self, z_values: np.ndarray) -> np.ndarray:
        z_values = np.asarray(z_values, dtype=np.complex128)
        out = np.empty((z_values.size, self.n_classes), dtype=np.complex128)
        bs = max(1, int(self.batch_config.batch_size))
        for start in range(0, z_values.size, bs):
            stop = min(start + bs, z_values.size)
            z = np.ascontiguousarray(z_values[start:stop], dtype=np.complex128)
            out[start:stop, :] = _rk4_batch_inplace(
                self.Q0, z, self.ii, self.jj, self.kk, self.ww, self.rho,
                float(self.params.lam), float(self.params.delta), float(self.params.u),
                float(self.params.T), int(self.n_steps),
            )
        return out

    def pgf_many(self, z_values: np.ndarray) -> np.ndarray:
        z_values = np.asarray(z_values, dtype=np.complex128)
        out = np.empty(z_values.size, dtype=np.complex128)
        bs = max(1, int(self.batch_config.batch_size))
        for start in range(0, z_values.size, bs):
            stop = min(start + bs, z_values.size)
            z = np.ascontiguousarray(z_values[start:stop], dtype=np.complex128)
            Q = _rk4_batch_inplace(
                self.Q0, z, self.ii, self.jj, self.kk, self.ww, self.rho,
                float(self.params.lam), float(self.params.delta), float(self.params.u),
                float(self.params.T), int(self.n_steps),
            )
            out[start:stop] = _founder_dot_batch(
                Q, z, self.rho, float(self.params.theta_f), float(self.ext_prob), float(self.surv), bool(self.params.condition_on_survival)
            )
        return out

    def pgf_many_theta(self, z_values: np.ndarray, theta_values: np.ndarray) -> np.ndarray:
        """Return PGF values with shape (n_theta, n_nodes), reusing Q integration.

        The finite-depth ODE depends on z and alpha but not theta_f; founder load
        enters only through the final Poisson thinning factor.  This routine is
        therefore the main theta-bundling speed win for grids with many founder loads.
        """
        z_values = np.asarray(z_values, dtype=np.complex128)
        theta_values = np.asarray(theta_values, dtype=np.float64)
        out = np.empty((theta_values.size, z_values.size), dtype=np.complex128)
        bs = max(1, int(self.batch_config.batch_size))
        for start in range(0, z_values.size, bs):
            stop = min(start + bs, z_values.size)
            z = np.ascontiguousarray(z_values[start:stop], dtype=np.complex128)
            Q = _rk4_batch_inplace(
                self.Q0, z, self.ii, self.jj, self.kk, self.ww, self.rho,
                float(self.params.lam), float(self.params.delta), float(self.params.u),
                float(self.params.T), int(self.n_steps),
            )
            out[:, start:stop] = _founder_dot_multi_theta_batch(
                Q, z, self.rho, theta_values, float(self.ext_prob), float(self.surv), bool(self.params.condition_on_survival)
            )
        return out
