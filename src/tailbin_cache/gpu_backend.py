from __future__ import annotations

"""Optional CuPy GPU backend for contour-node finite-depth PGF evaluation.

This backend is intentionally optional: importing tailbin_cache does not require
CuPy.  On an O2 GPU node with CUDA/CuPy available, set ``pgf_backend: cupy`` in
the build config to evaluate many complex Cauchy nodes on the GPU.  The
mathematical target is identical to the CPU fixed-step RK4 backend; only the
execution device changes.

The implementation mirrors BatchedComplexPGFGridEngine but uses array-level GPU
operations over the node batch.  It is designed as a first production benchmark
backend: correctness is still certified by the existing base/refined Cauchy
ladder and by optional CPU-vs-GPU audit runs.
"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from readsampled_cdf.distribution import ModelParams, extinction_probability
from readsampled_cdf.frequency_volume import AdaptiveFrequencyVolumeEngine
from readsampled_cdf.fast_frequency import FastFrequencyConfig, make_fast_grid_config
from readsampled_cdf.fourier_reference import FourierCDFConfig


@dataclass(frozen=True)
class CuPyPGFConfig:
    batch_size: int = 4096
    # Use complex128 by default.  Mixed precision can be added only after
    # CPU/refined audits show it is safe for the target z-error budget.
    dtype: str = "complex128"


class CuPyComplexPGFGridEngine:
    """Evaluate finite-depth PGF contour nodes on a CUDA GPU using CuPy.

    The expensive ODE state Q(z) is advanced for a batch of complex z nodes in
    parallel.  Founder loads theta_f are evaluated after integration, reusing
    the same Q(z) for all theta values exactly as in the CPU theta-bundled
    backend.
    """

    def __init__(self, params: ModelParams, config: Optional[FourierCDFConfig] = None, *, gpu_config: Optional[CuPyPGFConfig] = None):
        try:
            import cupy as cp  # type: ignore
        except Exception as e:  # pragma: no cover - exercised on GPU nodes only
            raise RuntimeError("pgf_backend='cupy' requires CuPy and a CUDA GPU") from e
        self.cp = cp
        self.params = params
        self.config = config or FourierCDFConfig(use_solve_ivp=False)
        self.gpu_config = gpu_config or CuPyPGFConfig()
        if bool(self.config.use_solve_ivp):
            raise ValueError("CuPyComplexPGFGridEngine requires use_solve_ivp=False")
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
            raise RuntimeError("CuPy backend requires sparse linear/uniform addition arrays")
        self.rho_np = np.asarray(self.grid_engine.rho, dtype=np.float64)
        self.ii_np = np.asarray(self.grid_engine._uniform_i, dtype=np.int32)
        self.jj_np = np.asarray(self.grid_engine._uniform_j, dtype=np.int32)
        self.kk_np = np.asarray(self.grid_engine._uniform_k, dtype=np.int32)
        self.ww_np = np.asarray(self.grid_engine._uniform_w, dtype=np.float64)
        self.Q0_np = np.asarray(self.grid_engine._initial_Q(), dtype=np.complex128)
        self.ext_prob = float(extinction_probability(params.lam, params.delta, params.T))
        self.surv = max(1.0 - self.ext_prob, np.finfo(float).tiny)
        self.n_steps = int(min(max(int(math.ceil(float(params.T) * float(self.config.steps_per_time))), int(self.config.min_steps)), int(self.config.max_steps)))
        # Device constants.
        cp = self.cp
        self.rho = cp.asarray(self.rho_np, dtype=cp.float64)
        self.ii = cp.asarray(self.ii_np, dtype=cp.int32)
        self.jj = cp.asarray(self.jj_np, dtype=cp.int32)
        self.kk = cp.asarray(self.kk_np, dtype=cp.int32)
        self.ww = cp.asarray(self.ww_np, dtype=cp.float64)
        self.Q0 = cp.asarray(self.Q0_np, dtype=cp.complex128)
        self._row_index_cache: dict[int, object] = {}

    @property
    def n_classes(self) -> int:
        return int(self.Q0_np.size)

    def _row_index(self, B: int):
        if B not in self._row_index_cache:
            self._row_index_cache[B] = self.cp.arange(B, dtype=self.cp.int32)[:, None]
        return self._row_index_cache[B]

    def _conv(self, Q):
        cp = self.cp
        B = int(Q.shape[0])
        J = int(Q.shape[1])
        out = cp.zeros((B, J), dtype=cp.complex128)
        # vals shape: B x E.  Scatter add along destination class kk.
        vals = Q[:, self.ii] * Q[:, self.jj] * self.ww[None, :]
        cp.add.at(out, (self._row_index(B), self.kk[None, :]), vals)
        return out

    def _rhs(self, Q, z):
        p = self.params
        conv = self._conv(Q)
        c = z[:, None] - 1.0
        out = float(p.lam) * (conv - Q) - float(p.delta) * Q + float(p.u) * c * self.rho[None, :] * Q
        out[:, 0] = out[:, 0] + float(p.delta)
        return out

    def _integrate_Q_batch(self, z):
        cp = self.cp
        B = int(z.size)
        Q = cp.broadcast_to(self.Q0[None, :], (B, self.n_classes)).copy()
        if self.n_steps <= 0 or float(self.params.T) == 0.0:
            return Q
        dt = float(self.params.T) / float(self.n_steps)
        for _ in range(int(self.n_steps)):
            k1 = self._rhs(Q, z)
            k2 = self._rhs(Q + 0.5 * dt * k1, z)
            k3 = self._rhs(Q + 0.5 * dt * k2, z)
            k4 = self._rhs(Q + dt * k3, z)
            Q = Q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return Q

    def pgf_many_theta(self, z_values: np.ndarray, theta_values: np.ndarray) -> np.ndarray:
        cp = self.cp
        z_values = np.asarray(z_values, dtype=np.complex128)
        theta_values = np.asarray(theta_values, dtype=np.float64)
        out_np = np.empty((theta_values.size, z_values.size), dtype=np.complex128)
        bs = max(1, int(self.gpu_config.batch_size))
        theta_gpu = cp.asarray(theta_values, dtype=cp.float64)
        for start in range(0, z_values.size, bs):
            stop = min(start + bs, z_values.size)
            z = cp.asarray(np.ascontiguousarray(z_values[start:stop], dtype=np.complex128), dtype=cp.complex128)
            Q = self._integrate_Q_batch(z)
            # factors shape H x B x J; sum over classes.  This can be memory-heavy
            # for very large H/B/J, so batch size should be tuned on O2.
            factors = cp.exp(theta_gpu[:, None, None] * (z[None, :, None] - 1.0) * self.rho[None, None, :])
            vals = cp.sum(Q[None, :, :] * factors, axis=2)
            if bool(self.params.condition_on_survival):
                vals = (vals - float(self.ext_prob)) / float(self.surv)
            out_np[:, start:stop] = cp.asnumpy(vals)
        return out_np

    def pgf_many(self, z_values: np.ndarray) -> np.ndarray:
        vals = self.pgf_many_theta(z_values, np.asarray([float(self.params.theta_f)], dtype=np.float64))
        return vals[0]
