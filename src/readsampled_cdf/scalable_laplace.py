"""v25 scalable read-depth candidate engines.

This module is deliberately conservative: it introduces a population-size-free
Laplace-domain candidate for fixed-depth read sampling, plus validation helpers
that compare it with the exact coefficient engine when small-N validation is
possible.

The mathematical target is the Feynman--Kac branching transform

    Q(t,b;s) = E[ exp(-b N_t/N) exp(s Z_t) ],

where Z_t is the read-sampled mutation count and the read probability is

    rho(p) = P[Binomial(depth, min(p,1)) >= ceil(alpha*depth)].

If rho is approximated by an exponential sum

    rho(p) ~= c_0 + sum_j c_j exp(-a_j p),

then multiplication by rho in terminal-frequency space becomes shifted
Laplace evaluations Q(b+a_j).  The birth/death part is pointwise in b:

    dQ/dt = lambda(Q^2-Q) + delta(1-Q) + u(exp(s)-1) Rho[Q].

This avoids both terminal-population truncation and coefficient convolution.
It is an exact-ish / convergent route: increasing the exponential approximation
and b-grid resolution should converge in regimes where the approximation is
well conditioned.  The module exposes diagnostics and refuses to call itself
certified when the fits are numerically poor.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import bdtr
from scipy.stats import norm

from .distribution import ModelParams, ExactSeriesEngine, extinction_probability, fixed_depth_rho_f


@dataclass(frozen=True)
class ExpSumFit:
    """Represent f(p) ~= coeffs[0] + sum_j coeffs[j+1] exp(-rates[j] p)."""

    rates: np.ndarray
    coeffs: np.ndarray
    p_max: float
    max_abs_error: float
    rms_error: float
    coeff_l1: float
    coeff_max_abs: float
    condition_estimate: float
    family: str
    ridge: float

    def __call__(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        flat = p.reshape(-1)
        y = np.full(flat.shape, float(self.coeffs[0]))
        if self.rates.size:
            y += np.exp(-np.outer(flat, self.rates)).dot(self.coeffs[1:])
        return y.reshape(p.shape)

    def diagnostic_dict(self, prefix: str = "fit") -> Dict[str, float | str]:
        return {
            f"{prefix}_family": self.family,
            f"{prefix}_p_max": float(self.p_max),
            f"{prefix}_max_abs_error": float(self.max_abs_error),
            f"{prefix}_rms_error": float(self.rms_error),
            f"{prefix}_coeff_l1": float(self.coeff_l1),
            f"{prefix}_coeff_max_abs": float(self.coeff_max_abs),
            f"{prefix}_condition_estimate": float(self.condition_estimate),
            f"{prefix}_n_terms": float(self.rates.size),
            f"{prefix}_ridge": float(self.ridge),
        }


@dataclass(frozen=True)
class ScalableLaplaceConfig:
    """Numerical settings for ScalableLaplaceReadDepthEngine."""

    n_exp_terms: int = 80
    n_b_nodes: int = 260
    fit_family: str = "geom"  # "geom" or "polyexp"
    fit_p_max: float = 2.0
    fit_ridge: float = 1e-7
    fit_alpha_weight: float = 12.0
    fit_lowp_weight: float = 2.0
    b_cascade_depth: float = 4.0
    ode_method: str = "DOP853"
    rtol: float = 2e-6
    atol_factor: float = 1e-2
    max_rho_fit_abs_error_certified: float = 2e-4
    max_coeff_l1_certified: float = 2e4
    max_founder_fit_abs_error_certified: float = 2e-4
    founder_fit_terms_multiplier: float = 1.25


def read_threshold(alpha: float, depth: int) -> int:
    return int(math.ceil(float(alpha) * int(depth) - 1e-14))


def read_depth_rho(p: np.ndarray, alpha: float, depth: int) -> np.ndarray:
    p = np.minimum(np.maximum(np.asarray(p, dtype=float), 0.0), 1.0)
    c = read_threshold(alpha, depth)
    if c <= 0:
        return np.ones_like(p)
    if c > depth:
        return np.zeros_like(p)
    return 1.0 - bdtr(c - 1, int(depth), p)


def _fit_grid(alpha: float, depth: int, p_max: float, n_base: int = 2600,
              alpha_weight: float = 12.0, lowp_weight: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    alpha = float(alpha)
    depth = int(depth)
    p_max = float(p_max)
    base = np.linspace(0.0, p_max, int(n_base))
    width = max(1e-4, 8.0 * math.sqrt(max(alpha * (1.0 - alpha), 1e-12) / max(depth, 1)))
    local = np.linspace(max(0.0, alpha - width), min(p_max, alpha + width), max(600, n_base // 2))
    tiny_hi = min(max(alpha, 1e-6), p_max)
    tiny = np.geomspace(1e-12, max(tiny_hi, 1e-6), 360)
    p = np.unique(np.concatenate(([0.0, 1.0, p_max], base, local, tiny)))
    w = np.ones_like(p)
    # Emphasize the read-depth transition and tiny p values, where clone sizes
    # are common for large N.
    w += float(alpha_weight) * np.exp(-0.5 * ((p - alpha) / max(width / 5.0, 1e-4)) ** 2)
    w += float(lowp_weight) * np.exp(-p / max(alpha, 1e-3))
    return p, w


def _rates(family: str, n_terms: int, alpha: float, p_max: float) -> np.ndarray:
    n_terms = int(n_terms)
    if n_terms <= 0:
        return np.zeros(0, dtype=float)
    family = str(family).lower()
    if family == "geom":
        # Broad dictionary.  The upper scale must resolve steep D=90..200 read
        # transitions at low alpha.
        a_min = 1e-4
        a_max = max(500.0, 120.0 / max(float(alpha), 1e-3))
        return np.geomspace(a_min, a_max, n_terms)
    if family == "polyexp":
        # Powers of exp(-gamma p).  This is often better conditioned for low
        # thresholds and gives a simple convergence ladder.
        gamma = max(0.5, min(12.0, 0.25 / max(float(alpha), 1e-3)))
        return gamma * np.arange(1, n_terms + 1, dtype=float)
    raise ValueError("fit_family must be 'geom' or 'polyexp'")


def fit_exponential_sum_for_function(
    func,
    *,
    alpha: float,
    depth: int,
    config: ScalableLaplaceConfig,
    n_terms: Optional[int] = None,
    rates: Optional[np.ndarray] = None,
    family: Optional[str] = None,
    ridge: Optional[float] = None,
) -> ExpSumFit:
    """Fit func(p) by a constant plus exponentials using weighted ridge LS."""
    fam = str(family or config.fit_family)
    rr = np.asarray(rates if rates is not None else _rates(fam, int(n_terms or config.n_exp_terms), alpha, config.fit_p_max), dtype=float)
    ridge_val = float(config.fit_ridge if ridge is None else ridge)
    p, w = _fit_grid(alpha, depth, config.fit_p_max, alpha_weight=config.fit_alpha_weight,
                     lowp_weight=config.fit_lowp_weight)
    y = np.asarray(func(p), dtype=float)
    A = np.ones((p.size, rr.size + 1), dtype=float)
    if rr.size:
        A[:, 1:] = np.exp(-np.outer(p, rr))
    sw = np.sqrt(w)
    Aw = A * sw[:, None]
    yw = y * sw
    if ridge_val > 0.0:
        reg = math.sqrt(ridge_val) * np.eye(A.shape[1])
        # Avoid biasing the asymptotic constant too strongly.
        reg[0, 0] *= 1e-3
        Aw = np.vstack([Aw, reg])
        yw = np.concatenate([yw, np.zeros(A.shape[1])])
    coeffs, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    pred = A.dot(coeffs)
    err = pred - y
    # Condition estimate of the weighted design, intentionally capped to avoid
    # inf-only diagnostics.
    try:
        svals = np.linalg.svd(Aw[:A.shape[0]], compute_uv=False)
        cond = float(svals[0] / max(svals[-1], np.finfo(float).tiny))
    except Exception:
        cond = math.inf
    return ExpSumFit(
        rates=rr,
        coeffs=coeffs,
        p_max=float(config.fit_p_max),
        max_abs_error=float(np.max(np.abs(err))),
        rms_error=float(np.sqrt(np.mean(err * err))),
        coeff_l1=float(np.sum(np.abs(coeffs))),
        coeff_max_abs=float(np.max(np.abs(coeffs))) if coeffs.size else 0.0,
        condition_estimate=cond,
        family=fam,
        ridge=ridge_val,
    )


def fit_read_depth_rho(alpha: float, depth: int, config: ScalableLaplaceConfig) -> ExpSumFit:
    def f(p):
        return read_depth_rho(p, alpha, depth)
    return fit_exponential_sum_for_function(f, alpha=alpha, depth=depth, config=config)


def _make_b_grid(max_rate: float, n_nodes: int, cascade_depth: float) -> np.ndarray:
    max_rate = max(float(max_rate), 1.0)
    b_max = float(cascade_depth) * max_rate
    # Use a hybrid grid.  Many reads of Q(b+a) occur near fit rates, so include
    # all rate points and a smooth envelope grid.
    near = np.geomspace(1e-10, max_rate, max(32, int(n_nodes) // 3))
    lin = np.linspace(0.0, b_max, max(64, int(n_nodes)))
    return np.unique(np.concatenate(([0.0], near, lin))).astype(float)


class ScalableLaplaceReadDepthEngine:
    """Population-size-free candidate K(s) engine for fixed read depth.

    This is currently a candidate/experimental engine.  The metadata reports a
    certification flag based on representation quality and ODE success.  Use the
    exact coefficient engine for small-N validation.
    """

    def __init__(self, params: ModelParams, *, config: Optional[ScalableLaplaceConfig] = None,
                 rho_fit: Optional[ExpSumFit] = None):
        self.params = params
        self.config = config or ScalableLaplaceConfig()
        self.rho_fit = rho_fit or fit_read_depth_rho(params.alpha, params.depth, self.config)
        max_rate = float(np.max(self.rho_fit.rates)) if self.rho_fit.rates.size else 1.0
        self.b = _make_b_grid(max_rate, self.config.n_b_nodes, self.config.b_cascade_depth)
        self.ext_prob = extinction_probability(params.lam, params.delta, params.T)
        self._K_cache: Dict[Tuple[float, float, str], Tuple[Tuple[float, float, float], Dict[str, float]]] = {}

    def fit_quality_ok(self) -> bool:
        cfg = self.config
        return (
            np.isfinite(self.rho_fit.max_abs_error)
            and self.rho_fit.max_abs_error <= cfg.max_rho_fit_abs_error_certified
            and self.rho_fit.coeff_l1 <= cfg.max_coeff_l1_certified
        )

    def _interp(self, arr: np.ndarray, x: np.ndarray, right_value: float) -> np.ndarray:
        return np.interp(x, self.b, arr, left=float(arr[0]), right=float(right_value))

    def _apply_fit(self, fit: ExpSumFit, arr: np.ndarray, right_value: float) -> np.ndarray:
        out = float(fit.coeffs[0]) * arr.copy()
        for rate, coeff in zip(fit.rates, fit.coeffs[1:]):
            if coeff != 0.0:
                out += float(coeff) * self._interp(arr, self.b + float(rate), right_value)
        return out

    def _apply_fit_at_zero(self, fit: ExpSumFit, arr: np.ndarray, right_value: float) -> float:
        val = float(fit.coeffs[0]) * float(arr[0])
        for rate, coeff in zip(fit.rates, fit.coeffs[1:]):
            if coeff != 0.0:
                val += float(coeff) * float(self._interp(arr, np.array([float(rate)]), right_value)[0])
        return float(val)

    def _solve_QAB(self, s: float, rtol: Optional[float], ode_method: Optional[str]):
        p = self.params
        cfg = self.config
        b = self.b
        nb = b.size
        rtol_val = float(cfg.rtol if rtol is None else rtol)
        atol = max(np.finfo(float).eps, rtol_val * cfg.atol_factor)
        method = ode_method or cfg.ode_method
        es = math.exp(float(s))
        c = es - 1.0
        Q0 = np.exp(-b / float(p.N_const))
        A0 = np.zeros_like(Q0)
        B0 = np.zeros_like(Q0)
        y0 = np.concatenate([Q0, A0, B0])

        def rhs(t, flat):
            Q = flat[:nb]
            A = flat[nb:2*nb]
            B = flat[2*nb:]
            ext_t = extinction_probability(p.lam, p.delta, float(t))
            RQ = self._apply_fit(self.rho_fit, Q, right_value=ext_t)
            RA = self._apply_fit(self.rho_fit, A, right_value=0.0)
            RB = self._apply_fit(self.rho_fit, B, right_value=0.0)
            dQ = p.lam * (Q * Q - Q) + p.delta * (1.0 - Q) + p.u * c * RQ
            dA = p.lam * (2.0 * Q * A - A) - p.delta * A + p.u * (es * RQ + c * RA)
            dB = p.lam * (2.0 * A * A + 2.0 * Q * B - B) - p.delta * B + p.u * (es * RQ + 2.0 * es * RA + c * RB)
            return np.concatenate([dQ, dA, dB])

        sol = solve_ivp(rhs, (0.0, p.T), y0, method=method, rtol=rtol_val, atol=atol)
        Y = sol.y[:, -1]
        return Y[:nb], Y[nb:2*nb], Y[2*nb:], sol

    def _fit_founder(self, s: float) -> Tuple[ExpSumFit, ExpSumFit, ExpSumFit, float]:
        p = self.params
        cfg = self.config
        es = math.exp(float(s))
        c = es - 1.0
        theta = float(p.theta_f)
        log_scale = max(0.0, theta * c)
        n_terms = max(8, int(math.ceil(cfg.n_exp_terms * cfg.founder_fit_terms_multiplier)))
        rates = _rates(cfg.fit_family, n_terms, p.alpha, cfg.fit_p_max)

        def rho(x):
            return read_depth_rho(x, p.alpha, p.depth)

        def f0(x):
            rr = rho(x)
            return np.exp(theta * c * rr - log_scale)

        def f1(x):
            rr = rho(x)
            gp = theta * es * rr
            return gp * np.exp(theta * c * rr - log_scale)

        def f2(x):
            rr = rho(x)
            gp = theta * es * rr
            return (gp + gp * gp) * np.exp(theta * c * rr - log_scale)

        # These fits are intentionally diagnosed separately; they are often the
        # limiting factor for theta_f near 2000 and positive saddlepoints.
        f0fit = fit_exponential_sum_for_function(f0, alpha=p.alpha, depth=p.depth, config=cfg,
                                                 n_terms=n_terms, rates=rates, ridge=max(cfg.fit_ridge, 1e-7))
        f1fit = fit_exponential_sum_for_function(f1, alpha=p.alpha, depth=p.depth, config=cfg,
                                                 n_terms=n_terms, rates=rates, ridge=max(cfg.fit_ridge, 1e-7))
        f2fit = fit_exponential_sum_for_function(f2, alpha=p.alpha, depth=p.depth, config=cfg,
                                                 n_terms=n_terms, rates=rates, ridge=max(cfg.fit_ridge, 1e-7))
        return f0fit, f1fit, f2fit, float(log_scale)

    def K012(self, s: float, rtol: Optional[float] = None, ode_method: Optional[str] = None):
        key = (round(float(s), 12), float(self.config.rtol if rtol is None else rtol), ode_method or self.config.ode_method)
        if key in self._K_cache:
            vals, meta = self._K_cache[key]
            return vals[0], vals[1], vals[2], {**meta, "cached": 1.0}
        t0 = time.perf_counter()
        p = self.params
        Q, A, B, sol = self._solve_QAB(float(s), rtol, ode_method)
        founder_scale = 0.0
        founder_maxerr = 0.0
        founder_l1 = 0.0
        founder_ok = True
        if p.theta_f:
            f0, f1, f2, founder_scale = self._fit_founder(float(s))
            G0 = self._apply_fit_at_zero(f0, Q, right_value=self.ext_prob)
            G1 = self._apply_fit_at_zero(f0, A, right_value=0.0) + self._apply_fit_at_zero(f1, Q, right_value=0.0)
            G2 = (self._apply_fit_at_zero(f0, B, right_value=0.0)
                  + 2.0 * self._apply_fit_at_zero(f1, A, right_value=0.0)
                  + self._apply_fit_at_zero(f2, Q, right_value=0.0))
            founder_maxerr = float(max(f0.max_abs_error, f1.max_abs_error, f2.max_abs_error))
            founder_l1 = float(max(f0.coeff_l1, f1.coeff_l1, f2.coeff_l1))
            founder_ok = founder_maxerr <= self.config.max_founder_fit_abs_error_certified and founder_l1 <= self.config.max_coeff_l1_certified
        else:
            G0, G1, G2 = float(Q[0]), float(A[0]), float(B[0])
        if p.condition_on_survival:
            den_surv = max(1.0 - self.ext_prob, np.finfo(float).tiny)
            G0c = G0 - self.ext_prob * math.exp(-founder_scale)
            G0c = max(float(G0c), np.finfo(float).tiny)
            K0 = founder_scale + math.log(G0c) - math.log(den_surv)
        else:
            G0c = max(float(G0), np.finfo(float).tiny)
            K0 = founder_scale + math.log(G0c)
        K1 = float(G1 / G0c) if np.isfinite(G1) and G0c > 0 else math.nan
        K2raw = float(G2 / G0c - K1 * K1) if np.isfinite(G2) and np.isfinite(K1) and G0c > 0 else math.nan
        K2 = float(max(K2raw, 1e-14)) if np.isfinite(K2raw) else math.nan
        certified = bool(self.fit_quality_ok() and founder_ok and sol.success and np.isfinite(K0) and np.isfinite(K1) and np.isfinite(K2))
        meta: Dict[str, float | str] = {
            "engine": "scalable_laplace_read_depth_v25_candidate",
            "seconds": float(time.perf_counter() - t0),
            "ode_success": float(bool(sol.success)),
            "nfev": float(sol.nfev),
            "n_b_nodes": float(self.b.size),
            "N_const": float(p.N_const),
            "depth": float(p.depth),
            "alpha": float(p.alpha),
            "theta_f": float(p.theta_f),
            "founder_log_scale": float(founder_scale),
            "founder_fit_max_abs_error": float(founder_maxerr),
            "founder_fit_coeff_l1": float(founder_l1),
            "fit_quality_ok": float(self.fit_quality_ok()),
            "founder_fit_quality_ok": float(founder_ok),
            "certified_candidate": float(certified),
            "cached": 0.0,
        }
        meta.update(self.rho_fit.diagnostic_dict("rho"))
        # Compatibility aliases for validation tables and downstream CSVs.
        meta["rho_fit_max_abs_error"] = float(self.rho_fit.max_abs_error)
        meta["rho_fit_rms_error"] = float(self.rho_fit.rms_error)
        meta["rho_fit_coeff_l1"] = float(self.rho_fit.coeff_l1)
        vals = (float(K0), float(K1), float(K2))
        self._K_cache[key] = (vals, meta)
        return vals[0], vals[1], vals[2], meta

    def cdf_lr(self, k: int, *, continuity: float = 0.5, rtol: Optional[float] = None,
               max_abs_s: float = 12.0) -> Tuple[float, Dict[str, float]]:
        """Internal saddlepoint helper retained for compatibility; not used by cache-builder production CLI."""
        t0 = time.perf_counter()
        x = max(float(k) + float(continuity), 0.0)
        _, mu, var, m0 = self.K012(0.0, rtol=rtol)
        sigma = math.sqrt(max(float(var), 1e-14))
        if not np.isfinite(mu) or not np.isfinite(var):
            return math.nan, {"method": "failed_nonfinite_moments", "seconds": time.perf_counter() - t0, **m0}
        if abs(x - mu) < 1e-7 * max(1.0, sigma):
            return float(norm.cdf((x - mu) / sigma)), {"method": "normal_at_mean", "mean": mu, "var": var, "seconds": time.perf_counter()-t0, **m0}

        def f(ss: float) -> float:
            return self.K012(float(ss), rtol=rtol)[1] - x

        lo, hi = -0.125, 0.125
        flo, fhi = f(lo), f(hi)
        while np.isfinite(flo) and flo > 0.0 and abs(lo) < max_abs_s:
            lo *= 2.0; flo = f(lo)
        while np.isfinite(fhi) and fhi < 0.0 and abs(hi) < max_abs_s:
            hi *= 2.0; fhi = f(hi)
        if (not np.isfinite(flo)) or (not np.isfinite(fhi)) or flo > 0.0 or fhi < 0.0:
            F = float(norm.cdf((x - mu) / sigma))
            return F, {"method": "normal_bracket_fallback", "mean": mu, "var": var, "bracket_flo": float(flo), "bracket_fhi": float(fhi), "seconds": time.perf_counter()-t0, **m0}
        root = brentq(f, lo, hi, xtol=1e-9, rtol=1e-9, maxiter=80)
        K, _, K2, meta = self.K012(root, rtol=rtol)
        rad = max(2.0 * (root * x - K), 1e-300)
        w = (1.0 if root >= 0 else -1.0) * math.sqrt(rad)
        u = root * math.sqrt(max(K2, 1e-300))
        if abs(root) < 1e-8 or abs(w) < 1e-10 or abs(u) < 1e-10:
            F = float(norm.cdf((x - mu) / sigma)); method = "normal_singular"
        else:
            F = float(norm.cdf(w) + norm.pdf(w) * (1.0 / w - 1.0 / u)); method = "lugannani_rice"
        return float(np.clip(F, 0.0, 1.0)), {"method": method, "s_hat": float(root), "mean": mu, "var": var, "seconds": time.perf_counter()-t0, **meta}


def validate_scalable_against_exact(
    params: ModelParams,
    *,
    M: int,
    s_values: Sequence[float] = (0.0, 0.01, 0.05),
    config: Optional[ScalableLaplaceConfig] = None,
    exact_rtol: float = 1e-7,
    scalable_rtol: Optional[float] = None,
) -> List[Dict[str, float | str]]:
    """Compare v25 scalable K012 against the exact coefficient engine."""
    exact = ExactSeriesEngine(params, M=int(M))
    approx = ScalableLaplaceReadDepthEngine(params, config=config)
    rows: List[Dict[str, float | str]] = []
    for s in s_values:
        eK, eK1, eK2, em = exact.K012(float(s), rtol=exact_rtol, ode_method="DOP853")
        aK, aK1, aK2, am = approx.K012(float(s), rtol=scalable_rtol)
        rows.append({
            "s": float(s),
            "exact_K": float(eK), "approx_K": float(aK), "abs_err_K": float(abs(aK-eK)),
            "exact_K1": float(eK1), "approx_K1": float(aK1), "abs_err_K1": float(abs(aK1-eK1)),
            "exact_K2": float(eK2), "approx_K2": float(aK2), "abs_err_K2": float(abs(aK2-eK2)),
            "approx_certified_candidate": float(am.get("certified_candidate", 0.0)),
            "rho_fit_max_abs_error": float(am.get("rho_fit_max_abs_error", math.nan)),
            "founder_fit_max_abs_error": float(am.get("founder_fit_max_abs_error", math.nan)),
            "approx_seconds": float(am.get("seconds", math.nan)),
        })
    return rows


def convergence_ladder(
    params: ModelParams,
    *,
    term_grid: Sequence[int] = (32, 48, 64, 96),
    s_values: Sequence[float] = (0.0, 0.01),
    base_config: Optional[ScalableLaplaceConfig] = None,
) -> List[Dict[str, float | str]]:
    """Run a representation/grid convergence ladder for K012 values."""
    base = base_config or ScalableLaplaceConfig()
    out: List[Dict[str, float | str]] = []
    prev: Dict[float, Tuple[float, float, float]] = {}
    for n in term_grid:
        cfg = ScalableLaplaceConfig(**{**asdict(base), "n_exp_terms": int(n), "n_b_nodes": max(base.n_b_nodes, 3*int(n))})
        eng = ScalableLaplaceReadDepthEngine(params, config=cfg)
        for s in s_values:
            K0, K1, K2, meta = eng.K012(float(s))
            pk = prev.get(float(s))
            out.append({
                "n_exp_terms": float(n), "n_b_nodes": float(eng.b.size), "s": float(s),
                "K": float(K0), "K1": float(K1), "K2": float(K2),
                "delta_K_vs_prev": float(abs(K0-pk[0])) if pk else math.nan,
                "delta_K1_vs_prev": float(abs(K1-pk[1])) if pk else math.nan,
                "delta_K2_vs_prev": float(abs(K2-pk[2])) if pk else math.nan,
                "rho_fit_max_abs_error": float(meta.get("rho_fit_max_abs_error", math.nan)),
                "founder_fit_max_abs_error": float(meta.get("founder_fit_max_abs_error", math.nan)),
                "certified_candidate": float(meta.get("certified_candidate", 0.0)),
            })
            prev[float(s)] = (K0, K1, K2)
    return out
