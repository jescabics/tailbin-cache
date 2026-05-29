"""Minimal finite-depth birth-death/read-sampling utilities vendored for the
tail-bin cache builder.

Only the model parameter container, birth-death probability helpers, and
coefficient/PGF utilities needed by the cache-generation backend are retained
for this standalone package.  Production cache generation uses the
Cauchy/FFT coefficient path in tailbin_cache; no saddlepoint approximation is
part of the public cache-builder workflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import math
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import bdtr
from scipy.stats import norm


def extinction_probability(lam: float, delta: float, T: float) -> float:
    if T <= 0:
        return 0.0
    if abs(lam - delta) < 1e-14:
        return (lam * T) / (1.0 + lam * T)
    r = lam - delta
    er = np.exp(-r * T)
    return float(delta * (1.0 - er) / (lam - delta * er))


def bd_positive_geometric_B(lam: float, delta: float, T: float) -> float:
    if T <= 0:
        return 0.0
    if abs(lam - delta) < 1e-14:
        return float((lam * T) / (1.0 + lam * T))
    r = lam - delta
    er = np.exp(-r * T)
    return float(lam * (1.0 - er) / (lam - delta * er))


def bd_tail_gt_M(lam: float, delta: float, T: float, M: int) -> float:
    if M < 0:
        return 1.0
    A = extinction_probability(lam, delta, T)
    B = bd_positive_geometric_B(lam, delta, T)
    if B <= 0:
        return 0.0
    if M == 0:
        return 1.0 - A
    return float((1.0 - A) * np.exp(M * np.log(B)))




def required_M_for_tail(lam: float, delta: float, T: float, tail_tol: float = 1e-12) -> float:
    """Return the ideal geometric truncation M needed for P(N_T>M) <= tail_tol.

    The return value may be math.inf when the requested tail is impossible to
    represent under double precision or when the geometric parameter is
    numerically indistinguishable from one.  This is a diagnostic helper;
    choose_M_for_tail applies the operational M_cap.
    """
    A = extinction_probability(lam, delta, T)
    B = bd_positive_geometric_B(lam, delta, T)
    if (not np.isfinite(A)) or (not np.isfinite(B)) or B >= 1.0:
        return math.inf
    if B <= 0.0:
        return 1.0
    target = tail_tol / max(1.0 - A, np.finfo(float).tiny)
    if (not np.isfinite(target)) or target <= 0.0:
        return math.inf
    if target >= 1.0:
        return 1.0
    denom = math.log(B)
    if (not np.isfinite(denom)) or denom >= 0.0:
        return math.inf
    raw = math.log(target) / denom
    return float(raw) if np.isfinite(raw) else math.inf


def truncation_diagnostics(lam: float, delta: float, T: float, tail_tol: float = 1e-12,
                           M_cap: int = 2_000_000) -> Dict[str, float]:
    """Summarize the terminal clone-size truncation problem for reporting."""
    A = extinction_probability(lam, delta, T)
    B = bd_positive_geometric_B(lam, delta, T)
    required = required_M_for_tail(lam, delta, T, tail_tol)
    M, tail, ok = choose_M_for_tail(lam, delta, T, tail_tol, M_cap)
    return {
        "extinction_probability": float(A),
        "positive_geometric_B": float(B),
        "requested_tail_tol": float(tail_tol),
        "required_M_ideal": float(required),
        "M_cap": float(M_cap),
        "chosen_M": float(M),
        "tail_at_chosen_M": float(tail),
        "truncation_ok_under_cap": float(ok),
    }
def choose_M_for_tail(lam: float, delta: float, T: float, tail_tol: float = 1e-12,
                      M_cap: int = 2_000_000) -> Tuple[int, float, bool]:
    """Choose terminal clone-size truncation M.

    Hotfix: in very long-time or near-supercritical regimes B can be numerically
    indistinguishable from 1, making log(B) ~ 0 and the required M effectively
    infinite under the requested tolerance.  Do not crash by converting inf to int.
    Return M_cap, tail=1.0, ok=False so production marks the point uncertified.
    """
    A = extinction_probability(lam, delta, T)
    B = bd_positive_geometric_B(lam, delta, T)

    if (not np.isfinite(A)):
        A = 0.0

    if (not np.isfinite(B)) or B >= 1.0:
        return int(M_cap), 1.0, False

    if B <= 0:
        return 1, 0.0, True

    target = tail_tol / max(1.0 - A, np.finfo(float).tiny)
    if (not np.isfinite(target)) or target <= 0.0:
        return int(M_cap), 1.0, False

    if target >= 1:
        return 1, bd_tail_gt_M(lam, delta, T, 1), True

    denom = np.log(B)
    if (not np.isfinite(denom)) or denom >= 0.0:
        return int(M_cap), 1.0, False

    raw = np.log(target) / denom
    if (not np.isfinite(raw)) or raw > M_cap:
        return int(M_cap), bd_tail_gt_M(lam, delta, T, int(M_cap)), False

    M = max(1, int(np.ceil(raw)))
    return M, bd_tail_gt_M(lam, delta, T, M), True


def fixed_depth_rho_f(f, alpha: float, depth: int):
    ff = np.minimum(np.maximum(np.asarray(f, dtype=float), 0.0), 1.0)
    c = int(np.ceil(alpha * depth - 1e-14))
    if c <= 0:
        return np.ones_like(ff)
    if c > depth:
        return np.zeros_like(ff)
    return 1.0 - bdtr(c - 1, depth, ff)


def fixed_depth_rho_values(m, N_const, alpha, depth):
    return fixed_depth_rho_f(np.minimum(np.asarray(m, dtype=float) / N_const, 1.0), alpha, depth)


@dataclass(frozen=True)
class ModelParams:
    lam: float
    delta: float
    u: float
    T: float
    N_const: float
    alpha: float
    depth: int
    theta_f: float = 0.0
    condition_on_survival: bool = True


class ExactSeriesEngine:
    def __init__(self, params: ModelParams, M: Optional[int] = None, tail_tol: float = 1e-12,
                 M_cap: int = 2_000_000, exact_series_k_limit: int = 256,
                 exact_series_M_limit: int = 8192,
                 exact_fft_k_limit: int = 512,
                 exact_fft_M_limit: int = 4096):
        self.params = params
        if M is None:
            M, tail, ok = choose_M_for_tail(params.lam, params.delta, params.T, tail_tol, M_cap)
        else:
            tail, ok = bd_tail_gt_M(params.lam, params.delta, params.T, M), True
        # If the requested tail tolerance cannot be certified under M_cap, do
        # not allocate the full cap-sized state space.  Downstream production
        # code will mark the point uncertified before doing expensive CDF work.
        # This prevents impossible/near-supercritical points from causing OOM or
        # hours of wasted cluster time merely during engine construction.
        self.tail_bound = float(tail)
        self.truncation_ok = bool(ok)
        self.exact_series_k_limit = int(exact_series_k_limit)
        self.exact_series_M_limit = int(exact_series_M_limit)
        self.exact_fft_k_limit = int(exact_fft_k_limit)
        self.exact_fft_M_limit = int(exact_fft_M_limit)
        if not self.truncation_ok:
            M = min(int(M), max(1, self.exact_series_M_limit))
        self.M = int(M)
        self.m = np.arange(self.M + 1, dtype=float)
        self.rho = fixed_depth_rho_values(self.m, params.N_const, params.alpha, params.depth)
        nfft = 1
        while nfft < 2 * (self.M + 1) - 1:
            nfft *= 2
        self.nfft_conv = nfft
        self.ext_prob = extinction_probability(params.lam, params.delta, params.T)
        self._K_cache: Dict[Tuple[float, float, str], Tuple[Tuple[float, float, float], Dict[str, float]]] = {}
        self._fft_cache: Dict[Tuple[int, float, float, str], Tuple[np.ndarray, np.ndarray, Dict[str, float]]] = {}
        self._series_cache: Dict[Tuple[int, float, str], Tuple[np.ndarray, np.ndarray, Dict[str, float]]] = {}
        self._p0_cache: Dict[Tuple[float, str], Tuple[float, Dict[str, float]]] = {}

    def _conv_real(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.fft.irfft(np.fft.rfft(x, self.nfft_conv) * np.fft.rfft(y, self.nfft_conv), self.nfft_conv)[:self.M + 1]

    def _conv_complex_batch(self, q: np.ndarray, nfft: int) -> np.ndarray:
        fq = np.fft.fft(q, nfft, axis=1)
        return np.fft.ifft(fq * fq, axis=1)[:, :self.M + 1]

    def K012(self, s: float, rtol: float = 1e-6, atol: Optional[float] = None,
             ode_method: str = "RK45") -> Tuple[float, float, float, Dict[str, float]]:
        key = (round(float(s), 12), float(rtol), ode_method)
        if key in self._K_cache:
            vals, meta = self._K_cache[key]
            return vals[0], vals[1], vals[2], {**meta, "cached": 1.0}
        t0 = time.perf_counter()
        p = self.params
        if atol is None:
            atol = rtol * 1e-2
        n = self.M + 1
        es = float(np.exp(s))
        c = es - 1.0
        rho = self.rho
        y0 = np.zeros(3*n, dtype=float)
        y0[1] = 1.0

        def rhs(_t, flat):
            q = flat[:n]
            a = flat[n:2*n]
            b = flat[2*n:]
            qq = self._conv_real(q, q)
            qa2 = 2.0 * self._conv_real(q, a)
            qb_aa2 = 2.0 * self._conv_real(q, b) + 2.0 * self._conv_real(a, a)
            dq = p.lam * (qq - q) - p.delta * q + p.u * c * rho * q
            dq[0] += p.delta
            da = p.lam * (qa2 - a) - p.delta * a + p.u * (es * rho * q + c * rho * a)
            db = p.lam * (qb_aa2 - b) - p.delta * b + p.u * (es * rho * q + 2.0 * es * rho * a + c * rho * b)
            return np.concatenate([dq, da, db])

        sol = solve_ivp(rhs, (0.0, p.T), y0, rtol=rtol, atol=atol, method=ode_method)
        q = sol.y[:n, -1]
        a = sol.y[n:2*n, -1]
        b = sol.y[2*n:, -1]
        # For real s, q and the first two s-derivative states represent
        # nonnegative generating-function/moment contributions.  Tiny negative
        # components can appear from adaptive ODE/FFT convolution error and are
        # especially damaging after shifted founder weighting concentrates on a
        # few terminal states.  Enforce the mathematical positivity before
        # forming founder-weighted sums.
        q = np.maximum(q, 0.0)
        a = np.maximum(a, 0.0)
        b = np.maximum(b, 0.0)

        founder_exponent_max = 0.0
        used_scaled_founder = 0.0
        if p.theta_f:
            th = float(p.theta_f)
            # Founder mutations are analytically Poisson-mixed through
            # exp(theta_f * (exp(s)-1) * rho_m).  Directly forming that
            # exponential overflows for large theta_f and positive saddlepoints.
            # Work in a shifted log scale; the common exp(max g_m) cancels in
            # K'(s) and K''(s), and is restored only in K(s).
            g = th * c * rho
            founder_exponent_max = float(np.max(g))
            h = np.exp(g - founder_exponent_max)
            gp = th * es * rho
            gpp = th * es * rho
            Gs = float(np.dot(q, h))
            Gps = float(np.dot(a, h) + np.dot(q, gp * h))
            Gpps = float(np.dot(b, h) + 2.0*np.dot(a, gp * h) + np.dot(q, (gpp + gp * gp) * h))
            used_scaled_founder = 1.0
        else:
            Gs = float(np.sum(q)); Gps = float(np.sum(a)); Gpps = float(np.sum(b))

        if p.condition_on_survival:
            den_surv = 1.0 - self.ext_prob
            # Extinction mass contributes only to the zero terminal-size state.
            # In shifted founder scale it is multiplied by exp(-max g_m).
            ext_scaled = self.ext_prob * math.exp(-founder_exponent_max) if founder_exponent_max < 745.0 else 0.0
            denom_scaled = Gs - ext_scaled
            if denom_scaled <= 0.0 or not np.isfinite(denom_scaled):
                denom_scaled = max(Gs, np.finfo(float).tiny)
            K0 = founder_exponent_max + math.log(max(denom_scaled, np.finfo(float).tiny)) - math.log(den_surv)
            K1 = Gps / denom_scaled
            K2 = float(max(Gpps / denom_scaled - K1 * K1, 1e-14))
        else:
            denom_scaled = max(Gs, np.finfo(float).tiny)
            K0 = founder_exponent_max + math.log(denom_scaled)
            K1 = Gps / denom_scaled
            K2 = float(max(Gpps / denom_scaled - K1 * K1, 1e-14))
        meta = {"seconds": time.perf_counter() - t0, "nfev": float(sol.nfev),
                "success": float(sol.success), "M": float(self.M),
                "tail_bound_P_N_gt_M": self.tail_bound, "cached": 0.0,
                "engine": "exact_series_K012_scaled_founder",
                "founder_theta_f": float(p.theta_f),
                "founder_exponent_max": float(founder_exponent_max),
                "used_scaled_founder": float(used_scaled_founder),
                "truncation_ok_under_cap": float(self.truncation_ok)}
        self._K_cache[key] = ((K0, K1, K2), meta)
        return K0, K1, K2, meta

    def mean_variance(self, **kwargs):
        _, mu, var, meta = self.K012(0.0, **kwargs)
        return mu, var, meta

    def pgf_values_batch(self, ys: np.ndarray, rtol: float = 1e-7, atol: Optional[float] = None,
                         ode_method: str = "DOP853") -> Tuple[np.ndarray, Dict[str, float]]:
        t0 = time.perf_counter()
        p = self.params
        ys = np.asarray(ys, dtype=np.complex128)
        ny = ys.size
        if atol is None:
            atol = rtol * 1e-2
        q0 = np.zeros((ny, self.M + 1), dtype=np.complex128)
        q0[:, 1] = 1.0
        rho = self.rho.astype(np.complex128)[None, :]
        yy = (ys - 1.0)[:, None]
        nfft = 1
        while nfft < 2 * (self.M + 1) - 1:
            nfft *= 2

        def rhs(_t, flat):
            q = flat.reshape((ny, self.M + 1))
            conv = self._conv_complex_batch(q, nfft)
            dq = p.lam * (conv - q) - p.delta * q + p.u * yy * rho * q
            dq[:, 0] += p.delta
            return dq.reshape(ny * (self.M + 1))

        sol = solve_ivp(rhs, (0.0, p.T), q0.reshape(ny * (self.M + 1)), rtol=rtol, atol=atol, method=ode_method)
        qT = sol.y[:, -1].reshape((ny, self.M + 1))
        if p.theta_f:
            vals = np.sum(qT * np.exp(p.theta_f * yy * rho), axis=1)
        else:
            vals = np.sum(qT, axis=1)
        if p.condition_on_survival:
            vals = (vals - self.ext_prob) / (1.0 - self.ext_prob)
        meta = {"engine": "exact_series_exact_complex_pgf_batch", "success": float(sol.success),
                "nfev": float(sol.nfev), "ny": float(ny), "M": float(self.M),
                "tail_bound_P_N_gt_M": self.tail_bound, "seconds": time.perf_counter() - t0}
        return vals, meta

    def p0_exact(self, rtol: float = 1e-7, ode_method: str = "DOP853") -> Tuple[float, Dict[str, float]]:
        key = (float(rtol), ode_method)
        if key in self._p0_cache:
            p0, meta = self._p0_cache[key]
            return p0, {**meta, "cached": 1.0}
        vals, meta = self.pgf_values_batch(np.array([0.0 + 0.0j]), rtol=rtol, ode_method=ode_method)
        p0 = float(np.clip(vals[0].real, 0.0, 1.0))
        meta = {**meta, "method": "exact_pgf_at_zero", "cached": 0.0}
        self._p0_cache[key] = (p0, meta)
        return p0, meta


    def exact_pmf_cdf_series(self, k_max: int, rtol: float = 1e-7, atol: Optional[float] = None,
                             ode_method: str = "RK45", clip: bool = True
                             ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Exact coefficients up to k_max via a triangular Taylor system in y.

        This avoids the slow complex Cauchy/FFT batch for low-count CDF queries.
        Let q_j[m,t] be the coefficient of y^j in the clone PGF state at size m.
        The read-sampling term u*(y-1)*rho*q couples only q_j and q_{j-1}, so
        the system is triangular in mutation-count degree.  At terminal time we
        fold in founder mutations exactly through the Poisson coefficients of
        exp(theta_f*(y-1)*rho_m).
        """
        k_max = int(k_max)
        key = (k_max, float(rtol), ode_method)
        if key in self._series_cache:
            pmf, cdf, meta = self._series_cache[key]
            return pmf.copy(), cdf.copy(), {**meta, "cached": 1.0, "cache_reuse": "exact"}

        # Big exact_series win: reuse any previously computed larger Taylor system.
        # v12 cached only exact k_max, so cdf(20), cdf(40), cdf(80) solved
        # three separate ODEs.  Here, once degree 80 exists, all lower degrees
        # are O(1) slices.
        best_key = None
        best_K = None
        for ck in self._series_cache:
            K_cached, r_cached, method_cached = ck
            if r_cached == float(rtol) and method_cached == ode_method and K_cached >= k_max:
                if best_K is None or K_cached < best_K:
                    best_key = ck
                    best_K = K_cached
        if best_key is not None:
            pmf_big, cdf_big, meta_big = self._series_cache[best_key]
            return (pmf_big[:k_max + 1].copy(), cdf_big[:k_max + 1].copy(),
                    {**meta_big, "cached": 1.0, "cache_reuse": "larger_series",
                     "source_k_max": float(best_K), "k_max": float(k_max)})

        t0 = time.perf_counter()
        p = self.params
        if atol is None:
            atol = rtol * 1e-2
        K = k_max
        M1 = self.M + 1
        rho = self.rho
        nfft = self.nfft_conv
        y0 = np.zeros((K + 1, M1), dtype=float)
        y0[0, 1] = 1.0

        def rhs(_t, flat):
            Q = flat.reshape((K + 1, M1))
            F = np.fft.rfft(Q, nfft, axis=1)
            conv = np.empty_like(Q)
            for j in range(K + 1):
                acc = np.zeros(F.shape[1], dtype=np.complex128)
                for a in range(j + 1):
                    acc += F[a] * F[j - a]
                conv[j] = np.fft.irfft(acc, nfft)[:M1]
            dQ = p.lam * (conv - Q) - p.delta * Q - p.u * rho[None, :] * Q
            if K >= 1:
                dQ[1:] += p.u * rho[None, :] * Q[:-1]
            dQ[0, 0] += p.delta
            return dQ.reshape((K + 1) * M1)

        sol = solve_ivp(rhs, (0.0, p.T), y0.reshape((K + 1) * M1), rtol=rtol, atol=atol, method=ode_method)
        Q = sol.y[:, -1].reshape((K + 1, M1))

        pmf = np.zeros(K + 1, dtype=float)
        if p.theta_f:
            th_rho = p.theta_f * rho
            base = np.exp(-th_rho)
            pois = np.zeros((K + 1, M1), dtype=float)
            pois[0] = base
            for l in range(1, K + 1):
                pois[l] = pois[l - 1] * th_rho / float(l)
            for j in range(K + 1):
                s = 0.0
                for l in range(j + 1):
                    s += float(np.dot(Q[j - l], pois[l]))
                pmf[j] = s
        else:
            pmf = np.sum(Q, axis=1)

        if p.condition_on_survival:
            den = 1.0 - self.ext_prob
            pmf[0] = (pmf[0] - self.ext_prob) / den
            if K >= 1:
                pmf[1:] /= den
        if clip:
            pmf = np.clip(pmf, 0.0, 1.0)
        cdf = np.cumsum(pmf)
        if clip:
            cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
        meta = {"engine": "exact_series_exact_low_count_series", "success": float(sol.success),
                "nfev": float(sol.nfev), "M": float(self.M), "k_max": float(K),
                "tail_bound_P_N_gt_M": self.tail_bound, "seconds": time.perf_counter() - t0,
                "cached": 0.0, "computed_object": "exact_series_pmf_cdf"}
        self._series_cache[key] = (pmf.copy(), cdf.copy(), meta)
        return pmf, cdf, meta

    def exact_pmf_cdf_fft(self, k_max: int, radius: float = 0.72, n_fft: Optional[int] = None,
                          rtol: float = 1e-7, ode_method: str = "DOP853", clip: bool = True
                          ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        if n_fft is None:
            n_fft = 1
            while n_fft <= max(4 * (k_max + 1), 64):
                n_fft *= 2
        if k_max >= n_fft:
            raise ValueError("k_max must be smaller than n_fft")
        key = (int(k_max), float(radius), float(rtol), ode_method)
        if key in self._fft_cache:
            pmf, cdf, meta = self._fft_cache[key]
            return pmf.copy(), cdf.copy(), {**meta, "cached": 1.0}
        js = np.arange(n_fft)
        ys = radius * np.exp(2j * np.pi * js / n_fft)
        vals, meta = self.pgf_values_batch(ys, rtol=rtol, ode_method=ode_method)
        coeffs = np.fft.fft(vals) / n_fft
        pmf = np.array([(coeffs[k] / (radius ** k)).real for k in range(k_max + 1)])
        cdf = np.cumsum(pmf)
        if clip:
            pmf = np.clip(pmf, 0.0, 1.0)
            cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
        meta.update({"computed_object": "exact_fft_pmf_cdf", "radius": float(radius),
                     "n_fft": float(n_fft), "k_max": float(k_max), "cached": 0.0})
        self._fft_cache[key] = (pmf.copy(), cdf.copy(), meta)
        return pmf, cdf, meta

    def _bracket(self, x: float, rtol: float, ode_method: str, max_abs_s: float = 10.0):
        def f(s):
            return self.K012(s, rtol=rtol, ode_method=ode_method)[1] - x
        lo, hi = -0.125, 0.125
        flo, fhi = f(lo), f(hi)
        while flo > 0.0 and abs(lo) < max_abs_s:
            lo *= 2.0; flo = f(lo)
        while fhi < 0.0 and abs(hi) < max_abs_s:
            hi *= 2.0; fhi = f(hi)
        if flo > 0.0 or fhi < 0.0:
            return None, None, flo, fhi
        return lo, hi, flo, fhi

    def cdf_lr(self, k: int, continuity: float = 0.5, rtol: float = 1e-6,
               ode_method: str = "RK45") -> Tuple[float, Dict[str, float]]:
        t0 = time.perf_counter()
        x = max(float(k) + float(continuity), 0.0)
        _, mu, var, _ = self.K012(0.0, rtol=rtol, ode_method=ode_method)
        sigma = float(np.sqrt(max(var, 1e-14)))
        if abs(x - mu) < 1e-4 * max(1.0, sigma):
            return float(norm.cdf((x - mu) / sigma)), {"method": "normal_at_mean", "mean": mu, "var": var,
                                                       "seconds": time.perf_counter() - t0}
        lo, hi, flo, fhi = self._bracket(x, rtol, ode_method)
        if lo is None:
            return float(norm.cdf((x - mu) / sigma)), {"method": "normal_bracket_fallback", "mean": mu,
                                                       "var": var, "bracket_flo": flo, "bracket_fhi": fhi,
                                                       "seconds": time.perf_counter() - t0}
        root = brentq(lambda z: self.K012(z, rtol=rtol, ode_method=ode_method)[1] - x,
                      lo, hi, xtol=3e-4, rtol=3e-4, maxiter=36)
        K, _, K2, _ = self.K012(root, rtol=rtol, ode_method=ode_method)
        rad = max(2.0 * (root * x - K), 1e-14)
        w = (1.0 if root >= 0.0 else -1.0) * np.sqrt(rad)
        u = root * np.sqrt(K2)
        if abs(root) < 2e-3 or abs(w) < 1e-6 or abs(u) < 1e-6:
            F = norm.cdf((x - mu) / sigma); method = "normal_singular"
        else:
            F = norm.cdf(w) + norm.pdf(w) * (1.0 / w - 1.0 / u); method = "lugannani_rice"
        return float(np.clip(F, 0.0, 1.0)), {"method": method, "k": float(k), "x": x,
                                             "s_hat": float(root), "mean": mu, "var": var,
                                             "seconds": time.perf_counter() - t0,
                                             "cache_size": float(len(self._K_cache))}

    def cdf(self, k: int, *, prefer_exact: bool = False, validate: bool = False,
            lr_rtol: float = 1e-6, exact_rtol: float = 1e-7,
            ode_method_lr: str = "RK45", ode_method_exact: str = "DOP853") -> Tuple[float, Dict[str, float]]:
        t0 = time.perf_counter()
        k = int(k)
        if k < 0:
            return 0.0, {"method": "support_left_of_zero", "seconds": 0.0}
        if k == 0:
            # exact_series: zero mass comes from the real Taylor solver, avoiding the
            # complex PGF-at-zero ODE unless the user calls p0_exact directly.
            if self.M <= self.exact_series_M_limit:
                pmf, cdf, meta = self.exact_pmf_cdf_series(0, rtol=exact_rtol, ode_method=ode_method_exact)
                return float(cdf[0]), {**meta, "adaptive_regime": "zero_mass_series",
                                       "seconds_total": time.perf_counter() - t0}
            F, meta = self.p0_exact(rtol=exact_rtol, ode_method=ode_method_exact)
            return F, {**meta, "adaptive_regime": "zero_mass", "seconds_total": time.perf_counter() - t0}

        series_allowed = (self.M <= self.exact_series_M_limit and k <= self.exact_series_k_limit)
        if prefer_exact or series_allowed:
            pmf, cdf, meta = self.exact_pmf_cdf_series(k, rtol=exact_rtol, ode_method=ode_method_exact)
            return float(cdf[k]), {**meta, "adaptive_regime": "exact_series", "seconds_total": time.perf_counter() - t0}

        F, meta = self.cdf_lr(k, rtol=lr_rtol, ode_method=ode_method_lr)
        out = {**meta, "adaptive_regime": "saddlepoint_lr", "seconds_total": time.perf_counter() - t0,
               "tail_bound_P_N_gt_M": self.tail_bound, "truncation_ok_under_cap": float(self.truncation_ok)}
        if validate and self.M <= self.exact_series_M_limit and k <= max(self.exact_series_k_limit, k):
            # Validation can be expensive; it is opt-in and only done under the M cap.
            pmf, cdf, emeta = self.exact_pmf_cdf_series(k, rtol=exact_rtol, ode_method=ode_method_exact)
            out.update({"validation_exact_cdf": float(cdf[k]), "validation_abs_error": abs(F - float(cdf[k])),
                        "validation_seconds": emeta.get("seconds", np.nan)})
        return F, out

    def cdf_many(self, ks: Iterable[int], **kwargs) -> List[Tuple[int, float, Dict[str, float]]]:
        """Evaluate many CDF thresholds with one exact-series solve when possible.

        This is the main exact_series production improvement.  Likelihood grids usually
        ask for many thresholds under the same parameter point.  v12 solved the
        triangular Taylor ODE separately per k.  exact_series groups all thresholds that
        are eligible for exact-series evaluation, solves once to max(k), then
        returns slices.  Larger thresholds still use LR individually.
        """
        t0 = time.perf_counter()
        ks_list = [int(k) for k in ks]
        if not ks_list:
            return []

        prefer_exact = bool(kwargs.get("prefer_exact", False))
        exact_rtol = float(kwargs.get("exact_rtol", 1e-7))
        ode_method_exact = kwargs.get("ode_method_exact", "DOP853")

        out: Dict[int, Tuple[int, float, Dict[str, float]]] = {}
        nonneg = [k for k in ks_list if k >= 0]
        series_ks = [k for k in nonneg
                     if self.M <= self.exact_series_M_limit
                     and (prefer_exact or k <= self.exact_series_k_limit)]
        if series_ks:
            K = max(series_ks)
            pmf, cdf, meta = self.exact_pmf_cdf_series(K, rtol=exact_rtol, ode_method=ode_method_exact)
            for k in series_ks:
                reg = "zero_mass_series" if k == 0 else "exact_series_batch"
                out[k] = (k, float(cdf[k]), {**meta, "adaptive_regime": reg,
                                             "batch_k_max": float(K),
                                             "seconds_total_batch": time.perf_counter() - t0})

        for k in ks_list:
            if k < 0:
                out[k] = (k, 0.0, {"method": "support_left_of_zero", "seconds": 0.0})
            elif k not in out:
                F, meta = self.cdf(k, **kwargs)
                out[k] = (k, F, meta)

        return [out[k] for k in ks_list]

    def calibrate_lr(self, ks: Iterable[int], lr_rtol: float = 1e-5, exact_rtol: float = 1e-7,
                     ode_method_lr: str = "RK45", ode_method_exact: str = "DOP853") -> List[Dict[str, float]]:
        out = []
        for k in sorted(set(int(x) for x in ks if int(x) >= 0)):
            if self.M > self.exact_fft_M_limit:
                raise RuntimeError("M too large for exact series calibration under exact_series_M_limit")
            F_lr, mlr = self.cdf_lr(k, rtol=lr_rtol, ode_method=ode_method_lr)
            _, cdf, mex = self.exact_pmf_cdf_series(k, rtol=exact_rtol, ode_method=ode_method_exact)
            F_ex = float(cdf[k])
            out.append({"k": float(k), "cdf_lr": F_lr, "cdf_exact": F_ex,
                        "abs_error": abs(F_lr - F_ex), "lr_method": mlr.get("method", ""),
                        "lr_seconds": mlr.get("seconds", np.nan), "exact_seconds": mex.get("seconds", np.nan)})
        return out


def smoke_benchmark():
    params = ModelParams(lam=1.0, delta=0.2, u=1.0, T=3.0, N_const=1000.0,
                         alpha=0.05, depth=100, theta_f=200.0, condition_on_survival=True)
    eng = ExactSeriesEngine(params, M=120, exact_series_k_limit=80, exact_fft_k_limit=0)
    mu, var, meta = eng.mean_variance(rtol=1e-5)
    sd = float(np.sqrt(var))
    ks = [0, max(1, int(mu - 2*sd)), int(mu), int(mu + 2*sd), int(mu + 4*sd)]
    rows = []
    for k in ks:
        F, m = eng.cdf(k, lr_rtol=1e-5, exact_rtol=1e-6)
        rows.append((k, F, m["adaptive_regime"], m.get("method", ""), m.get("seconds_total", m.get("seconds", np.nan))))
    cal_ks = [max(1, int(mu - sd)), int(mu), int(mu + sd)]
    cal = eng.calibrate_lr(cal_ks, lr_rtol=1e-5, exact_rtol=1e-6)
    return {"mean": mu, "var": var, "M": eng.M, "tail_bound": eng.tail_bound,
            "mean_meta": meta, "rows": rows, "calibration": cal}


if __name__ == "__main__":
    out = smoke_benchmark()
    print("mean", out["mean"], "var", out["var"], "M", out["M"], "tail", out["tail_bound"])
    print("adaptive rows")
    for row in out["rows"]:
        print(row)
    print("calibration")
    for row in out["calibration"]:
        print(row)
