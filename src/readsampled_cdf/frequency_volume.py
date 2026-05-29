"""v27 adaptive frequency-volume engine for finite-depth read sampling.

This module is a scalable, convergent-in-resolution candidate for the exact
finite-depth read-sampled marginal CDF problem.  It avoids the impossible
integer terminal-size state space m=0,...,N_const by discretizing terminal
*frequency* p=m/N_const into adaptive finite-volume bins.

The state is

    q_j(t;s) ~= E[ exp(s Z_t) 1{N_t/N_const lies in frequency bin j} ].

For each bin, finite-depth read sampling is evaluated with the actual binomial
read-depth tail

    rho(p) = P[Binomial(depth, min(p,1)) >= ceil(alpha*depth)].

The birth term requires adding two independent terminal frequencies.  Instead
of an integer convolution over m, we precompute a small bin-addition table over
frequency bins.  This is still a convolution-like operation, but its cost is
O(J^2) with J~50..300 rather than O(M log M) or O(N log N), and J is chosen by
accuracy/convergence diagnostics rather than by population size.  Refining the
frequency grid gives the intended exact-ish route.

This engine is not declared finally production-certified.  It is designed to be
compared against the exact coefficient engine for small N and against its own
resolution ladder for large N.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.stats import norm

from .distribution import ModelParams, ExactSeriesEngine, extinction_probability
from .scalable_laplace import read_depth_rho, read_threshold

try:  # optional acceleration for finite-volume convolution operators
    from numba import njit  # type: ignore

    @njit(cache=True)
    def _sparse_conv_accel(ii, jj, kk, ww, x, y, n_classes):
        out = np.zeros(n_classes, dtype=np.float64)
        for n in range(kk.size):
            out[kk[n]] += ww[n] * x[ii[n]] * y[jj[n]]
        return out

    @njit(cache=True)
    def _point_conv_accel(add_index, x, y, n_classes):
        out = np.zeros(n_classes, dtype=np.float64)
        for i in range(n_classes):
            xi = x[i]
            if xi != 0.0:
                for j in range(n_classes):
                    out[add_index[i, j]] += xi * y[j]
        return out

except Exception:  # pragma: no cover - exercised only when numba is absent
    _sparse_conv_accel = None
    _point_conv_accel = None


@dataclass(frozen=True)
class FrequencyVolumeConfig:
    """Numerical settings for AdaptiveFrequencyVolumeEngine."""

    n_bins: int = 160
    transition_width_multiplier: float = 10.0
    transition_fraction: float = 0.55
    log_fraction: float = 0.30
    linear_fraction: float = 0.10
    upper_tail_fraction: float = 0.15
    representative: str = "geometric"  # geometric or midpoint
    addition_rule: str = "uniform"  # uniform, point, or linear fixed-pivot addition
    ode_method: str = "DOP853"
    rtol: float = 2e-7
    atol_factor: float = 1e-2
    max_abs_s: float = 12.0
    min_positive_p: Optional[float] = None
    # When positive and N_const is an integer no larger than this value, use an
    # exact capped integer grid: classes m=0,1,...,N-1 plus a single high class
    # m>=N.  This is mainly for validation and regression testing; production
    # large-N runs leave it disabled.
    integer_grid_max_N: int = 0
    # Resolution-change diagnostic thresholds, used by helper ladders.
    ladder_K_tol: float = 5e-4
    ladder_K1_tol: float = 5e-3
    ladder_K2_rel_tol: float = 5e-2


def _unique_sorted_unit(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.unique(arr)
    arr.sort()
    if arr.size == 0 or arr[0] != 0.0:
        arr = np.concatenate([[0.0], arr])
    if arr[-1] != 1.0:
        arr = np.concatenate([arr, [1.0]])
    return np.unique(arr)


def frequency_edges(alpha: float, depth: int, N_const: float, config: FrequencyVolumeConfig) -> np.ndarray:
    """Construct adaptive frequency edges in [0,1].

    The grid is dense around the read-depth transition near alpha_eff=h/depth,
    includes a log grid near zero, and includes a coarse linear background.  The
    absorbing high bin [1, infinity) is represented separately by the engine.
    """
    n = max(12, int(config.n_bins))
    depth = int(depth)
    h = read_threshold(alpha, depth)
    alpha_eff = 0.0 if depth <= 0 else min(max(float(h) / float(depth), 0.0), 1.0)
    p_min = config.min_positive_p
    if p_min is None:
        p_min = max(1.0 / max(float(N_const), 1.0), 1e-14)
    p_min = min(max(float(p_min), 1e-14), 1.0)
    # Binomial transition width in frequency.  For h=0 or h=depth, keep a small
    # buffer so the grid still has a meaningful local region.
    var = max(alpha_eff * (1.0 - alpha_eff), 1e-10)
    width = max(3.0 / max(depth, 1), float(config.transition_width_multiplier) * math.sqrt(var / max(depth, 1)))
    lo = max(0.0, alpha_eff - width)
    hi = min(1.0, alpha_eff + width)

    n_trans = max(8, int(round(n * config.transition_fraction)))
    n_log = max(8, int(round(n * config.log_fraction)))
    n_upper = max(8, int(round(n * config.upper_tail_fraction)))
    n_lin = max(8, n - n_trans - n_log - n_upper)

    vals: List[float] = [0.0, p_min, alpha_eff, lo, hi, 1.0]
    # Background linear grid.
    vals.extend(np.linspace(0.0, 1.0, n_lin + 1).tolist())
    # Near-zero grid; useful because most terminal frequencies can be tiny when
    # N_const is huge.
    vals.extend(np.geomspace(p_min, 1.0, n_log + 1).tolist())
    # Near-one grid; important in long-time regimes because rare near-N_const
    # subclones and the capped high class can dominate high read-depth moments.
    upper_dist = np.geomspace(p_min, 1.0, n_upper + 1)
    vals.extend((1.0 - upper_dist).tolist())
    # Transition grid: use a warped grid with extra points near alpha_eff.
    if hi > lo:
        vals.extend(np.linspace(lo, hi, n_trans + 1).tolist())
        # Add quantile-like clustering around alpha_eff.
        z = np.linspace(-1.0, 1.0, max(9, n_trans // 2))
        local = alpha_eff + width * np.tanh(2.2 * z) / math.tanh(2.2)
        vals.extend(np.clip(local, lo, hi).tolist())
    return _unique_sorted_unit(vals)


def _bin_representatives(edges: np.ndarray, representative: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bin left/right/representative for finite bins [edge_i, edge_{i+1}).

    The point mass at zero is represented as its own class 0.  Positive finite
    bins cover (0,1), and a final high class [1, infinity) is added by the
    engine with representative p=1 for read sampling and bin addition.
    """
    edges = np.asarray(edges, dtype=float)
    # positive intervals: [edges[i], edges[i+1]) for i=0..len-2, but the first
    # interval begins at 0.  Class 0 is exact zero, so for the first finite bin
    # use a positive representative.
    left = edges[:-1].copy()
    right = edges[1:].copy()
    rep_mode = representative.lower()
    if rep_mode == "midpoint":
        rep = 0.5 * (left + right)
    elif rep_mode == "geometric":
        rep = np.empty_like(left)
        for i, (a, b) in enumerate(zip(left, right)):
            if a <= 0.0:
                rep[i] = 0.5 * b
            else:
                rep[i] = math.sqrt(a * b)
    else:
        raise ValueError("representative must be 'geometric' or 'midpoint'")
    rep = np.clip(rep, 0.0, 1.0)
    return left, right, rep


def _uniform_sum_cdf(z: float, a: float, b: float, c: float, d: float) -> float:
    """CDF of X+Y at z for independent uniforms on [a,b] and [c,d].

    Uses the exact inclusion-exclusion formula.  Degenerate intervals are
    handled by the caller; this helper assumes b>a and d>c.
    """
    def pp2(x: float) -> float:
        return x * x if x > 0.0 else 0.0
    den = 2.0 * (b - a) * (d - c)
    val = (pp2(z - a - c) - pp2(z - b - c) - pp2(z - a - d) + pp2(z - b - d)) / den
    if val <= 0.0:
        return 0.0
    if val >= 1.0:
        return 1.0
    return float(val)


def _uniform_sum_interval_prob(lo: float, hi: float, a: float, b: float, c: float, d: float) -> float:
    """Probability that Uniform[a,b]+Uniform[c,d] lies in [lo,hi)."""
    if hi <= lo:
        return 0.0
    if b <= a and d <= c:
        s = a + c
        return 1.0 if lo <= s < hi else 0.0
    if b <= a:
        return max(0.0, min(hi - (a + c), d - c) - max(lo - (a + c), 0.0)) / max(d - c, np.finfo(float).tiny)
    if d <= c:
        return max(0.0, min(hi - (c + a), b - a) - max(lo - (c + a), 0.0)) / max(b - a, np.finfo(float).tiny)
    return max(0.0, _uniform_sum_cdf(hi, a, b, c, d) - _uniform_sum_cdf(lo, a, b, c, d))


class AdaptiveFrequencyVolumeEngine:
    """Finite-depth frequency-volume K(s) engine.

    This is the v27 candidate for the broad parameter regime.  It includes read
    depth exactly at bin representatives and avoids terminal-size truncation.
    """

    def __init__(self, params: ModelParams, *, config: Optional[FrequencyVolumeConfig] = None):
        self.params = params
        self.config = config or FrequencyVolumeConfig()
        N_round = int(round(float(params.N_const)))
        self._integer_N: Optional[int] = None
        if (int(self.config.integer_grid_max_N) > 0
                and abs(float(params.N_const) - float(N_round)) < 1e-9
                and 1 <= N_round <= int(self.config.integer_grid_max_N)):
            # Exact capped small-N grid: 0, 1/N, ..., (N-1)/N, and one high
            # class for all m>=N where the read probability is exactly 1.
            self._integer_N = N_round
            self.edges = np.linspace(0.0, 1.0, N_round + 1)
            self.finite_left = np.arange(1, N_round, dtype=float) / float(N_round)
            self.finite_right = self.finite_left.copy()
            self.finite_rep = self.finite_left.copy()
            self.p_rep = np.concatenate([[0.0], self.finite_rep, [1.0]])
        else:
            self.edges = frequency_edges(params.alpha, params.depth, params.N_const, self.config)
            left, right, finite_rep = _bin_representatives(self.edges, self.config.representative)
            # Classes: 0 = exact extinct/zero size.  1..F = finite positive bins in
            # [0,1).  Last = high bin [1, infinity).  The finite bins include the
            # first interval [0, eps); class 0 is a separate atom and convolution
            # treats it exactly.
            self.finite_left = left
            self.finite_right = right
            self.finite_rep = finite_rep
            self.p_rep = np.concatenate([[0.0], finite_rep, [1.0]])
        self.n_classes = int(self.p_rep.size)
        self.high_index = self.n_classes - 1
        self.rho = read_depth_rho(self.p_rep, params.alpha, params.depth)
        self.rho[0] = 0.0
        self.rho[self.high_index] = 1.0
        self.ext_prob = extinction_probability(params.lam, params.delta, params.T)
        self._add_index = self._build_addition_index()
        self._uniform_i: Optional[np.ndarray] = None
        self._uniform_j: Optional[np.ndarray] = None
        self._uniform_k: Optional[np.ndarray] = None
        self._uniform_w: Optional[np.ndarray] = None
        rule = str(self.config.addition_rule).lower()
        if rule == "uniform" and self._integer_N is None:
            self._build_uniform_addition_sparse()
        elif rule == "linear" and self._integer_N is None:
            self._build_linear_addition_sparse()
        self._K_cache: Dict[Tuple[float, float, str], Tuple[Tuple[float, float, float], Dict[str, float]]] = {}

    def _class_for_p(self, pval: float) -> int:
        pval = float(pval)
        if pval <= 0.0:
            return 0
        if pval >= 1.0:
            return self.high_index
        if self._integer_N is not None:
            m = int(round(pval * float(self._integer_N)))
            if m <= 0:
                return 0
            if m >= self._integer_N:
                return self.high_index
            return m
        # finite bin index in edges -> class offset +1
        idx = int(np.searchsorted(self.edges, pval, side="right") - 1)
        idx = min(max(idx, 0), len(self.edges) - 2)
        return idx + 1

    def _class_interval(self, idx: int) -> Tuple[float, float]:
        """Frequency interval represented by class idx.

        Class 0 is the exact zero atom.  The high class is [1, infinity) but is
        treated separately in addition; finite classes are subintervals of
        [0,1).
        """
        if idx == 0:
            return 0.0, 0.0
        if idx == self.high_index:
            return 1.0, math.inf
        k = idx - 1
        return float(self.finite_left[k]), float(self.finite_right[k])

    def _build_uniform_addition_sparse(self) -> None:
        """Precompute conservative finite-volume birth-addition weights.

        For finite frequency bins, this distributes the product mass from a pair
        of source bins according to the exact sum distribution of two independent
        uniforms on the source intervals.  It is still an approximation to the
        true within-bin shape, but it is a conservative finite-volume operator:
        each pair's weights sum to one, and refinement converges to the integer
        addition law without choosing a single representative point.
        """
        ii: List[int] = []
        jj: List[int] = []
        kk: List[int] = []
        ww: List[float] = []
        edges = self.edges
        # Finite target intervals are classes 1..high_index-1, with edges
        # [edges[t-1], edges[t]).  The high class receives all mass at >=1.
        for i in range(self.n_classes):
            ai, bi = self._class_interval(i)
            for j in range(self.n_classes):
                if i == 0:
                    ii.append(i); jj.append(j); kk.append(j); ww.append(1.0); continue
                if j == 0:
                    ii.append(i); jj.append(j); kk.append(i); ww.append(1.0); continue
                if i == self.high_index or j == self.high_index:
                    ii.append(i); jj.append(j); kk.append(self.high_index); ww.append(1.0); continue
                aj, bj = self._class_interval(j)
                lo_s = ai + aj
                hi_s = bi + bj
                if hi_s <= 0.0:
                    ii.append(i); jj.append(j); kk.append(0); ww.append(1.0); continue
                if lo_s >= 1.0:
                    ii.append(i); jj.append(j); kk.append(self.high_index); ww.append(1.0); continue
                # finite target classes overlapping [lo_s, min(hi_s,1))
                t0 = max(0, int(np.searchsorted(edges, lo_s, side="right") - 1))
                t1 = min(len(edges) - 2, int(np.searchsorted(edges, min(hi_s, 1.0), side="left")))
                mass = 0.0
                for t in range(t0, t1 + 1):
                    lo = float(edges[t]); hi = float(edges[t + 1])
                    pr = _uniform_sum_interval_prob(lo, hi, ai, bi, aj, bj)
                    if pr > 1e-15:
                        ii.append(i); jj.append(j); kk.append(t + 1); ww.append(float(pr)); mass += float(pr)
                # high target [1, infinity)
                pr_hi = max(0.0, 1.0 - _uniform_sum_cdf(1.0, ai, bi, aj, bj)) if bi > ai and bj > aj else (1.0 if ai + aj >= 1.0 else 0.0)
                if pr_hi > 1e-15:
                    ii.append(i); jj.append(j); kk.append(self.high_index); ww.append(float(pr_hi)); mass += float(pr_hi)
                # Renormalize each pair defensively to avoid tiny edge leakage.
                if mass <= 0.0:
                    target = self._class_for_p(self.p_rep[i] + self.p_rep[j])
                    ii.append(i); jj.append(j); kk.append(target); ww.append(1.0)
                elif abs(mass - 1.0) > 1e-12:
                    # Renormalize weights appended for this pair only.
                    # Walk backwards until the pair changes.
                    r = len(ii) - 1
                    while r >= 0 and ii[r] == i and jj[r] == j:
                        ww[r] = ww[r] / mass
                        r -= 1
        self._uniform_i = np.asarray(ii, dtype=np.int32)
        self._uniform_j = np.asarray(jj, dtype=np.int32)
        self._uniform_k = np.asarray(kk, dtype=np.int32)
        self._uniform_w = np.asarray(ww, dtype=float)


    def _build_linear_addition_sparse(self) -> None:
        """Precompute fixed-pivot, mean-preserving addition weights.

        The older point rule sends a pair of source classes to one target class,
        which can bias the growth wave on coarse log-frequency grids.  The
        linear fixed-pivot rule sends each pair to the two neighboring target
        pivots bracketing p_i+p_j, with weights chosen so that the represented
        frequency mean is exactly preserved.  This is still a deterministic
        sectional approximation, but it is much less biased and remains exact on
        the small-N integer validation grid when sums land on integer pivots.
        """
        ii: List[int] = []
        jj: List[int] = []
        kk: List[int] = []
        ww: List[float] = []
        piv = np.asarray(self.p_rep, dtype=float)
        J = self.n_classes
        for i in range(J):
            for j in range(J):
                if i == 0:
                    ii.append(i); jj.append(j); kk.append(j); ww.append(1.0); continue
                if j == 0:
                    ii.append(i); jj.append(j); kk.append(i); ww.append(1.0); continue
                if i == self.high_index or j == self.high_index:
                    ii.append(i); jj.append(j); kk.append(self.high_index); ww.append(1.0); continue
                z = float(piv[i] + piv[j])
                if z >= 1.0:
                    ii.append(i); jj.append(j); kk.append(self.high_index); ww.append(1.0); continue
                if z <= 0.0:
                    ii.append(i); jj.append(j); kk.append(0); ww.append(1.0); continue
                hi = int(np.searchsorted(piv, z, side="left"))
                if hi <= 0:
                    ii.append(i); jj.append(j); kk.append(0); ww.append(1.0); continue
                if hi >= J:
                    ii.append(i); jj.append(j); kk.append(self.high_index); ww.append(1.0); continue
                if abs(float(piv[hi]) - z) <= 1e-14 * max(1.0, z):
                    ii.append(i); jj.append(j); kk.append(hi); ww.append(1.0); continue
                lo = hi - 1
                plo = float(piv[lo]); phi = float(piv[hi])
                if phi <= plo:
                    ii.append(i); jj.append(j); kk.append(hi); ww.append(1.0); continue
                whi = (z - plo) / (phi - plo)
                whi = min(max(float(whi), 0.0), 1.0)
                wlo = 1.0 - whi
                if wlo > 1e-15:
                    ii.append(i); jj.append(j); kk.append(lo); ww.append(wlo)
                if whi > 1e-15:
                    ii.append(i); jj.append(j); kk.append(hi); ww.append(whi)
        self._uniform_i = np.asarray(ii, dtype=np.int32)
        self._uniform_j = np.asarray(jj, dtype=np.int32)
        self._uniform_k = np.asarray(kk, dtype=np.int32)
        self._uniform_w = np.asarray(ww, dtype=float)

    def _build_addition_index(self) -> np.ndarray:
        J = self.n_classes
        out = np.empty((J, J), dtype=np.int32)
        for i in range(J):
            for j in range(J):
                if i == 0:
                    out[i, j] = j
                elif j == 0:
                    out[i, j] = i
                elif i == self.high_index or j == self.high_index:
                    out[i, j] = self.high_index
                else:
                    out[i, j] = self._class_for_p(self.p_rep[i] + self.p_rep[j])
        return out

    def _conv(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if str(self.config.addition_rule).lower() in ("uniform", "linear") and self._uniform_i is not None:
            if _sparse_conv_accel is not None:
                return _sparse_conv_accel(self._uniform_i, self._uniform_j, self._uniform_k, self._uniform_w, x, y, self.n_classes)
            vals = self._uniform_w * x[self._uniform_i] * y[self._uniform_j]
            return np.bincount(self._uniform_k, weights=vals, minlength=self.n_classes)
        if _point_conv_accel is not None:
            return _point_conv_accel(self._add_index, x, y, self.n_classes)
        idx = self._add_index.ravel()
        vals = (x[:, None] * y[None, :]).ravel()
        return np.bincount(idx, weights=vals, minlength=self.n_classes)

    def _initial_Q(self) -> np.ndarray:
        q = np.zeros(self.n_classes, dtype=float)
        p0 = 1.0 / max(float(self.params.N_const), 1.0)
        # On coarse log grids, assigning the initial single cell to one bin can
        # move its represented frequency upward by a nontrivial multiplicative
        # factor.  Use the same fixed-pivot interpolation as the birth operator
        # so the represented initial frequency is preserved exactly.  The exact
        # small-N integer grid still lands on a single integer pivot.
        if self._integer_N is not None:
            q[self._class_for_p(p0)] = 1.0
            return q
        piv = np.asarray(self.p_rep, dtype=float)
        if p0 <= piv[0]:
            q[0] = 1.0
            return q
        hi = int(np.searchsorted(piv, p0, side="left"))
        if hi >= self.n_classes:
            q[self.high_index] = 1.0
            return q
        if abs(float(piv[hi]) - p0) <= 1e-14 * max(1.0, p0):
            q[hi] = 1.0
            return q
        lo = max(0, hi - 1)
        plo = float(piv[lo]); phi = float(piv[hi])
        if phi <= plo:
            q[hi] = 1.0
            return q
        whi = min(max((p0 - plo) / (phi - plo), 0.0), 1.0)
        q[lo] = 1.0 - whi
        q[hi] = whi
        return q

    def _solve_QAB(self, s: float, rtol: Optional[float], ode_method: Optional[str]):
        p = self.params
        cfg = self.config
        J = self.n_classes
        rtol_val = float(cfg.rtol if rtol is None else rtol)
        atol = max(np.finfo(float).eps, rtol_val * cfg.atol_factor)
        method = ode_method or cfg.ode_method
        es = math.exp(float(s))
        c = es - 1.0
        Q0 = self._initial_Q()
        A0 = np.zeros_like(Q0)
        B0 = np.zeros_like(Q0)
        y0 = np.concatenate([Q0, A0, B0])

        lam = float(p.lam); delta = float(p.delta); u = float(p.u)
        rho = self.rho

        def rhs(t, flat):
            Q = flat[:J]
            A = flat[J:2*J]
            B = flat[2*J:]
            conv_QQ = self._conv(Q, Q)
            conv_QA = self._conv(Q, A)
            conv_AA = self._conv(A, A)
            dQ = lam * (conv_QQ - Q) - delta * Q + u * c * rho * Q
            dQ[0] += delta
            dA = lam * (2.0 * conv_QA - A) - delta * A + u * (es * rho * Q + c * rho * A)
            dB = lam * (2.0 * conv_AA + 2.0 * self._conv(Q, B) - B) - delta * B + u * (es * rho * Q + 2.0 * es * rho * A + c * rho * B)
            return np.concatenate([dQ, dA, dB])

        sol = solve_ivp(rhs, (0.0, float(p.T)), y0, method=method, rtol=rtol_val, atol=atol)
        Y = sol.y[:, -1]
        return Y[:J], Y[J:2*J], Y[2*J:], sol

    def K012(self, s: float, rtol: Optional[float] = None, ode_method: Optional[str] = None):
        key = (round(float(s), 12), float(self.config.rtol if rtol is None else rtol), ode_method or self.config.ode_method)
        if key in self._K_cache:
            vals, meta = self._K_cache[key]
            return vals[0], vals[1], vals[2], {**meta, "cached": 1.0}
        t0 = time.perf_counter()
        p = self.params
        Q, A, B, sol = self._solve_QAB(float(s), rtol, ode_method)
        es = math.exp(float(s))
        c = es - 1.0
        theta = float(p.theta_f)
        g = theta * c * self.rho
        scale = float(np.max(g)) if g.size else 0.0
        # Keep scale nonnegative to preserve conditioning subtraction logic.
        scale = max(0.0, scale)
        w0 = np.exp(g - scale)
        gp = theta * es * self.rho
        w1 = gp * w0
        w2 = (gp + gp * gp) * w0
        G0 = float(np.dot(Q, w0))
        G1 = float(np.dot(A, w0) + np.dot(Q, w1))
        G2 = float(np.dot(B, w0) + 2.0 * np.dot(A, w1) + np.dot(Q, w2))
        if p.condition_on_survival:
            den = max(1.0 - self.ext_prob, np.finfo(float).tiny)
            # Extinct state is class 0, rho=0, w0=exp(-scale).
            G0c = G0 - self.ext_prob * math.exp(-scale)
            G0c = max(float(G0c), np.finfo(float).tiny)
            K0 = scale + math.log(G0c) - math.log(den)
        else:
            G0c = max(float(G0), np.finfo(float).tiny)
            K0 = scale + math.log(G0c)
        K1 = float(G1 / G0c) if G0c > 0 and np.isfinite(G1) else math.nan
        K2raw = float(G2 / G0c - K1 * K1) if G0c > 0 and np.isfinite(G2) and np.isfinite(K1) else math.nan
        K2 = float(max(K2raw, 1e-14)) if np.isfinite(K2raw) else math.nan
        finite = bool(sol.success and np.isfinite(K0) and np.isfinite(K1) and np.isfinite(K2))
        meta: Dict[str, float | str] = {
            "engine": "adaptive_frequency_volume_v29_uniform_candidate",
            "seconds": float(time.perf_counter() - t0),
            "ode_success": float(bool(sol.success)),
            "nfev": float(sol.nfev),
            "n_classes": float(self.n_classes),
            "addition_rule": str(self.config.addition_rule),
            "uniform_addition_nnz": float(0 if self._uniform_k is None else self._uniform_k.size),
            "n_finite_bins": float(len(self.finite_rep)),
            "N_const": float(p.N_const),
            "alpha": float(p.alpha),
            "depth": float(p.depth),
            "read_threshold_count_h": float(read_threshold(p.alpha, p.depth)),
            "theta_f": float(p.theta_f),
            "founder_log_scale": float(scale),
            "min_positive_edge": float(self.edges[1] if self.edges.size > 1 else 1.0),
            "transition_alpha_eff": float(read_threshold(p.alpha, p.depth) / max(int(p.depth), 1)),
            "integer_validation_grid_N": float(self._integer_N or 0),
            "certified_candidate": float(finite),
            "finite_depth_exact_at_bin_resolution": 1.0,
            "cached": 0.0,
        }
        vals = (float(K0), float(K1), float(K2))
        self._K_cache[key] = (vals, meta)
        return vals[0], vals[1], vals[2], meta

    def cdf_lr(self, k: int, *, continuity: float = 0.5, rtol: Optional[float] = None,
               max_abs_s: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
        t0 = time.perf_counter()
        x = max(float(k) + float(continuity), 0.0)
        _, mu, var, m0 = self.K012(0.0, rtol=rtol)
        sigma = math.sqrt(max(float(var), 1e-14))
        if not np.isfinite(mu) or not np.isfinite(var):
            return math.nan, {"method": "failed_nonfinite_moments", "seconds": time.perf_counter() - t0, **m0}
        if abs(x - mu) < 1e-7 * max(1.0, sigma):
            return float(norm.cdf((x - mu) / sigma)), {"method": "normal_at_mean", "mean": mu, "var": var, "seconds": time.perf_counter() - t0, **m0}

        def f(sv):
            _, k1, _, _ = self.K012(float(sv), rtol=rtol)
            return k1 - x

        max_s = float(self.config.max_abs_s if max_abs_s is None else max_abs_s)
        # Saddlepoints in the production regime are often tiny because the
        # mutation-count variance can be enormous.  Evaluating K'(s) at a fixed
        # endpoint such as +/-12 can overflow long before it is needed.  Start
        # from the normal-theory estimate s ~= (x-mu)/var and expand only until
        # the root is bracketed.
        direction = 1.0 if x >= mu else -1.0
        s0 = abs((x - mu) / max(float(var), 1e-14))
        step = min(max_s, max(1e-10, 2.0 * s0 + 1e-10))
        lo, hi = (0.0, direction * step) if direction > 0 else (direction * step, 0.0)
        try:
            f0 = f(0.0)
            f_edge = f(hi if direction > 0 else lo)
            expansion_rounds = 0
            while np.isfinite(f_edge) and f0 * f_edge > 0.0 and step < max_s:
                step = min(max_s, step * 2.0)
                lo, hi = (0.0, direction * step) if direction > 0 else (direction * step, 0.0)
                f_edge = f(hi if direction > 0 else lo)
                expansion_rounds += 1
            flo, fhi = (f0, f_edge) if direction > 0 else (f_edge, f0)
            if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0.0:
                # Controlled normal fallback; metadata says so.
                F = float(norm.cdf((x - mu) / sigma))
                return F, {"method": "normal_fallback_unbracketed", "mean": mu, "var": var, "bracket_lo": lo, "bracket_hi": hi, "f_lo": float(flo), "f_hi": float(fhi), "saddle_expansion_rounds": float(expansion_rounds), "seconds": time.perf_counter() - t0, **m0}
            shat = brentq(f, lo, hi, xtol=1e-9, rtol=1e-9, maxiter=80)
            K, _, K2, ms = self.K012(shat, rtol=rtol)
            w_arg = 2.0 * (shat * x - K)
            w = math.copysign(math.sqrt(max(w_arg, 0.0)), shat)
            v = shat * math.sqrt(max(K2, 1e-14))
            if abs(w) < 1e-8 or abs(v) < 1e-12:
                F = float(norm.cdf((x - mu) / sigma))
                method = "normal_fallback_singular_lr"
            else:
                F = float(norm.cdf(w) + norm.pdf(w) * (1.0 / w - 1.0 / v))
                F = float(np.clip(F, 0.0, 1.0))
                method = "lugannani_rice_frequency_volume"
            return F, {"method": method, "saddle_s": float(shat), "saddle_expansion_rounds": float(expansion_rounds), "mean": mu, "var": var, "seconds": time.perf_counter() - t0, **ms}
        except Exception as exc:
            F = float(norm.cdf((x - mu) / sigma))
            return F, {"method": "normal_fallback_exception", "exception": str(exc), "mean": mu, "var": var, "seconds": time.perf_counter() - t0, **m0}


    def cdf_lr_many(self, k_values: Sequence[int], *, continuity: float = 0.5,
                    rtol: Optional[float] = None, max_abs_s: Optional[float] = None) -> List[Dict[str, float | str]]:
        """Evaluate several endpoint CDFs with shared K012 cache.

        This is a convenience/performance helper for production endpoint tables.
        The expensive ODE solves are cached inside the engine, so evaluating a
        sorted endpoint list reuses moments and previously visited saddlepoints
        as much as possible.
        """
        rows: List[Dict[str, float | str]] = []
        for k in sorted({int(x) for x in k_values}):
            F, meta = self.cdf_lr(int(k), continuity=continuity, rtol=rtol, max_abs_s=max_abs_s)
            rows.append({"k": float(k), "cdf": float(F), **meta})
        return rows


def validate_frequency_volume_against_exact(
    params: ModelParams,
    *,
    s_values: Sequence[float] = (0.0, 0.01, 0.05),
    n_bins: int = 240,
    exact_M: Optional[int] = None,
    exact_tail_tol: float = 1e-12,
) -> List[Dict[str, float | str]]:
    """Compare v27 frequency-volume K012 values against exact small-N engine."""
    # Use the exact capped integer grid when possible so this helper can test
    # the finite-volume equations independently from frequency-grid coarsening.
    cfg = FrequencyVolumeConfig(n_bins=int(n_bins), integer_grid_max_N=max(0, int(math.ceil(params.N_const)) if params.N_const <= max(10000, n_bins) else 0))
    fv = AdaptiveFrequencyVolumeEngine(params, config=cfg)
    ex = ExactSeriesEngine(params, M=exact_M, tail_tol=exact_tail_tol, M_cap=max(10000, int(4 * params.N_const)))
    rows: List[Dict[str, float | str]] = []
    for s in s_values:
        K0e, K1e, K2e, me = ex.K012(float(s), rtol=2e-7, ode_method="DOP853")
        K0v, K1v, K2v, mv = fv.K012(float(s), rtol=2e-7)
        rows.append({
            "s": float(s),
            "exact_K": float(K0e), "fv_K": float(K0v), "abs_K_error": float(abs(K0v - K0e)),
            "exact_K1": float(K1e), "fv_K1": float(K1v), "abs_K1_error": float(abs(K1v - K1e)),
            "exact_K2": float(K2e), "fv_K2": float(K2v), "abs_K2_error": float(abs(K2v - K2e)),
            "fv_n_classes": float(fv.n_classes),
            "fv_certified_candidate": float(mv.get("certified_candidate", 0.0)),
            "exact_truncation_ok": float(me.get("truncation_ok", float(ex.truncation_ok))),
        })
    return rows


def frequency_volume_convergence_ladder(
    params: ModelParams,
    *,
    bin_grid: Sequence[int] = (60, 100, 160, 240),
    s_values: Sequence[float] = (0.0, 0.01, 0.05),
    base_config: Optional[FrequencyVolumeConfig] = None,
) -> List[Dict[str, float | str]]:
    """Resolution ladder for v27 finite-volume K012 values."""
    base = base_config or FrequencyVolumeConfig()
    out: List[Dict[str, float | str]] = []
    prev: Dict[float, Tuple[float, float, float]] = {}
    for nb in bin_grid:
        cfg = FrequencyVolumeConfig(**{**asdict(base), "n_bins": int(nb)})
        eng = AdaptiveFrequencyVolumeEngine(params, config=cfg)
        for s in s_values:
            K0, K1, K2, meta = eng.K012(float(s))
            pk = prev.get(float(s))
            out.append({
                "n_bins": float(nb),
                "n_classes": float(eng.n_classes),
                "s": float(s),
                "K": float(K0), "K1": float(K1), "K2": float(K2),
                "delta_K_vs_prev": float(abs(K0 - pk[0])) if pk else math.nan,
                "delta_K1_vs_prev": float(abs(K1 - pk[1])) if pk else math.nan,
                "delta_K2_vs_prev": float(abs(K2 - pk[2])) if pk else math.nan,
                "certified_candidate": float(meta.get("certified_candidate", 0.0)),
                "seconds": float(meta.get("seconds", math.nan)),
            })
            prev[float(s)] = (K0, K1, K2)
    return out
