"""v51 compiled finite-depth frequency CGF engine.

This module targets the main hard-regime bottleneck: repeated SciPy solve_ivp
calls for the frequency-volume CGF system.  It reuses the v33/v44 frequency grid
and fixed-pivot addition operator, but integrates the Q/A/B ODE with a compiled
fixed-step RK4 scheme.  The method is still the same grid-refinable finite-depth
approximation; this is an implementation/backend acceleration, not a new
heuristic distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math, time

import numpy as np
from scipy.stats import norm

from .distribution import ModelParams, extinction_probability
from .frequency_volume import AdaptiveFrequencyVolumeEngine, FrequencyVolumeConfig

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco


@dataclass(frozen=True)
class FastFrequencyConfig:
    n_bins: int = 120
    # Steps per unit time.  Accuracy should be checked with a step ladder.
    steps_per_time: float = 8.0
    min_steps: int = 32
    max_steps: int = 1200
    # Underlying grid config mirrors the convergent-frequency production path.
    transition_width_multiplier: float = 12.0
    transition_fraction: float = 0.35
    log_fraction: float = 0.50
    linear_fraction: float = 0.05
    upper_tail_fraction: float = 0.10
    max_abs_s: float = 12.0
    addition_rule: str = "linear"
    representative: str = "geometric"


def make_fast_grid_config(cfg: FastFrequencyConfig) -> FrequencyVolumeConfig:
    return FrequencyVolumeConfig(
        n_bins=int(cfg.n_bins),
        transition_width_multiplier=float(cfg.transition_width_multiplier),
        transition_fraction=float(cfg.transition_fraction),
        log_fraction=float(cfg.log_fraction),
        linear_fraction=float(cfg.linear_fraction),
        upper_tail_fraction=float(cfg.upper_tail_fraction),
        representative=str(cfg.representative),
        addition_rule=str(cfg.addition_rule),
        ode_method="DOP853",
        rtol=1e-5,
        atol_factor=1e-2,
        max_abs_s=float(cfg.max_abs_s),
        integer_grid_max_N=0,
    )


@njit(cache=True)
def _conv_inplace(ii, jj, kk, ww, x, y, out):
    for a in range(out.size):
        out[a] = 0.0
    for n in range(kk.size):
        out[kk[n]] += ww[n] * x[ii[n]] * y[jj[n]]


@njit(cache=True)
def _rhs_inplace(Q, A, B, dQ, dA, dB, ii, jj, kk, ww, rho, lam, delta, u, es, c,
                 tmpQQ, tmpQA, tmpAA, tmpQB):
    J = Q.size
    _conv_inplace(ii, jj, kk, ww, Q, Q, tmpQQ)
    _conv_inplace(ii, jj, kk, ww, Q, A, tmpQA)
    _conv_inplace(ii, jj, kk, ww, A, A, tmpAA)
    _conv_inplace(ii, jj, kk, ww, Q, B, tmpQB)
    for i in range(J):
        rr = rho[i]
        dQ[i] = lam * (tmpQQ[i] - Q[i]) - delta * Q[i] + u * c * rr * Q[i]
        dA[i] = lam * (2.0 * tmpQA[i] - A[i]) - delta * A[i] + u * (es * rr * Q[i] + c * rr * A[i])
        dB[i] = lam * (2.0 * tmpAA[i] + 2.0 * tmpQB[i] - B[i]) - delta * B[i] + u * (es * rr * Q[i] + 2.0 * es * rr * A[i] + c * rr * B[i])
    dQ[0] += delta


@njit(cache=True)
def _rk4_integrate(Q0, A0, B0, ii, jj, kk, ww, rho, lam, delta, u, es, T, n_steps):
    J = Q0.size
    Q = Q0.copy(); A = A0.copy(); B = B0.copy()
    c = es - 1.0
    dt = T / n_steps
    # stage arrays
    k1Q=np.empty(J); k1A=np.empty(J); k1B=np.empty(J)
    k2Q=np.empty(J); k2A=np.empty(J); k2B=np.empty(J)
    k3Q=np.empty(J); k3A=np.empty(J); k3B=np.empty(J)
    k4Q=np.empty(J); k4A=np.empty(J); k4B=np.empty(J)
    tQ=np.empty(J); tA=np.empty(J); tB=np.empty(J)
    tmpQQ=np.empty(J); tmpQA=np.empty(J); tmpAA=np.empty(J); tmpQB=np.empty(J)
    for step in range(n_steps):
        _rhs_inplace(Q,A,B,k1Q,k1A,k1B,ii,jj,kk,ww,rho,lam,delta,u,es,c,tmpQQ,tmpQA,tmpAA,tmpQB)
        for i in range(J):
            tQ[i]=Q[i]+0.5*dt*k1Q[i]; tA[i]=A[i]+0.5*dt*k1A[i]; tB[i]=B[i]+0.5*dt*k1B[i]
        _rhs_inplace(tQ,tA,tB,k2Q,k2A,k2B,ii,jj,kk,ww,rho,lam,delta,u,es,c,tmpQQ,tmpQA,tmpAA,tmpQB)
        for i in range(J):
            tQ[i]=Q[i]+0.5*dt*k2Q[i]; tA[i]=A[i]+0.5*dt*k2A[i]; tB[i]=B[i]+0.5*dt*k2B[i]
        _rhs_inplace(tQ,tA,tB,k3Q,k3A,k3B,ii,jj,kk,ww,rho,lam,delta,u,es,c,tmpQQ,tmpQA,tmpAA,tmpQB)
        for i in range(J):
            tQ[i]=Q[i]+dt*k3Q[i]; tA[i]=A[i]+dt*k3A[i]; tB[i]=B[i]+dt*k3B[i]
        _rhs_inplace(tQ,tA,tB,k4Q,k4A,k4B,ii,jj,kk,ww,rho,lam,delta,u,es,c,tmpQQ,tmpQA,tmpAA,tmpQB)
        for i in range(J):
            Q[i] += (dt/6.0)*(k1Q[i]+2.0*k2Q[i]+2.0*k3Q[i]+k4Q[i])
            A[i] += (dt/6.0)*(k1A[i]+2.0*k2A[i]+2.0*k3A[i]+k4A[i])
            B[i] += (dt/6.0)*(k1B[i]+2.0*k2B[i]+2.0*k3B[i]+k4B[i])
            # Damp tiny negative roundoff; leave serious instability visible.
            if Q[i] < 0.0 and Q[i] > -1e-12:
                Q[i]=0.0
    return Q,A,B


class FastFrequencyCGFEngine:
    """Compiled RK backend used for moment preflight and saturation certificates."""
    def __init__(self, params: ModelParams, *, config: Optional[FastFrequencyConfig]=None):
        self.params = params
        self.config = config or FastFrequencyConfig()
        self.grid_engine = AdaptiveFrequencyVolumeEngine(params, config=make_fast_grid_config(self.config))
        self.rho = np.asarray(self.grid_engine.rho, dtype=np.float64)
        self.ext_prob = extinction_probability(params.lam, params.delta, params.T)
        if self.grid_engine._uniform_i is None:
            raise RuntimeError("fast backend requires sparse linear/uniform addition arrays")
        self.ii = np.asarray(self.grid_engine._uniform_i, dtype=np.int32)
        self.jj = np.asarray(self.grid_engine._uniform_j, dtype=np.int32)
        self.kk = np.asarray(self.grid_engine._uniform_k, dtype=np.int32)
        self.ww = np.asarray(self.grid_engine._uniform_w, dtype=np.float64)
        self.Q0 = np.asarray(self.grid_engine._initial_Q(), dtype=np.float64)
        self.A0 = np.zeros_like(self.Q0)
        self.B0 = np.zeros_like(self.Q0)
        self.n_classes = self.Q0.size
        self._cache: Dict[Tuple[float,int,float], Tuple[Tuple[float,float,float], Dict[str,float|str]]] = {}
        self._raw_cache: Dict[Tuple[float,int], Tuple[np.ndarray,np.ndarray,np.ndarray,Dict[str,float|str]]] = {}
        self.n_steps = int(min(max(int(math.ceil(float(params.T)*float(self.config.steps_per_time))), int(self.config.min_steps)), int(self.config.max_steps)))

    def _raw_QAB(self, s: float):
        raw_key=(round(float(s),12), int(self.n_steps))
        if raw_key in self._raw_cache:
            Q,A,B,meta=self._raw_cache[raw_key]
            return Q,A,B,{**meta,"raw_cached":1.0}
        t0=time.perf_counter()
        p=self.params
        es=math.exp(float(s))
        Q,A,B = _rk4_integrate(self.Q0,self.A0,self.B0,self.ii,self.jj,self.kk,self.ww,self.rho,
                               float(p.lam),float(p.delta),float(p.u),float(es),float(p.T),int(self.n_steps))
        meta={"raw_seconds":float(time.perf_counter()-t0),"raw_cached":0.0}
        self._raw_cache[raw_key]=(Q,A,B,meta)
        return Q,A,B,meta

    def K012_theta(self, s: float, theta_f: float):
        key=(round(float(s),12), int(self.n_steps), round(float(theta_f),12))
        if key in self._cache:
            vals, meta = self._cache[key]
            return vals[0], vals[1], vals[2], {**meta, "cached":1.0}
        t0=time.perf_counter()
        p=self.params
        es=math.exp(float(s)); c=es-1.0
        Q,A,B,raw_meta = self._raw_QAB(float(s))
        theta=float(theta_f)
        g=theta*c*self.rho
        scale=max(0.0, float(np.max(g)) if g.size else 0.0)
        w0=np.exp(g-scale)
        gp=theta*es*self.rho
        w1=gp*w0
        w2=(gp+gp*gp)*w0
        G0=float(np.dot(Q,w0)); G1=float(np.dot(A,w0)+np.dot(Q,w1)); G2=float(np.dot(B,w0)+2*np.dot(A,w1)+np.dot(Q,w2))
        if p.condition_on_survival:
            den=max(1.0-self.ext_prob, np.finfo(float).tiny)
            G0c=G0 - self.ext_prob*math.exp(-scale)
            G0c=max(float(G0c), np.finfo(float).tiny)
            K0=scale+math.log(G0c)-math.log(den)
        else:
            G0c=max(float(G0), np.finfo(float).tiny); K0=scale+math.log(G0c)
        K1=float(G1/G0c) if G0c>0 and np.isfinite(G1) else math.nan
        K2raw=float(G2/G0c - K1*K1) if G0c>0 and np.isfinite(G2) and np.isfinite(K1) else math.nan
        K2=float(max(K2raw,1e-14)) if np.isfinite(K2raw) else math.nan
        finite=bool(np.isfinite(K0) and np.isfinite(K1) and np.isfinite(K2))
        meta={"engine":"fast_frequency_rk4_v52_raw_theta_cache","seconds":float(time.perf_counter()-t0),"n_steps":float(self.n_steps),
              "n_classes":float(self.n_classes),"n_bins":float(self.config.n_bins),"numba_available":float(NUMBA_AVAILABLE),
              "theta_f":float(theta_f),"founder_log_scale":float(scale),"certified_candidate":float(finite),
              "finite_depth_exact_at_bin_resolution":1.0,"cached":0.0, **raw_meta}
        vals=(float(K0),float(K1),float(K2)); self._cache[key]=(vals,meta)
        return vals[0], vals[1], vals[2], meta

    def K012(self, s: float, rtol: Optional[float]=None, ode_method: Optional[str]=None):
        return self.K012_theta(float(s), float(self.params.theta_f))

    def cdf_lr(self, k:int, *, continuity:float=0.5, rtol:Optional[float]=None, max_abs_s:Optional[float]=None):
        from scipy.optimize import brentq
        t0=time.perf_counter(); x=max(float(k)+float(continuity),0.0)
        _,mu,var,m0=self.K012(0.0)
        sig=math.sqrt(max(float(var),1e-14))
        # v80 production guard: if the requested CDF is in an extreme right
        # endpoint and the finite-depth mean already gives a rigorous Markov
        # certificate, skip saddlepoint bracketing entirely.  This prevents
        # rare expensive/fragile LR searches in the small-mean, moderate-k
        # regime while preserving a one-sided finite-depth guarantee.
        if x >= float(mu) and np.isfinite(mu) and float(mu) >= 0.0:
            markov_u = float(mu) / max(float(k) + 1.0, 1.0)
            if markov_u <= 1e-12:
                return 1.0, {"method":"fast_markov_right_tail_certified_pre_lr","mean":mu,"var":var,"markov_tail_bound":float(markov_u),"seconds":time.perf_counter()-t0,**m0}
        if abs(x-float(mu)) < 1e-7*max(1.0,sig):
            F=float(norm.cdf((x-float(mu))/sig)); return F,{"method":"fast_normal_at_mean","mean":mu,"var":var,"seconds":time.perf_counter()-t0,**m0}
        direction=1.0 if x>=mu else -1.0
        max_s=float(self.config.max_abs_s if max_abs_s is None else max_abs_s)
        def f(sv):
            _,k1,_,_=self.K012(float(sv)); return k1-x
        # Robust bracket search.  The older v52 logic jumped directly to a
        # Newton-like step; in large-theta/high-growth regimes K(s) may overflow
        # at that step even though a valid finite sign change exists at a
        # smaller |s|.  Scan outward and ignore non-finite points instead of
        # immediately falling back to a normal approximation.
        f0=f(0.0)
        lo=hi=0.0; found=False; prev_s=0.0; prev_f=f0
        grid=np.concatenate((np.array([1e-10,1e-8,1e-6,1e-4,1e-3,1e-2,5e-2,1e-1,2e-1,5e-1,1.0]),
                             np.linspace(1.25, max_s, 12)))
        for g in grid:
            sv=direction*float(g)
            fg=f(sv)
            if not np.isfinite(fg):
                continue
            if np.isfinite(prev_f) and prev_f*fg <= 0.0:
                lo,hi=(prev_s,sv) if prev_s < sv else (sv,prev_s)
                found=True
                break
            prev_s,prev_f=sv,fg
        if not found:
            F=float(norm.cdf((x-float(mu))/sig)); return F,{"method":"fast_normal_fallback_unbracketed","mean":mu,"var":var,"seconds":time.perf_counter()-t0,**m0}
        sh=brentq(f,lo,hi,xtol=1e-8,rtol=1e-8,maxiter=60)
        K,_,K2,ms=self.K012(sh)
        w_arg=2.0*(sh*x-K); w=math.copysign(math.sqrt(max(w_arg,0.0)),sh); v=sh*math.sqrt(max(K2,1e-14))
        if abs(w)<1e-8 or abs(v)<1e-12:
            F=float(norm.cdf((x-float(mu))/sig)); method="fast_normal_fallback_singular_lr"
        else:
            F=float(norm.cdf(w)+norm.pdf(w)*(1.0/w-1.0/v)); F=float(np.clip(F,0,1)); method="fast_lugannani_rice"
        return F,{"method":method,"mean":mu,"var":var,"saddle_s":float(sh),"seconds":time.perf_counter()-t0,**ms}
