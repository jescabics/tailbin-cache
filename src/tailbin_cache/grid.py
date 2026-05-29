from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import json
import math

import numpy as np


def logspace_endpoints(lo: float, hi: float, n: int) -> List[float]:
    if n <= 1:
        return [float(lo)]
    return [float(x) for x in np.exp(np.linspace(math.log(float(lo)), math.log(float(hi)), int(n)))]


def linspace_endpoints(lo: float, hi: float, n: int) -> List[float]:
    if n <= 1:
        return [float(lo)]
    return [float(x) for x in np.linspace(float(lo), float(hi), int(n))]


def default_alphas(n_alpha: int = 20, lo: float = 0.05, hi: float = 1.0) -> List[float]:
    """Default cumulative-tail cutoffs: equally spaced on log-frequency scale."""
    return logspace_endpoints(lo, hi, n_alpha)


@dataclass(frozen=True)
class ParameterPoint:
    """One biological/numerical parameter point, excluding alpha."""

    R: float
    T: float
    theta_f: float
    N: float
    depth: int = 120
    u: float = 20.0
    ploidy_factor: float = 2.0
    lam: float = 1.0
    condition_on_survival: bool = True

    @property
    def delta(self) -> float:
        # Existing finite-depth backend uses R = 1 - delta / lambda.
        return float(self.lam) * (1.0 - float(self.R))

    @property
    def Tb(self) -> float:
        """Founder/background time implied by theta_f = u * Tb."""
        if float(self.u) == 0.0:
            return math.inf
        return float(self.theta_f) / float(self.u)

    @property
    def total_age(self) -> float:
        """Total age constraint coordinate: T + Tb."""
        return float(self.T) + float(self.Tb)

    @property
    def effective_N(self) -> float:
        """Denominator used for latent mutation frequency.

        For diploid genomes, one mutant DNA molecule in N cells has observed
        allele frequency m/(2N), so the default ploidy_factor is 2.
        """
        return float(self.ploidy_factor) * float(self.N)

    def key(self) -> str:
        return (
            f"R{self.R:.8g}_T{self.T:.8g}_theta{self.theta_f:.8g}_"
            f"N{self.N:.8g}_P{self.ploidy_factor:.8g}_D{self.depth}_u{self.u:.8g}"
        ).replace(".", "p").replace("+", "").replace("-", "m")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["delta"] = self.delta
        d["Tb"] = self.Tb
        d["total_age"] = self.total_age
        d["effective_N"] = self.effective_N
        return d


@dataclass(frozen=True)
class CacheGrid:
    """Programmable grid specification for table generation."""

    R_values: Sequence[float]
    T_values: Sequence[float]
    theta_values: Sequence[float]
    N_values: Sequence[float]
    alphas: Sequence[float]
    depth_values: Sequence[int] = (120,)
    u: float = 20.0
    ploidy_factor: float = 2.0
    lam: float = 1.0
    condition_on_survival: bool = True
    max_age: float = 100.0
    enforce_age_constraint: bool = True
    age_constraint_mode: str = "upper_bound"
    age_exact: Optional[float] = None
    paired_T_theta_values: Optional[Sequence[Tuple[float, float]]] = None

    def theta_is_valid_for_T(self, theta: float, T: float) -> bool:
        """Return True if the age constraint accepts this T/theta_f pair."""
        mode = str(self.age_constraint_mode or "upper_bound")
        if not bool(self.enforce_age_constraint) and mode != "exact":
            return True
        if float(self.u) == 0.0:
            return float(theta) <= 0.0
        Tb = float(theta) / float(self.u)
        total_age = float(T) + Tb
        if mode == "none":
            return True
        if mode == "exact":
            if self.age_exact is None:
                raise ValueError("age_constraint_mode='exact' requires age_exact")
            return abs(total_age - float(self.age_exact)) <= 1e-9
        if mode == "upper_bound":
            return total_age <= float(self.max_age) + 1e-12
        raise ValueError(f"unknown age_constraint_mode {self.age_constraint_mode!r}")

    def valid_theta_values_for_T(self, T: float) -> List[float]:
        if self.paired_T_theta_values is not None:
            return [
                float(theta)
                for paired_T, theta in self.paired_T_theta_values
                if abs(float(paired_T) - float(T)) <= 1e-9
            ]
        return [float(theta) for theta in self.theta_values if self.theta_is_valid_for_T(float(theta), float(T))]

    def parameter_points(self) -> Iterator[ParameterPoint]:
        if self.paired_T_theta_values is not None:
            for R, N, depth, pair in product(self.R_values, self.N_values, self.depth_values, self.paired_T_theta_values):
                T, theta = pair
                if not self.theta_is_valid_for_T(float(theta), float(T)):
                    continue
                yield ParameterPoint(
                    R=float(R), T=float(T), theta_f=float(theta), N=float(N), depth=int(depth),
                    u=float(self.u), ploidy_factor=float(self.ploidy_factor), lam=float(self.lam), condition_on_survival=bool(self.condition_on_survival),
                )
            return
        for R, T, theta, N, depth in product(self.R_values, self.T_values, self.theta_values, self.N_values, self.depth_values):
            if not self.theta_is_valid_for_T(float(theta), float(T)):
                continue
            yield ParameterPoint(
                R=float(R), T=float(T), theta_f=float(theta), N=float(N), depth=int(depth),
                u=float(self.u), ploidy_factor=float(self.ploidy_factor), lam=float(self.lam), condition_on_survival=bool(self.condition_on_survival),
            )

    def table_specs(self) -> Iterator[tuple[ParameterPoint, float, int]]:
        for p in self.parameter_points():
            for ai, alpha in enumerate(self.alphas):
                yield p, float(alpha), int(ai)

    @property
    def n_parameter_points(self) -> int:
        return sum(1 for _ in self.parameter_points())

    @property
    def n_tables(self) -> int:
        return self.n_parameter_points * len(self.alphas)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "R_values": [float(x) for x in self.R_values],
            "T_values": [float(x) for x in self.T_values],
            "theta_values": [float(x) for x in self.theta_values],
            "N_values": [float(x) for x in self.N_values],
            "alphas": [float(x) for x in self.alphas],
            "depth_values": [int(x) for x in self.depth_values],
            "u": float(self.u),
            "ploidy_factor": float(self.ploidy_factor),
            "lam": float(self.lam),
            "condition_on_survival": bool(self.condition_on_survival),
            "max_age": float(self.max_age),
            "enforce_age_constraint": bool(self.enforce_age_constraint),
            "age_constraint_mode": str(self.age_constraint_mode),
            "age_exact": None if self.age_exact is None else float(self.age_exact),
            "paired_T_theta_values": None if self.paired_T_theta_values is None else [
                {"T": float(T), "theta_f": float(theta), "Tb": float(theta) / float(self.u) if float(self.u) != 0.0 else math.inf}
                for T, theta in self.paired_T_theta_values
            ],
        }


def grid_from_dict(cfg: Dict[str, Any]) -> CacheGrid:
    g = cfg.get("grid", cfg)

    def values(name: str, default: Optional[Sequence[float]] = None, *, log: bool = False) -> List[float]:
        if name in g and isinstance(g[name], list):
            return [float(x) for x in g[name]]
        count_name = name.replace("_values", "_count")
        min_name = name.replace("_values", "_min")
        max_name = name.replace("_values", "_max")
        if count_name in g and min_name in g and max_name in g:
            n = int(g[count_name])
            return logspace_endpoints(float(g[min_name]), float(g[max_name]), n) if log else linspace_endpoints(float(g[min_name]), float(g[max_name]), n)
        if default is None:
            raise KeyError(f"Missing grid field {name} or {count_name}/{min_name}/{max_name}")
        return [float(x) for x in default]

    def has_values_axis(name: str) -> bool:
        count_name = name.replace("_values", "_count")
        min_name = name.replace("_values", "_min")
        max_name = name.replace("_values", "_max")
        return name in g or (count_name in g and min_name in g and max_name in g)

    R_values = values("R_values", [0.01, 0.99], log=False)
    u_val = float(g.get("u", 20.0))

    theta_sources = [
        ("Tb_values", has_values_axis("Tb_values")),
        ("theta_f_values", has_values_axis("theta_f_values")),
        ("theta_values", has_values_axis("theta_values")),
    ]
    explicit_theta_sources = [name for name, present in theta_sources if present]
    if len(explicit_theta_sources) > 1:
        raise ValueError(
            "Specify only one founder/background axis among Tb_values, theta_f_values, and theta_values; "
            f"got {explicit_theta_sources}"
        )

    Tb_values: Optional[List[float]] = None
    if has_values_axis("Tb_values"):
        Tb_values = values("Tb_values", [0.0], log=False)
        theta_values = [float(u_val) * float(tb) for tb in Tb_values]
    elif has_values_axis("theta_f_values"):
        theta_values = values("theta_f_values", [0.0, 100, 200, 300, 400, 500, 600, 1000, 2000], log=False)
        Tb_values = [float(theta) / float(u_val) if float(u_val) != 0.0 else math.inf for theta in theta_values]
    else:
        theta_values = values("theta_values", [0.0, 100, 200, 300, 400, 500, 600, 1000, 2000], log=False)
        Tb_values = [float(theta) / float(u_val) if float(u_val) != 0.0 else math.inf for theta in theta_values]

    age_constraint_mode = str(g.get("age_constraint_mode", "upper_bound")).strip().lower()
    if age_constraint_mode not in {"upper_bound", "exact", "none"}:
        raise ValueError(f"unknown age_constraint_mode {age_constraint_mode!r}")
    age_exact = None if g.get("age_exact", None) is None else float(g["age_exact"])

    paired_T_theta_values: Optional[List[Tuple[float, float]]] = None
    if age_constraint_mode == "exact":
        if age_exact is None:
            raise ValueError("age_constraint_mode='exact' requires age_exact")
        if not explicit_theta_sources:
            raise ValueError("age_constraint_mode='exact' requires Tb_values, theta_f_values, or theta_values")
        if has_values_axis("T_values"):
            raise ValueError("age_constraint_mode='exact' derives paired T values from age_exact and Tb/theta; do not also specify T_values")
        paired_T_theta_values = [(float(age_exact) - float(tb), float(theta)) for tb, theta in zip(Tb_values or [], theta_values)]
        T_values = [T for T, _theta in paired_T_theta_values]
    else:
        T_values = values("T_values", [1.0, 100.0], log=False)

    N_values = values("N_values", [1e4, 1e8], log=True)

    if "alphas" in g:
        alphas = [float(x) for x in g["alphas"]]
    else:
        alphas = default_alphas(int(g.get("n_alpha", 20)), float(g.get("alpha_min", 0.05)), float(g.get("alpha_max", 1.0)))

    depth_values = [int(x) for x in g.get("depth_values", [int(g.get("depth", 120))])]
    return CacheGrid(
        R_values=R_values,
        T_values=T_values,
        theta_values=theta_values,
        N_values=N_values,
        alphas=alphas,
        depth_values=depth_values,
        u=u_val,
        ploidy_factor=float(g.get("ploidy_factor", 2.0)),
        lam=float(g.get("lam", 1.0)),
        condition_on_survival=bool(g.get("condition_on_survival", True)),
        max_age=float(g.get("max_age", 100.0)),
        enforce_age_constraint=bool(g.get("enforce_age_constraint", age_constraint_mode != "none")),
        age_constraint_mode=age_constraint_mode,
        age_exact=age_exact,
        paired_T_theta_values=paired_T_theta_values,
    )


def write_default_config(path: str | Path, *, smoke: bool = True) -> None:
    import yaml
    if smoke:
        cfg = {
            "grid": {
                "R_values": [0.9],
                "T_values": [1],
                "theta_values": [0],
                "N_values": [1e4],
                "depth_values": [100],
                "u": 20.0,
                "ploidy_factor": 2.0,
                "max_age": 100.0,
                "enforce_age_constraint": True,
                "n_alpha": 1,
                "alpha_min": 0.05,
                "alpha_max": 0.05,
            },
            "build": {
                "Kmax": 64,
                "base_node_factor": 2.0,
                "refine_node_factor": 4.0,
                "base_alias_eta": 24.0,
                "refine_alias_eta": 30.0,
                "n_bins": 16,
                "steps_per_time": 1.0,
                "max_steps": 60,
                "storage": "int16",
                "target_max_abs_z_error": 0.05,
                "target_max_abs_cdf_error": 1e-4,
                "pgf_backend": "batched",
                "batch_size": 64,
                "compressed_npz": True,
                "use_conjugate_symmetry": True,
                "stable_pgf_fallback": True,
                "stable_rk4_fallback_step_multiplier": 10.0,
                "stable_rk4_fallback_max_steps": 2000,
                "solve_ivp_fallback_node_cap": 4096,
                "resume": True,
            },
            "refinement": {
                "enabled": True,
                "max_levels": 2,
                "node_factor_growth": 1.5,
                "alias_eta_growth": 4.0,
                "n_bins_increment": 2,
                "steps_growth": 1.15,
                "max_steps_growth": 1.25,
                "max_seconds_per_bundle": 300,
                "fail_on_uncertified": False,
            },
            "parallel": {"n_jobs": 1},
        }
    else:
        cfg = {
            "grid": {
                "R_min": 0.01, "R_max": 0.99, "R_count": 30,
                "T_min": 1, "T_max": 100, "T_count": 50,
                "theta_values": [0, 100, 200, 300, 400, 500, 600, 1000, 2000],
                "N_min": 1e4, "N_max": 1e8, "N_count": 8,
                "depth_values": [120],
                "u": 20.0,
                "ploidy_factor": 2.0,
                "max_age": 100.0,
                "enforce_age_constraint": True,
                "n_alpha": 20,
                "alpha_min": 0.05,
                "alpha_max": 1.0,
            },
            "build": {
                "Kmax": 50_000,
                "base_node_factor": 2.0,
                "refine_node_factor": 4.0,
                "base_alias_eta": 30.0,
                "refine_alias_eta": 36.0,
                "n_bins": 24,
                "steps_per_time": 1.4,
                "max_steps": 120,
                "storage": "int16",
                "target_max_abs_z_error": 0.01,
                "target_max_abs_cdf_error": 1e-5,
                "pgf_backend": "batched",
                "batch_size": 128,
                "compressed_npz": True,
                "use_conjugate_symmetry": True,
                "stable_pgf_fallback": True,
                "stable_rk4_fallback_step_multiplier": 10.0,
                "stable_rk4_fallback_max_steps": 2000,
                "solve_ivp_fallback_node_cap": 4096,
                "resume": True,
            },
            "refinement": {
                "enabled": True,
                "max_levels": 5,
                "node_factor_growth": 1.6,
                "alias_eta_growth": 4.0,
                "n_bins_increment": 4,
                "steps_growth": 1.25,
                "max_steps_growth": 1.35,
                "max_seconds_per_bundle": 1800,
                "fail_on_uncertified": False,
            },
            "parallel": {"n_jobs": 20},
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
