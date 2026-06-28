"""Analytical fallbacks for essential self-adjointness verification."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from openfermion import BosonOperator
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from quantum_hamiltonian_analyzer.filters_base import FilterResult, SelfAdjointFilter
from quantum_hamiltonian_analyzer.utils import (
    ACTION_ANNIHILATE,
    ACTION_CREATE,
    get_modes,
    max_total_degree,
)

Term = Tuple[Tuple[int, int], ...]


def _term_to_phase_space(
    term: Term,
    x: Sequence[float],
    p: Sequence[float],
    modes: Sequence[int],
) -> complex:
    """Evaluate a single monomial under ``a -> (x + i p)/sqrt(2)`` substitution."""
    value = 1.0 + 0.0j
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    mode_index = {m: i for i, m in enumerate(modes)}
    for mode, action in term:
        idx = mode_index[mode]
        if action == ACTION_CREATE:
            value *= (x[idx] - 1j * p[idx]) * inv_sqrt2
        else:
            value *= (x[idx] + 1j * p[idx]) * inv_sqrt2
    return value


def _hamiltonian_phase_space(
    H: BosonOperator,
    x: Sequence[float],
    p: Sequence[float],
    modes: Sequence[int],
    homogeneous_degree: Optional[int] = None,
) -> complex:
    """Evaluate the Weyl/symbol of ``H`` in phase space."""
    total = complex(H.constant) if homogeneous_degree in (None, 0) else 0.0
    for term, coeff in H.terms.items():
        deg = len(term)
        if homogeneous_degree is not None and deg != homogeneous_degree:
            continue
        total += coeff * _term_to_phase_space(term, x, p, modes)
    return total


class SemiclassicalPhaseSpace(SelfAdjointFilter):
    """Semiclassical symbol analysis via highest-degree homogeneous part."""

    def __init__(
        self,
        n_starts: int = 32,
        tol: float = 1e-8,
    ) -> None:
        super().__init__()
        self.n_starts = n_starts
        self.tol = tol

    def evaluate(self, H: BosonOperator) -> FilterResult:
        """Return ``False`` if the leading symbol has a negative direction on ``S^{2M-1}``."""
        D = max_total_degree(H)
        if D == 0:
            return None

        modes = get_modes(H)
        if not modes:
            return None

        dim = 2 * len(modes)

        def objective(vec: np.ndarray) -> float:
            x = vec[: len(modes)]
            p = vec[len(modes) :]
            symbol = _hamiltonian_phase_space(H, x, p, modes, homogeneous_degree=D)
            return float(np.real(symbol))

        best = math.inf
        rng = np.random.default_rng(0)
        for _ in range(self.n_starts):
            vec = rng.normal(size=dim)
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                continue
            vec = vec / norm
            res = minimize(
                objective,
                vec,
                method="SLSQP",
                constraints={
                    "type": "eq",
                    "fun": lambda v: float(np.dot(v, v) - 1.0),
                },
                options={"ftol": self.tol, "maxiter": 200},
            )
            if res.fun < best:
                best = res.fun

        if best < -self.tol:
            return False
        if best > self.tol:
            return True
        return None


class RayMonodromy(SelfAdjointFilter):
    """Monte Carlo ray-shooting with radial ODE integration."""

    def __init__(
        self,
        n_rays: int = 16,
        r_max: float = 4.0,
        growth_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_rays = n_rays
        self.r_max = r_max
        self.growth_threshold = growth_threshold

    def evaluate(self, H: BosonOperator) -> FilterResult:
        """Return ``False`` if any ray exhibits super-Gaussian amplitude growth."""
        modes = get_modes(H)
        if not modes:
            return None

        D = max_total_degree(H)
        rng = np.random.default_rng(1)

        def amplitude_along_ray(z: np.ndarray, r: float) -> float:
            x = (r * np.real(z)).tolist()
            p = (r * np.imag(z)).tolist()
            val = _hamiltonian_phase_space(H, x, p, modes)
            return abs(val)

        for _ in range(self.n_rays):
            z = rng.normal(size=len(modes)) + 1j * rng.normal(size=len(modes))
            norm = np.linalg.norm(z)
            if norm < 1e-12:
                continue
            z = z / norm

            def rhs(_r: float, y: np.ndarray) -> List[float]:
                r_val = float(_r)
                eps = 1e-6
                amp = amplitude_along_ray(z, r_val)
                amp_p = amplitude_along_ray(z, r_val + eps)
                growth = (amp_p - amp) / eps
                return [growth]

            sol = solve_ivp(
                rhs,
                (0.0, self.r_max),
                [amplitude_along_ray(z, 0.0)],
                max_step=0.1,
                rtol=1e-6,
                atol=1e-8,
            )
            if not sol.success:
                continue

            r_vals = sol.t
            amps = sol.y[0]
            gauss_bound = np.exp(0.5 * r_vals**2)
            ratio = amps / np.maximum(gauss_bound, 1e-30)
            if np.any(ratio > 1.0 + self.growth_threshold):
                return False

        if D <= 2:
            return True
        return None


class NewtonPolygon(SelfAdjointFilter):
    """Asymptotic analysis via Newton polygon and convex hulls (stub)."""

    def evaluate(self, H: BosonOperator) -> FilterResult:
        """Planned asymptotic analysis of the differential operator symbol.

        Args:
            H: Bosonic Hamiltonian.

        Returns:
            Filter result once implemented.

        Raises:
            NotImplementedError: Always, until future development.
        """
        raise NotImplementedError("Planned for future development")


class PicardLefschetzThimbles(SelfAdjointFilter):
    """Multimode saddle search in complex space (stub)."""

    def evaluate(self, H: BosonOperator) -> FilterResult:
        """Planned Picard-Lefschetz thimble gradient descent.

        Args:
            H: Bosonic Hamiltonian.

        Returns:
            Filter result once implemented.

        Raises:
            NotImplementedError: Always, until future development.
        """
        raise NotImplementedError("Planned for future development")


class MultidimensionalWKB(SelfAdjointFilter):
    """Multidimensional eikonal / WKB analysis (stub)."""

    def evaluate(self, H: BosonOperator) -> FilterResult:
        """Planned multidimensional WKB eikonal solver.

        Args:
            H: Bosonic Hamiltonian.

        Returns:
            Filter result once implemented.

        Raises:
            NotImplementedError: Always, until future development.
        """
        raise NotImplementedError("Planned for future development")
