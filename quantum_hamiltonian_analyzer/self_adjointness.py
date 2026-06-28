"""Heuristic essential self-adjointness analysis via filter pipelines."""

from __future__ import annotations

from typing import List, Optional

from openfermion import BosonOperator

from quantum_hamiltonian_analyzer.analytics import RayMonodromy, SemiclassicalPhaseSpace
from quantum_hamiltonian_analyzer.filters_base import FilterResult, SelfAdjointFilter
from quantum_hamiltonian_analyzer.utils import (
    ACTION_CREATE,
    max_imbalance,
    max_total_degree,
)


class FilterDegree(SelfAdjointFilter):
    """Quadratic (and lower) polynomials are essentially self-adjoint."""

    def evaluate(self, H: BosonOperator) -> FilterResult:
        """Return ``True`` when the maximum total degree is at most two."""
        if max_total_degree(H) <= 2:
            return True
        return None


class FilterSemiBounded(SelfAdjointFilter):
    """Detect dominating positive diagonal highest-degree terms."""

    def evaluate(self, H: BosonOperator) -> FilterResult:
        """Return ``True`` when a positive ``(a^dagger a)^{D/2}`` term dominates."""
        D = max_total_degree(H)
        if D == 0 or D % 2 != 0:
            return None

        target_len = D // 2
        diagonal_coeff = 0.0
        max_other = 0.0

        for term, coeff in H.terms.items():
            if len(term) != D:
                continue
            creates = sum(1 for _, a in term if a == ACTION_CREATE)
            annihilates = len(term) - creates
            is_diagonal = creates == annihilates == target_len
            if is_diagonal:
                modes = [m for m, _ in term]
                if len(set(modes)) == 1 and creates == target_len:
                    diagonal_coeff += coeff.real
                    continue
            max_other = max(max_other, abs(coeff))

        if diagonal_coeff > 0 and diagonal_coeff > 2 * max_other:
            return True
        return None


class FilterCarleman(SelfAdjointFilter):
    """Carleman-type bound on coefficient shift growth."""

    def evaluate(self, H: BosonOperator) -> FilterResult:
        """Return ``True`` when imbalance-driven growth is ``O(n)`` or slower."""
        delta = max_imbalance(H)
        if delta == 0:
            return True

        D = max_total_degree(H)
        if D == 0:
            return True

        max_growth = 0.0
        for term, coeff in H.terms.items():
            if len(term) != D:
                continue
            imbalance = sum(
                1 if a == ACTION_CREATE else -1 for _, a in term
            )
            growth_rate = abs(imbalance) / max(D, 1)
            max_growth = max(max_growth, growth_rate)

        if max_growth <= 1.0 + 1e-12:
            return True
        return None


class FilterPathologicalOdd(SelfAdjointFilter):
    """Flag odd-degree pathologies without even stabilizers."""

    def evaluate(self, H: BosonOperator) -> FilterResult:
        """Return ``False`` for odd ``D`` with pure imaginary odd monomials."""
        D = max_total_degree(H)
        if D == 0 or D % 2 == 0:
            return None

        has_pathology = False
        even_stabilizer = 0.0

        for term, coeff in H.terms.items():
            deg = len(term)
            if deg == D and deg % 2 == 1:
                creates = sum(1 for _, a in term if a == ACTION_CREATE)
                annihilates = deg - creates
                if creates == 0 or annihilates == 0:
                    if abs(coeff.real) < 1e-12 and abs(coeff.imag) > 1e-12:
                        has_pathology = True
            if deg % 2 == 0 and deg >= D - 1:
                even_stabilizer = max(even_stabilizer, abs(coeff.real))

        if has_pathology and even_stabilizer < 1e-6:
            return False
        return None


class AnalyticalFallback(SelfAdjointFilter):
    """Run analytical methods when all heuristics are inconclusive."""

    def __init__(
        self,
        methods: Optional[List[SelfAdjointFilter]] = None,
    ) -> None:
        super().__init__()
        self._methods: List[SelfAdjointFilter] = methods or [
            SemiclassicalPhaseSpace(),
            RayMonodromy(),
        ]

    def evaluate(self, H: BosonOperator) -> FilterResult:
        """Sequentially invoke analytical fallbacks."""
        for method in self._methods:
            result = method.evaluate(H)
            if result is not None:
                return result
        return None


def build_default_pipeline() -> SelfAdjointFilter:
    """Construct the standard self-adjointness filter chain."""
    head = FilterDegree()
    (
        head.set_next(FilterSemiBounded())
        .set_next(FilterCarleman())
        .set_next(FilterPathologicalOdd())
        .set_next(AnalyticalFallback())
    )
    return head


def analyze_self_adjointness(H: BosonOperator) -> FilterResult:
    """Run the default pipeline to assess essential self-adjointness.

    Args:
        H: Bosonic Hamiltonian.

    Returns:
        ``True`` if essentially self-adjoint, ``False`` if defect indices are
        indicated, or ``None`` if all methods remain inconclusive.
    """
    return build_default_pipeline().handle(H)
