"""Automated analysis of polynomial bosonic Hamiltonians."""

from __future__ import annotations

from quantum_hamiltonian_analyzer.analytics import (
    MultidimensionalWKB,
    NewtonPolygon,
    PicardLefschetzThimbles,
    RayMonodromy,
    SemiclassicalPhaseSpace,
)
from quantum_hamiltonian_analyzer.filters_base import FilterResult, SelfAdjointFilter
from quantum_hamiltonian_analyzer.self_adjointness import (
    AnalyticalFallback,
    FilterCarleman,
    FilterDegree,
    FilterPathologicalOdd,
    FilterSemiBounded,
    analyze_self_adjointness,
    build_default_pipeline,
)
from quantum_hamiltonian_analyzer.symmetry import (
    check_symmetry_auto,
    check_symmetry_hybrid,
    check_symmetry_naive_openfermion,
    check_symmetry_symbolic,
)

__all__ = [
    "AnalyticalFallback",
    "FilterCarleman",
    "FilterDegree",
    "FilterPathologicalOdd",
    "FilterResult",
    "FilterSemiBounded",
    "MultidimensionalWKB",
    "NewtonPolygon",
    "PicardLefschetzThimbles",
    "RayMonodromy",
    "SelfAdjointFilter",
    "SemiclassicalPhaseSpace",
    "analyze_self_adjointness",
    "build_default_pipeline",
    "check_symmetry_auto",
    "check_symmetry_hybrid",
    "check_symmetry_naive_openfermion",
    "check_symmetry_symbolic",
]

__version__ = "0.1.0"
