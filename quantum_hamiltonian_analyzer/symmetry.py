"""Symmetry (formal Hermiticity) analysis for bosonic Hamiltonians."""

from __future__ import annotations

import warnings

from openfermion import BosonOperator, normal_ordered

from quantum_hamiltonian_analyzer.utils import (
    apply_operator_matrix_free,
    difference_operator,
    is_zero_operator,
    iter_basis_states,
    mode_degrees,
    state_space_volume,
)


def check_symmetry_symbolic(H: BosonOperator, tol: float = 1e-10) -> bool:
    """Verify formal Hermiticity via symbolic normal ordering.

    Computes ``H - H^dagger`` and applies OpenFermion's ``normal_ordered``
    canonicalization. Returns ``True`` if the difference vanishes symbolically.

    Args:
        H: Bosonic Hamiltonian.
        tol: Coefficient tolerance for zero test.

    Returns:
        ``True`` if ``H`` is formally Hermitian, else ``False``.
    """
    diff = normal_ordered(difference_operator(H))
    return is_zero_operator(diff, tol=tol)


def check_symmetry_hybrid(H: BosonOperator, tol: float = 1e-10) -> bool:
    """Matrix-free truncated-basis check of ``H - H^dagger``.

    Does **not** use normal ordering. Extracts per-mode degrees ``D_i``,
    enumerates the Cartesian product of occupation numbers ``0..D_i``, and applies
    ``H - H^dagger`` to each basis vector using amplitude dictionaries only.

    Operators within each monomial are applied right-to-left with the correct
    bosonic ``sqrt(n)`` factors.

    Args:
        H: Bosonic Hamiltonian.
        tol: Amplitude tolerance for zero test.

    Returns:
        ``True`` if ``(H - H^dagger)|psi> = 0`` for all truncated basis states.
    """
    diff = difference_operator(H)
    if is_zero_operator(diff, tol=tol):
        return True

    degrees = mode_degrees(H)
    if not degrees:
        return is_zero_operator(diff, tol=tol)

    modes = sorted(degrees)
    max_occ = dict(degrees)

    for state in iter_basis_states(degrees):
        result = apply_operator_matrix_free(diff, state, modes, max_occ)
        if result:
            return False
    return True


def check_symmetry_auto(
    H: BosonOperator,
    space_limit: int = 100_000,
    tol: float = 1e-10,
) -> bool:
    """Route between hybrid and symbolic Hermiticity checks.

    Analyzes ``H.terms`` and compares the truncated Fock volume
    ``prod(D_i + 1)`` against ``space_limit``. Uses the fast hybrid method when
    the volume is below the limit; otherwise falls back to symbolic normal
    ordering. Emits a warning when any monomial length exceeds six operators in
    the symbolic path.

    Args:
        H: Bosonic Hamiltonian.
        space_limit: Maximum truncated volume for the hybrid algorithm.
        tol: Numerical tolerance passed to the selected backend.

    Returns:
        ``True`` if the Hamiltonian is formally Hermitian under the chosen test.
    """
    volume = state_space_volume(H)
    if volume <= space_limit:
        return check_symmetry_hybrid(H, tol=tol)

    max_len = max((len(term) for term in H.terms), default=0)
    if max_len > 6:
        warnings.warn(
            f"Monomial length {max_len} exceeds 6; symbolic normal ordering "
            "may be expensive or unreliable.",
            stacklevel=2,
        )
    return check_symmetry_symbolic(H, tol=tol)


def check_symmetry_naive_openfermion(H: BosonOperator, tol: float = 1e-10) -> bool:
    """Baseline: ``normal_ordered(H - H^dagger) == 0`` (OpenFermion native).

    Args:
        H: Bosonic Hamiltonian.
        tol: Coefficient tolerance.

    Returns:
        ``True`` if formally Hermitian by naive normal ordering.
    """
    return check_symmetry_symbolic(H, tol=tol)
