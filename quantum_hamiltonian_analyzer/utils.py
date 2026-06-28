"""Shared utilities for bosonic Hamiltonian analysis."""

from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from openfermion import BosonOperator
from openfermion.utils.operator_utils import hermitian_conjugated

Term = Tuple[Tuple[int, int], ...]
Occupation = Tuple[int, ...]
AmplitudeDict = Dict[Occupation, complex]

# OpenFermion convention: action 1 = creation (a^dagger), action 0 = annihilation (a).
ACTION_CREATE = 1
ACTION_ANNIHILATE = 0


def hermitian_conjugate(H: BosonOperator) -> BosonOperator:
    """Return the formal Hermitian conjugate of a bosonic operator.

    Args:
        H: Bosonic Hamiltonian or operator.

    Returns:
        The operator ``H^dagger`` using OpenFermion's conjugation rules.
    """
    return hermitian_conjugated(H)


def get_modes(H: BosonOperator) -> List[int]:
    """Collect sorted mode indices appearing in ``H.terms``.

    Args:
        H: Bosonic operator.

    Returns:
        Sorted list of distinct mode indices.
    """
    modes: set[int] = set()
    for term in H.terms:
        for mode, _ in term:
            modes.add(mode)
    return sorted(modes)


def mode_degrees(H: BosonOperator) -> Dict[int, int]:
    """Maximum operator count per mode across all monomials.

    For each mode ``i``, ``D_i`` is the maximum number of times mode ``i`` appears
    in any monomial of ``H`` (counting both creation and annihilation factors).

    Args:
        H: Bosonic operator.

    Returns:
        Mapping from mode index to maximum local degree ``D_i``.
    """
    degrees: Dict[int, int] = {}
    for term in H.terms:
        local: Dict[int, int] = {}
        for mode, _ in term:
            local[mode] = local.get(mode, 0) + 1
        for mode, count in local.items():
            degrees[mode] = max(degrees.get(mode, 0), count)
    return degrees


def total_degree(term: Term) -> int:
    """Return the total degree (operator count) of a monomial."""
    return len(term)


def max_total_degree(H: BosonOperator) -> int:
    """Return the maximum total degree among monomials of ``H``."""
    if not H.terms:
        return 0
    return max(len(term) for term in H.terms)


def creation_annihilation_imbalance(term: Term) -> int:
    """Return ``n_create - n_annihilate`` for a monomial."""
    creates = sum(1 for _, action in term if action == ACTION_CREATE)
    annihilates = len(term) - creates
    return creates - annihilates


def max_imbalance(H: BosonOperator) -> int:
    """Maximum absolute creation/annihilation imbalance over monomials."""
    if not H.terms:
        return 0
    return max(abs(creation_annihilation_imbalance(term)) for term in H.terms)


def state_space_volume(H: BosonOperator) -> int:
    """Compute truncated Fock-space volume ``prod(D_i + 1)``."""
    degrees = mode_degrees(H)
    if not degrees:
        return 1
    volume = 1
    for d in degrees.values():
        volume *= d + 1
    return volume


def iter_basis_states(degrees: Dict[int, int]) -> Iterator[Occupation]:
    """Iterate occupation-number basis states up to per-mode degrees.

    Args:
        degrees: Mapping mode index -> ``D_i``.

    Yields:
        Occupation tuples ``(n_0, n_1, ...)`` in lexicographic order.
    """
    if not degrees:
        yield ()
        return
    modes = sorted(degrees)
    ranges = [range(degrees[m] + 1) for m in modes]
    for occ_values in itertools.product(*ranges):
        yield tuple(occ_values)


def occupation_to_dict(
    state: Occupation, modes: Sequence[int]
) -> Dict[int, int]:
    """Map a compact occupation tuple to a mode-index dictionary."""
    return {mode: state[i] for i, mode in enumerate(modes)}


def apply_monomial_right_to_left(
    term: Term,
    coefficient: complex,
    state: Occupation,
    modes: Sequence[int],
    max_occupation: Dict[int, int],
) -> Optional[AmplitudeDict]:
    """Apply a single monomial to an occupation state without matrices.

    Operators are applied from right to left (reverse of the stored term order),
    multiplying by ``sqrt(n)`` factors for annihilation and ``sqrt(n+1)`` for
    creation.

    Args:
        term: Monomial key from ``H.terms``.
        coefficient: Monomial coefficient.
        state: Initial occupation numbers aligned with ``modes``.
        modes: Sorted mode indices.
        max_occupation: Per-mode truncation ``D_i``.
    Returns:
        A dictionary mapping output occupation tuples to amplitudes, or ``None``
        if the monomial annihilates the state (e.g. action on zero occupation).
    """
    occ = occupation_to_dict(state, modes)
    amp = complex(coefficient)

    for mode, action in reversed(term):
        n = occ.get(mode, 0)
        if action == ACTION_CREATE:
            amp *= math.sqrt(n + 1)
            occ[mode] = n + 1
        else:
            if n <= 0:
                return None
            amp *= math.sqrt(n)
            occ[mode] = n - 1

    out_state = tuple(occ[m] for m in modes)
    for m, n in occ.items():
        if n > max_occupation.get(m, 0):
            # Support outside truncation counts as non-zero leakage.
            return {out_state: amp}

    return {out_state: amp}


def apply_operator_matrix_free(
    H: BosonOperator,
    state: Occupation,
    modes: Sequence[int],
    max_occupation: Dict[int, int],
) -> AmplitudeDict:
    """Apply a bosonic operator to a single basis state using amplitudes only.

    Args:
        H: Bosonic operator.
        state: Initial occupation tuple.
        modes: Sorted mode indices.
        max_occupation: Truncation per mode.

    Returns:
        Dictionary of output states to amplitudes (empty if identically zero).
    """
    result: AmplitudeDict = {}
    # Reverse iteration order over terms (right-to-left monomial processing).
    for term in reversed(list(H.terms.keys())):
        coeff = H.terms[term]
        contribution = apply_monomial_right_to_left(
            term, coeff, state, modes, max_occupation
        )
        if contribution is None:
            continue
        for occ, amp in contribution.items():
            result[occ] = result.get(occ, 0.0) + amp
    # Remove numerically negligible entries.
    return {
        occ: amp
        for occ, amp in result.items()
        if abs(amp) > 1e-12
    }


def difference_operator(H: BosonOperator) -> BosonOperator:
    """Return ``H - H^dagger`` without normal ordering."""
    return H - hermitian_conjugate(H)


def is_zero_operator(H: BosonOperator, tol: float = 1e-10) -> bool:
    """Check whether all coefficients of an operator are negligible."""
    return all(abs(c) <= tol for c in H.terms.values())


def boson_to_qutip(H: BosonOperator, n_cutoff: int) -> "object":
    """Build a QuTiP ``Qobj`` matrix for ``H`` in a truncated Fock space.

    Args:
        H: Bosonic operator.
        n_cutoff: Single-mode occupation cutoff (dimension per mode).

    Returns:
        ``qutip.Qobj`` representing ``H``.
    """
    import qutip

    modes = get_modes(H)
    if not modes:
        return qutip.Qobj([[complex(H.constant)]], dims=[[1], [1]])

    destroy_ops = []
    for mode in modes:
        if len(modes) == 1:
            destroy_ops.append(qutip.destroy(n_cutoff))
        else:
            ops = [qutip.qeye(n_cutoff) for _ in modes]
            ops[modes.index(mode)] = qutip.destroy(n_cutoff)
            destroy_ops.append(qutip.tensor(*ops))

    result = qutip.Qobj(
        [[0.0]], dims=[[n_cutoff] * len(modes), [n_cutoff] * len(modes)]
    )
    if H.constant:
        id_op = qutip.qeye(n_cutoff)
        if len(modes) > 1:
            id_op = qutip.tensor([qutip.qeye(n_cutoff) for _ in modes])
        result += float(H.constant) * id_op

    for term, coeff in H.terms.items():
        op = qutip.Qobj(
            [[1.0]], dims=[[n_cutoff] * len(modes), [n_cutoff] * len(modes)]
        )
        if len(modes) > 1:
            op = qutip.tensor([qutip.qeye(n_cutoff) for _ in modes])
        for mode, action in term:
            base = destroy_ops[modes.index(mode)]
            op = op * (base.dag() if action == ACTION_CREATE else base)
        result += coeff * op
    return result


def random_boson_polynomial(
    num_modes: int,
    max_degree: int,
    density: float = 0.3,
    hermitian: bool = True,
    seed: Optional[int] = None,
) -> BosonOperator:
    """Generate a random bosonic polynomial for benchmarking.

    Args:
        num_modes: Number of modes ``M``.
        max_degree: Maximum total degree ``D``.
        density: Fraction of possible monomials (up to degree ``D``) to include.
        hermitian: If ``True``, symmetrize by ``H + H^dagger`` and take real part.
        seed: Optional RNG seed.

    Returns:
        Random ``BosonOperator``.
    """
    rng = np.random.default_rng(seed)
    H = BosonOperator()

    for degree in range(max_degree + 1):
        if degree == 0:
            if rng.random() < density:
                H += BosonOperator() * float(rng.normal())
            continue
        # Enumerate monomial patterns up to total degree.
        for _ in range(max(1, int(density * num_modes * (max_degree + 1)))):
            term_ops: List[Tuple[int, int]] = []
            for _ in range(degree):
                mode = int(rng.integers(0, num_modes))
                action = int(rng.integers(0, 2))
                term_ops.append((mode, action))
            coeff = complex(rng.normal(), rng.normal())
            if abs(coeff) < 1e-14:
                continue
            mono = BosonOperator()
            for mode, action in term_ops:
                label = f"{mode}^" if action == ACTION_CREATE else str(mode)
                mono *= BosonOperator(label)
            H += coeff * mono

    if hermitian:
        H = H + hermitian_conjugate(H)
        # Force real coefficients where possible.
        real_terms: Dict[Term, complex] = {}
        for term, coeff in H.terms.items():
            real_terms[term] = (real_terms.get(term, 0.0) + coeff) / 2.0
        H = BosonOperator()
        for term, coeff in real_terms.items():
            if abs(coeff.imag) > 1e-12:
                coeff = complex(coeff.real)
            if abs(coeff) > 1e-14:
                mono = BosonOperator()
                for mode, action in term:
                    label = f"{mode}^" if action == ACTION_CREATE else str(mode)
                    mono *= BosonOperator(label)
                H += coeff * mono
    return H
