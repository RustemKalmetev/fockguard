# quantum_hamiltonian_analyzer

Automated analysis of polynomial Hamiltonians built from bosonic creation and annihilation operators.

## Features

- **Symmetry analysis**: symbolic, matrix-free hybrid, and auto-routed Hermiticity checks.
- **Self-adjointness heuristics**: chain-of-responsibility filters with analytical fallbacks.
- **Benchmarking**: accuracy and performance comparisons against QuTiP and naive OpenFermion baselines.

## Installation

```bash
pip install -e .
```

## Quick start

```python
from openfermion import BosonOperator
from quantum_hamiltonian_analyzer import check_symmetry_auto, analyze_self_adjointness

H = BosonOperator("0^ 0")  # number operator (Hermitian)
print(check_symmetry_auto(H))          # True
print(analyze_self_adjointness(H))     # True
```

## Benchmark

```bash
qha-benchmark --quick
```
