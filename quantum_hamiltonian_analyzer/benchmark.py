"""Automated benchmarking for quantum_hamiltonian_analyzer."""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openfermion import BosonOperator, normal_ordered
from rich.console import Console
from rich.table import Table

from quantum_hamiltonian_analyzer.symmetry import (
    check_symmetry_auto,
    check_symmetry_hybrid,
    check_symmetry_symbolic,
)
from quantum_hamiltonian_analyzer.utils import (
    boson_to_qutip,
    difference_operator,
    hermitian_conjugate,
    is_zero_operator,
    random_boson_polynomial,
)

console = Console()


@dataclass
class TimingResult:
    """Single benchmark timing record."""

    method: str
    modes: int
    degree: int
    hermitian: bool
    seconds: float
    result: Optional[bool]


def baseline_qutip_isherm(H: BosonOperator, max_degree: int) -> bool:
    """QuTiP ``Qobj.isherm`` baseline with Fock truncation ``D + 1``.

    Args:
        H: Bosonic Hamiltonian.
        max_degree: Maximum polynomial degree ``D``.

    Returns:
        QuTiP Hermiticity verdict.
    """
    n_cutoff = max_degree + 1
    qobj = boson_to_qutip(H, n_cutoff=n_cutoff)
    return bool(qobj.isherm)


def baseline_naive_normal_order(H: BosonOperator) -> bool:
    """Naive OpenFermion baseline: ``normal_ordered(H - H^dagger) == 0``."""
    diff = normal_ordered(difference_operator(H))
    return is_zero_operator(diff)


def _time_call(fn: Callable[[], bool]) -> Tuple[bool, float]:
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed


def run_accuracy_benchmark(
    samples: int = 20,
    modes_list: Optional[List[int]] = None,
    degrees: Optional[List[int]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare symmetry methods against baselines on random polynomials.

    Args:
        samples: Number of random Hamiltonians per configuration.
        modes_list: Mode counts to test.
        degrees: Degrees to test.
        seed: RNG seed.

    Returns:
        DataFrame with accuracy metrics.
    """
    modes_list = modes_list or [1, 2]
    degrees = degrees or [1, 2, 3]
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []

    for m in modes_list:
        for d in degrees:
            for hermitian in (True, False):
                for i in range(samples):
                    sample_seed = int(rng.integers(0, 2**31))
                    H = random_boson_polynomial(
                        num_modes=m,
                        max_degree=d,
                        density=0.25,
                        hermitian=hermitian,
                        seed=sample_seed,
                    )
                    truth, _ = _time_call(lambda: baseline_qutip_isherm(H, d))
                    auto, t_auto = _time_call(lambda: check_symmetry_auto(H))
                    hybrid, t_hybrid = _time_call(lambda: check_symmetry_hybrid(H))
                    symbolic, t_sym = _time_call(lambda: check_symmetry_symbolic(H))
                    naive, t_naive = _time_call(lambda: baseline_naive_normal_order(H))

                    rows.append(
                        {
                            "modes": m,
                            "degree": d,
                            "hermitian": hermitian,
                            "sample": i,
                            "truth_qutip": truth,
                            "auto": auto,
                            "hybrid": hybrid,
                            "symbolic": symbolic,
                            "naive": naive,
                            "auto_match": auto == truth,
                            "hybrid_match": hybrid == truth,
                            "symbolic_match": symbolic == truth,
                            "naive_match": naive == truth,
                            "t_auto": t_auto,
                            "t_hybrid": t_hybrid,
                            "t_symbolic": t_sym,
                            "t_naive": t_naive,
                        }
                    )
    return pd.DataFrame(rows)


def run_timing_vs_degree(
    modes: int = 1,
    degrees: Optional[List[int]] = None,
    repeats: int = 3,
) -> pd.DataFrame:
    """Benchmark execution time versus polynomial degree."""
    degrees = degrees or [1, 2, 3, 4, 5]
    rows: List[Dict[str, object]] = []

    for d in degrees:
        H = random_boson_polynomial(
            num_modes=modes,
            max_degree=d,
            density=0.2,
            hermitian=True,
            seed=100 + modes * 10 + d,
        )
        for method_name, fn in (
            ("hybrid", lambda: check_symmetry_hybrid(H)),
            ("symbolic", lambda: check_symmetry_symbolic(H)),
            ("naive", lambda: baseline_naive_normal_order(H)),
        ):
            times: List[float] = []
            for _ in range(repeats):
                _, elapsed = _time_call(fn)
                times.append(elapsed)
            rows.append(
                {
                    "modes": modes,
                    "degree": d,
                    "method": method_name,
                    "seconds": float(np.median(times)),
                }
            )
    return pd.DataFrame(rows)


def run_timing_vs_modes(
    degree: int = 3,
    modes_list: Optional[List[int]] = None,
    repeats: int = 3,
) -> pd.DataFrame:
    """Benchmark execution time versus number of modes."""
    modes_list = modes_list or [1, 2, 3]
    rows: List[Dict[str, object]] = []

    for m in modes_list:
        H = random_boson_polynomial(
            num_modes=m,
            max_degree=degree,
            density=0.2,
            hermitian=True,
            seed=200 + m * 10 + degree,
        )
        for method_name, fn in (
            ("auto", lambda: check_symmetry_auto(H)),
            ("hybrid", lambda: check_symmetry_hybrid(H)),
            ("symbolic", lambda: check_symmetry_symbolic(H)),
        ):
            times: List[float] = []
            for _ in range(repeats):
                _, elapsed = _time_call(fn)
                times.append(elapsed)
            rows.append(
                {
                    "modes": m,
                    "degree": degree,
                    "method": method_name,
                    "seconds": float(np.median(times)),
                }
            )
    return pd.DataFrame(rows)


def print_accuracy_summary(df: pd.DataFrame) -> None:
    """Render accuracy summary table with Rich."""
    summary = (
        df.groupby(["modes", "degree"])
        .agg(
            auto_acc=("auto_match", "mean"),
            hybrid_acc=("hybrid_match", "mean"),
            symbolic_acc=("symbolic_match", "mean"),
            naive_acc=("naive_match", "mean"),
        )
        .reset_index()
    )
    table = Table(title="Symmetry Accuracy vs QuTiP Baseline")
    for col in summary.columns:
        table.add_column(str(col))
    for _, row in summary.iterrows():
        table.add_row(
            *[f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]) for c in summary.columns]
        )
    console.print(table)


def plot_timing_vs_degree(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot execution time versus degree for each mode count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for modes, group in df.groupby("modes"):
        fig, ax = plt.subplots(figsize=(7, 4))
        for method, sub in group.groupby("method"):
            sub = sub.sort_values("degree")
            ax.plot(sub["degree"], sub["seconds"], marker="o", label=method)
        ax.set_xlabel("Polynomial degree D")
        ax.set_ylabel("Time (s)")
        ax.set_yscale("log")
        ax.set_title(f"Execution time vs degree (M={modes})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = output_dir / f"timing_vs_degree_M{modes}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        console.print(f"Saved {path}")


def plot_timing_vs_modes(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot execution time versus mode count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for degree, group in df.groupby("degree"):
        fig, ax = plt.subplots(figsize=(7, 4))
        for method, sub in group.groupby("method"):
            sub = sub.sort_values("modes")
            ax.plot(sub["modes"], sub["seconds"], marker="o", label=method)
        ax.set_xlabel("Number of modes M")
        ax.set_ylabel("Time (s)")
        ax.set_yscale("log")
        ax.set_title(f"Execution time vs modes (D={degree})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = output_dir / f"timing_vs_modes_D{degree}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        console.print(f"Saved {path}")


def main() -> None:
    """CLI entry point for the benchmark suite."""
    parser = argparse.ArgumentParser(description="Benchmark quantum_hamiltonian_analyzer")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced benchmark configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_output"),
        help="Directory for plots and CSV exports.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib plot generation.",
    )
    args = parser.parse_args()

    if args.quick:
        acc_df = run_accuracy_benchmark(samples=5, modes_list=[1, 2], degrees=[1, 2, 3])
        deg_frames = [run_timing_vs_degree(m, [1, 2, 3, 4], repeats=2) for m in (1, 2, 3)]
        deg_df = pd.concat(deg_frames, ignore_index=True)
        mode_df = run_timing_vs_modes(3, [1, 2, 3], repeats=2)
    else:
        acc_df = run_accuracy_benchmark()
        deg_frames = [run_timing_vs_degree(m) for m in (1, 2, 3)]
        deg_df = pd.concat(deg_frames, ignore_index=True)
        mode_df = run_timing_vs_modes(3)
        for d in (2, 4):
            mode_df = pd.concat(
                [mode_df, run_timing_vs_modes(d)], ignore_index=True
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    acc_df.to_csv(args.output_dir / "accuracy.csv", index=False)
    deg_df.to_csv(args.output_dir / "timing_vs_degree.csv", index=False)
    mode_df.to_csv(args.output_dir / "timing_vs_modes.csv", index=False)

    print_accuracy_summary(acc_df)

    if not args.no_plots:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_timing_vs_degree(deg_df, args.output_dir)
            plot_timing_vs_modes(mode_df, args.output_dir)


if __name__ == "__main__":
    main()
