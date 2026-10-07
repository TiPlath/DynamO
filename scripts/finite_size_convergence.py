"""Finite‑size convergence helper for the restitution sweep.

The goal is to run the same parameter sweep with several particle numbers
(`N`) and compare the resulting diffusion coefficients.  When the change
between successive ``N`` values falls below a user‑specified relative
tolerance we consider the results converged and report the smallest ``N``
that satisfies the criterion.

Usage example::

    python3 finite_size_convergence.py \
        -n 2048 2916 4000 5324 \
        -t 0.05 \
        -quick

The script will:

1. Select the quick or the full parameter ranges of ``restitution_common``
   and override ``N_VALUES`` for each run.  ``N`` must be ``4 * L**3``.
2. Execute (or resume) the sweep via :class:`pydynamo.SimManager`, one work
   directory per ``N``.
3. Collect the production-run results with ``SimManager.fetch_data`` (the
   equilibration blocks are excluded) and extract ``D_A`` and ``D_B`` for
   every state point.
4. Compare each state point to the previous ``N`` and stop when the median
   relative change of both diffusion coefficients is smaller than
   ``tolerance``.

Only a very lightweight statistical analysis is performed – the purpose is
to give a quick indication of when increasing ``N`` no longer improves the
observable of interest.  For a rigorous finite‑size study you would repeat
each ``N`` several times and analyse the full distribution, but that is
outside the scope of this helper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pydynamo
import restitution_common as common


def _check_fcc(N: int) -> None:
    L = round((N / 4) ** (1 / 3))
    if 4 * L**3 != N:
        raise SystemExit(f"N={N} does not fit an FCC lattice (needs 4*L^3, e.g. 2048, 2916, 4000, 5324)")


def _run_sweep(N: int, workdir: Path, processes: int | None) -> pd.DataFrame:
    """Run (or resume) the sweep for one particle number and return its results.

    An existing work directory is reused, so an interrupted run continues
    where it stopped.
    """
    common.WORKDIR = str(workdir)
    common.N_VALUES = [N]
    # A single restart keeps the number of directories predictable
    common.RESTARTS = 1
    common.update_statevars()

    mgr = pydynamo.SimManager(
        common.WORKDIR,
        common.STATEVARS,
        common.OUTPUTS,
        restarts=common.RESTARTS,
        processes=processes,
    )
    mgr.run(
        setup_worker=common.setup_worker,
        particle_equil_events=common.PARTICLE_EQUIL_EVENTS,
        particle_run_events=common.PARTICLE_RUN_EVENTS,
        particle_run_events_block_size=common.PARTICLE_RUN_EVENTS_BLOCK_SIZE,
    )
    return mgr.fetch_data(common.PARTICLE_EQUIL_EVENTS, only_current_statevars=True)


def _diffusion_table(df: pd.DataFrame) -> pd.DataFrame:
    """``D_A`` and ``D_B`` (nominal values) per state point."""
    if df.empty or "D_A" not in df.columns or "D_B" not in df.columns:
        return pd.DataFrame(columns=["D_A", "D_B"])

    def nominal(x):
        # State points without production data are plain NaN, not ufloats
        return x.nominal_value if hasattr(x, "nominal_value") else float("nan")

    table = df.set_index(["delta", "xhat", "phi", "e"])[["D_A", "D_B"]]
    return table.apply(lambda column: column.map(nominal))


def _median_relative_change(previous: pd.DataFrame, current: pd.DataFrame, column: str) -> tuple[float, int]:
    """Median of ``|new-old| / |old|`` over the state points present in both tables."""
    both = previous[[column]].join(current[[column]], lsuffix="_old", rsuffix="_new", how="inner").dropna()
    both = both[both[column + "_old"] != 0]
    if both.empty:
        return float("nan"), 0
    change = (both[column + "_new"] - both[column + "_old"]).abs() / both[column + "_old"].abs()
    return float(change.median()), len(both)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finite‑size convergence test for the restitution sweep."
    )
    parser.add_argument(
        "-n",
        "--particle-numbers",
        type=int,
        nargs="+",
        required=True,
        help="List of particle numbers to test (e.g. 2048 2916 4096).",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=0.05,
        help="Relative tolerance for convergence (default 5%%).",
    )
    parser.add_argument(
        "-quick",
        action="store_true",
        help="Run in QUICK_TEST mode to keep each sweep short.",
    )
    # New flags for elastic‑only and NVE runs
    parser.add_argument(
        "-elastic",
        action="store_true",
        help="Force elastic collisions only (e = 1.0).",
    )
    parser.add_argument(
        "-nve",
        action="store_true",
        help="Run in NVE ensemble (disable Gaussian thermostat).",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Apply global modifications requested by the user before any sweep.
    # ------------------------------------------------------------------
    if args.elastic:
        # Force elastic collisions only.
        common.E_VALUES = [1.0]
        # Re‑build STATEVARS to reflect the new ``e`` list.
        common.update_statevars()

    if args.nve:
        # Disable the Gaussian thermostat – the sweep will run in NVE.
        common.THERMOSTAT_MFT_VALUES = []
        # Re‑build STATEVARS to reflect the missing thermostat entry.
        common.update_statevars()

    particle_numbers = sorted(args.particle_numbers)
    previous_a: float | None = None
    previous_b: float | None = None
    chosen_n: int | None = None

    for N in particle_numbers:
        workdir = Path(f"RestitutionSweepWD_N_{N}")
        print(f"\nRunning sweep for N = {N} (workdir = {workdir}) …")
        _run_sweep(N, quick=args.quick, workdir=workdir)
        mean_a, mean_b = _collect_diffusion(workdir)
        print(f"  Mean D_A = {mean_a:.5g}, Mean D_B = {mean_b:.5g}")

        if previous_a is not None and previous_b is not None:
            rel_a = _relative_change(previous_a, mean_a)
            rel_b = _relative_change(previous_b, mean_b)
            print(
                f"  Relative change vs previous N: D_A {rel_a:.3%}, D_B {rel_b:.3%}"
            )
            if rel_a < args.tolerance and rel_b < args.tolerance:
                chosen_n = N
                print(
                    f"Converged at N = {N} (tolerance {args.tolerance:.1%})"
                )
                break
        previous_a, previous_b = mean_a, mean_b

    if chosen_n is None:
        print(
            "No convergence reached with the supplied particle numbers. "
            "Consider adding larger N values or relaxing the tolerance."
        )


if __name__ == "__main__":
    main()
