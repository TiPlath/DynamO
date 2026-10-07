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


def _run_sweep(
    N: int,
    workdir: Path,
    processes: int | None = None,
    quick: bool = False,
) -> pd.DataFrame:
    """Run (or resume) the sweep for one particle number and return its results.

    Parameters
    ----------
    N:
        Particle number for the current sweep.
    workdir:
        Directory where the sweep data will be stored. It is created by the
        caller if it does not already exist.
    processes:
        Number of parallel processes to hand to :class:`pydynamo.SimManager`.
        ``None`` lets the manager decide (default behaviour of the original
        script).
    quick:
        When ``True`` the sweep runs in *quick* mode.  The original script
        toggles this via the global ``common.QUICK_TEST`` flag; we set that
        flag here so that the function can be reused from other scripts that
        also accept a ``quick`` option.

    The function updates the ``common`` configuration, creates a
    :class:`pydynamo.SimManager`, runs the simulation and finally returns the
    collected production‑run data.
    """
    # Apply quick‑test mode locally if requested.  ``common.configure`` sets
    # *all* sweep parameters (event counts, RESTARTS, etc.) appropriate for a
    # quick or full run.  Passing the ``quick`` flag directly keeps the logic
    # simple and ensures the global ``common.QUICK_TEST`` flag is updated.
    common.configure(quick)

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

def _collect_diffusion(workdir: Path) -> tuple[float, float]:
    """Fetch the diffusion data for *workdir* and return the mean ``D_A`` and ``D_B``.

    The function mirrors the logic used in the original script where the
    sweep results are written to ``common.WORKDIR``.  It creates a temporary
    :class:`pydynamo.SimManager` that points at the same configuration and
    extracts the nominal diffusion coefficients via :func:`_diffusion_table`.
    ``NaN`` values (e.g., state points without production data) are ignored
    when computing the mean.
    """
    # Ensure the common module points at the correct directory before creating
    # the manager.  ``common.WORKDIR`` is a global that the SimManager reads.
    common.WORKDIR = str(workdir)
    # The state‑variable list may have been altered by the caller; rebuild it
    # to guarantee consistency.
    common.update_statevars()

    mgr = pydynamo.SimManager(
        common.WORKDIR,
        common.STATEVARS,
        common.OUTPUTS,
        restarts=common.RESTARTS,
        processes=1,
    )
    df = mgr.fetch_data(common.PARTICLE_EQUIL_EVENTS, only_current_statevars=True)
    table = _diffusion_table(df)
    # ``mean`` skips NaN by default.
    mean_a = float(table["D_A"].mean()) if not table.empty else float("nan")
    mean_b = float(table["D_B"].mean()) if not table.empty else float("nan")
    return mean_a, mean_b

def _relative_change(old: float, new: float) -> float:
    """Return the relative change ``|new‑old| / |old|``.

    If ``old`` is zero the function returns ``float('nan')`` to avoid a
    division‑by‑zero error, matching the behaviour of the median‑relative
    helper above.
    """
    if old == 0:
        return float("nan")
    return abs(new - old) / abs(old)


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
        # Pass the quick flag explicitly; the function signature expects
        # ``workdir`` as the second positional argument.
        _run_sweep(N, workdir, processes=None, quick=args.quick)
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
