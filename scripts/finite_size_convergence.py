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
    upgraded = common.upgrade_existing_start_configs(common.WORKDIR)
    if upgraded:
        print(f"Upgraded {upgraded} existing start configs for small-box neighbour lists")
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


def _relative_change_summary(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    column: str,
) -> tuple[float, float, int]:
    """Return median, 90th percentile, and count for matched relative changes."""
    if previous.empty or current.empty:
        return float("nan"), float("nan"), 0
    both = previous[[column]].join(current[[column]], lsuffix="_old", rsuffix="_new", how="inner").dropna()
    both = both[both[column + "_old"] != 0]
    if both.empty:
        return float("nan"), float("nan"), 0
    change = (both[column + "_new"] - both[column + "_old"]).abs() / both[column + "_old"].abs()
    return float(change.median()), float(change.quantile(0.9)), len(both)


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
        "--min-matched-points",
        type=int,
        default=10,
        help="Minimum shared valid state points required to declare convergence (default 10).",
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
    if args.min_matched_points < 1:
        parser.error("--min-matched-points must be at least 1")

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
    previous_table: pd.DataFrame | None = None
    chosen_n: int | None = None

    for N in particle_numbers:
        workdir = Path(f"RestitutionSweepWD_N_{N}")
        print(f"\nRunning sweep for N = {N} (workdir = {workdir}) …")
        # Pass the quick flag explicitly; the function signature expects
        # ``workdir`` as the second positional argument.
        df = _run_sweep(N, workdir, processes=None, quick=args.quick)
        table = _diffusion_table(df)
        valid_a = int(table["D_A"].notna().sum()) if not table.empty else 0
        valid_b = int(table["D_B"].notna().sum()) if not table.empty else 0
        print(f"  Valid state points: D_A {valid_a}, D_B {valid_b}")

        if previous_table is not None:
            median_a, p90_a, count_a = _relative_change_summary(previous_table, table, "D_A")
            median_b, p90_b, count_b = _relative_change_summary(previous_table, table, "D_B")
            print(
                f"  Matched state points: D_A {count_a}, D_B {count_b}"
            )
            print(f"  Median relative change: D_A {median_a:.3%}, D_B {median_b:.3%}")
            print(f"  90th percentile change: D_A {p90_a:.3%}, D_B {p90_b:.3%}")
            enough_matches = count_a >= args.min_matched_points and count_b >= args.min_matched_points
            if enough_matches and median_a < args.tolerance and median_b < args.tolerance:
                chosen_n = N
                print(
                    f"Converged at N = {N} (median tolerance {args.tolerance:.1%}; "
                    f"minimum matched points {args.min_matched_points})"
                )
                break
            if not enough_matches:
                print(
                    "  Not enough matched points to assess convergence "
                    f"(requires {args.min_matched_points})."
                )
        previous_table = table

    if chosen_n is None:
        print(
            "No convergence reached with the supplied particle numbers. "
            "Consider adding larger N values or relaxing the tolerance."
        )


if __name__ == "__main__":
    main()
