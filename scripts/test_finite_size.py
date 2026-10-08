"""Simple test to verify that changing the particle number creates a different
set of simulation directories. This is a lightweight sanity‑check for finite‑
size effects handling.

The test runs the sweep with ``QUICK_TEST=True`` (a very small state space)
for two different values of ``N`` and checks that the work directory contains
distinct restart folders for each ``N``.

It does **not** validate physics – only that the code reacts to the ``N``
parameter as expected.
"""

import os
import shutil
from pathlib import Path

import pytest
import pandas as pd

import restitution_common as common
import pydynamo
from finite_size_convergence import _relative_change_summary


@pytest.fixture(scope="function")
def clean_workdir(tmp_path: Path):
    """Create a temporary work directory and clean up after the test."""
    orig_workdir = common.WORKDIR
    common.WORKDIR = str(tmp_path / "RestitutionSweepWD")
    # Ensure a fresh directory
    if os.path.isdir(common.WORKDIR):
        shutil.rmtree(common.WORKDIR)
    yield
    # Restore original value and remove temporary files
    common.WORKDIR = orig_workdir
    if os.path.isdir(str(tmp_path)):
        shutil.rmtree(str(tmp_path))


def _run_sweep(N_value, quick_test=True):
    """Run a minimal sweep for a given ``N``.

    Parameters
    ----------
    N_value: int
        Number of particles (must be 4*L^3 for the FCC lattice).
    quick_test: bool
        Use the QUICK_TEST configuration to keep the run fast.
    """
    # Override the global configuration for this run
    common.configure(quick_test)
    common.N_VALUES = [N_value]
    # Use a single restart to keep the number of directories predictable
    common.RESTARTS = 1
    common.update_statevars()

    mgr = pydynamo.SimManager(
        common.WORKDIR,
        common.STATEVARS,
        common.OUTPUTS,
        restarts=common.RESTARTS,
        processes=1,  # run serially for the test environment
    )
    mgr.run(
        setup_worker=common.setup_worker,
        particle_equil_events=common.PARTICLE_EQUIL_EVENTS,
        particle_run_events=common.PARTICLE_RUN_EVENTS,
        particle_run_events_block_size=common.PARTICLE_RUN_EVENTS_BLOCK_SIZE,
    )


def test_finite_size_effects(clean_workdir):
    """Check that two different ``N`` values generate distinct restart folders.
    """
    # First run with the default N (2048)
    _run_sweep(N_value=4 * 8**3)
    dirs_after_first = set(os.listdir(common.WORKDIR))

    # Second run with a larger system (2916 particles)
    _run_sweep(N_value=4 * 9**3)
    dirs_after_second = set(os.listdir(common.WORKDIR))

    # The two sets should differ – at least one directory name must contain the
    # different N value.
    assert dirs_after_first != dirs_after_second
    # Ensure that both N values appear somewhere in the directory names.
    assert any("N_2048" in d for d in dirs_after_first)
    assert any("N_2916" in d for d in dirs_after_second)


def test_relative_change_summary_uses_matched_nonzero_points():
    index = pd.Index(["shared-a", "shared-b", "zero-baseline", "previous-only"])
    previous = pd.DataFrame({"D_A": [10.0, 20.0, 0.0, 50.0]}, index=index)
    current = pd.DataFrame(
        {"D_A": [11.0, 10.0, 100.0, float("nan")]}, index=index
    )

    median, p90, count = _relative_change_summary(previous, current, "D_A")

    assert median == pytest.approx(0.3)
    assert p90 == pytest.approx(0.46)
    assert count == 2


def test_relative_change_summary_handles_no_valid_matches():
    previous = pd.DataFrame({"D_A": [0.0]}, index=["point"])
    current = pd.DataFrame({"D_A": [1.0]}, index=["point"])

    median, p90, count = _relative_change_summary(previous, current, "D_A")

    assert median != median
    assert p90 != p90
    assert count == 0
