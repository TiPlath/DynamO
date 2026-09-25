#!/usr/bin/env python3
"""
Reproduce (an approximation of) Figure 1 of:

    Th. Voigtmann, "Multiple Glasses in Asymmetric Binary Hard Spheres",
    EPL 96, 36006 (2011)

using event-driven molecular dynamics of binary hard-sphere mixtures in DynamO,
while additionally sweeping the coefficient of restitution `e` of the hard-sphere
collisions.

Voigtmann's Fig. 1 plots the mode-coupling-theory glass-transition packing
fraction phi_c(delta, xhat) against the small-particle volume concentration
xhat = phi_small / phi_total, for several size ratios delta = d_small/d_large,
for *elastic*, equilibrium hard spheres. Here we instead run genuinely dynamic
(and optionally *inelastic*) hard-sphere MD on a grid of (N, delta, xhat, phi, e)
state points and measure the long-time diffusion coefficients of both species.
See analyze_restitution_sweep.py for how these are turned into (an attempt at)
phi_c(xhat) curves, plus a more robust fallback that just classifies each point
as fluid/arrested.

Since inelastic collisions dissipate energy (leading to inelastic collapse
without forcing), every run is held at a fixed granular temperature using the
Gaussian thermostat implemented in dynamo/systems/gaussianThermostat.cpp (SysGaussian):
a deterministic, system-wide velocity rescale fired at Poisson-distributed intervals
(mean free time "MFT"). This lets different restitution coefficients be compared on
an equal footing, at the cost of studying a driven steady state rather than a freely
cooling granular gas. dynamod has no CLI option for this thermostat (or for setting
Elasticity), so restitution_common.setup_worker() patches both into the generated XML.

Caveats:
    - The elastic e=1.0 points are the closest MD analogue of the paper, but are still
      finite-N, finite-time MD, not the idealised (N -> infinity, t -> infinity) MCT
      transition -- only qualitative trends are expected to match (the shape of phi_c(xhat),
      and its shift with decreasing e).
    - Locating phi_c precisely requires extrapolating D(phi) -> 0, which is noisy for
      finite MD runs. analyze_restitution_sweep.py both attempts this fit and, in case
      it fails or looks unreliable, produces a much simpler and more robust "state diagram"
      that just classifies each simulated (xhat, phi) point as liquid / single-glass /
      double-glass.
    - The simple FCC-lattice packer used here (dynamod pack-mode 8) cannot build valid
      configurations for very asymmetric (delta, xhat, phi) combinations -- see
      MAX_LATTICE_DENSITY in restitution_common.py. Those points are skipped automatically
      (and noted in each state's run.log).

Usage:
    Make sure "dynamod" and "dynarun" are on your PATH (e.g. activate the project's .venv,
    or add the build directory to PATH), then edit the QUICK_TEST flag and the state-variable
    ranges in restitution_common.py to suit your available compute budget, and run:

        python3 run_restitution_sweep.py

    This is safe to re-run: it resumes/extends previous runs rather than restarting them.
    Afterwards, run analyze_restitution_sweep.py to fit/plot the results.
"""
import pydynamo

import restitution_common as common

def main():

    mgr = pydynamo.SimManager(
        common.WORKDIR,
        common.STATEVARS,
        common.OUTPUTS,
        restarts=common.RESTARTS,
        processes=None,  # None = use all available processes
    )

    mgr.run(
        setup_worker=common.setup_worker,
        particle_equil_events=common.PARTICLE_EQUIL_EVENTS,
        particle_run_events=common.PARTICLE_RUN_EVENTS,
        particle_run_events_block_size=common.PARTICLE_RUN_EVENTS_BLOCK_SIZE,
    )   

if __name__ == "__main__":
    main()