"""Shared code for the restitution-coefficient glass-transition sweep.

Imported by both run_restitution_sweep.py and analyze_restitution_sweep.py
so that the state-variable/output-property definitions and the config
generator are guaranteed to match between the two.

See run_restitution_sweep.py for the physical background and caveats.
"""
import math

import numpy
import pydynamo
from pydynamo.config_files import ConfigFile
from pydynamo.output_properties import OutputProperty, SingleAttrib
from pydynamo.weighted_types import WeightedType

WORKDIR = "RestitutionSweepWD"

# A single FCC lattice is used for both species (see dynamod pack-mode 8),
# so the "density" fed to dynamod (which sets the large-species diameter)
# must stay comfortably below the FCC touching limit of sqrt(2), or the
# generated configuration will have overlapping particles. Combinations of
# (delta, xhat, phi) that would require exceeding this are skipped
# automatically in setup_worker().
MAX_LATTICE_DENSITY = 1.3

# Set QUICK_TEST=True for a quick sanity check of the workflow.
# Set QUICK_TEST=False for a more complete scan of the phase diagram.
# The full scan is needed to see clear glass transitions matching the
# literature (Voigtmann, EPL 96, 36006, 2011).
QUICK_TEST = False

if QUICK_TEST:
    # Smaller system for quick testing - 2048 particles works reliably
    # with the compression engine across the tested parameter range.
    N_VALUES = [4 * 8**3]  # 2048 particles
    DELTA_VALUES = [0.2, 0.35, 0.5]  # Size ratios to test
    XHAT_VALUES = [0.2, 0.4, 0.6, 0.8]  # Small particle concentration
    PHI_VALUES = list(numpy.round(numpy.arange(0.45, 0.62, 0.02), 3))  # Packing fractions
    E_VALUES = [1.0, 0.9, 0.8]  # Restitution coefficients
    RESTARTS = 1
    PARTICLE_EQUIL_EVENTS = 200
    PARTICLE_RUN_EVENTS = 400
    PARTICLE_RUN_EVENTS_BLOCK_SIZE = 200
else:
    # More comprehensive parameter space coverage
    N_VALUES = [4 * 8**3]  # 2048 particles
    DELTA_VALUES = [0.2, 0.3, 0.35, 0.4, 0.5]
    XHAT_VALUES = list(numpy.round(numpy.arange(0.1, 0.95, 0.1), 3))
    PHI_VALUES = list(numpy.round(numpy.arange(0.45, 0.62, 0.01), 3))
    E_VALUES = [1.0, 0.95, 0.9, 0.8, 0.7]
    RESTARTS = 2
    PARTICLE_EQUIL_EVENTS = 1000
    PARTICLE_RUN_EVENTS = 2000
    PARTICLE_RUN_EVENTS_BLOCK_SIZE = 500

# Mean free time (in reduced time units) between global Gaussian-thermostat
# rescale events. Keep this shorter than the natural collisional mean free
# time at the target packing fractions or the granular temperature will sag
# below the requested value for strongly dissipative (low e) state points.
THERMOSTAT_MFT_VALUES = [0.01]

STATEVARS = [
    [
        ("N", N_VALUES),
        ("delta", DELTA_VALUES),
        ("xhat", XHAT_VALUES),
        ("phi", PHI_VALUES),
        ("e", E_VALUES),
        ("thermostat_mft", THERMOSTAT_MFT_VALUES),
    ]
]

OUTPUTS = ["D_A", "D_B", "T_system", "phi_actual", "T_A", "T_B"]


# ---------------------------------------------------------------------------
# Binary hard-sphere mixture geometry (matches dynamod's IPPacker case 8)
# ---------------------------------------------------------------------------
def binary_particle_counts(n_total, delta, xhat):
    """Split n_total FCC lattice sites into (n_large, n_small) particles so
    that the small-particle volume concentration is xhat, at size ratio
    delta = d_small / d_large."""
    f_small = xhat / (delta**3 * (1.0 - xhat) + xhat)
    n_small = int(round(f_small * n_total))
    n_small = min(max(n_small, 1), n_total - 1)
    return n_total - n_small, n_small


def binary_density(n_total, n_large, n_small, delta, phi):
    """The dynamod -d/--density value that yields packing fraction phi for
    the given particle split and size ratio."""
    return phi * n_total * 6.0 / (math.pi * (n_large + n_small * delta**3))


def setup_worker(config, state, logfile, particle_equil_events):
    import os
    from subprocess import CalledProcessError, check_call

    # pydynamo passes state as a tuple of (name, value) pairs, not a dict.
    state = dict(state)

    N = state["N"]
    ncells_unrounded = (N / 4.0) ** (1.0 / 3.0)
    ncells = int(round(ncells_unrounded))
    if abs(ncells - ncells_unrounded) > 1e-6:
        raise RuntimeError(f"N={N} is not compatible with an FCC packing (4*C^3)")

    delta = state["delta"]
    xhat = state["xhat"]
    phi = state["phi"]
    e = state["e"]
    mft = state["thermostat_mft"]

    n_large, n_small = binary_particle_counts(N, delta, xhat)
    mass_ratio = delta**3  # equal mass density for both species
    workdir = os.path.dirname(config) or "."

    pack_args = [
        "dynamod",
        "-m",
        "8",
        "--i1",
        "0",
        "--f1",
        repr(delta),
        "--f2",
        repr(mass_ratio),
        "--i2",
        str(n_large),
        "-C",
        str(ncells),
    ]

    # A single FCC lattice is shared by both species, with each site's
    # diameter set after the fact (see dynamod's IPPacker case 8). For
    # size/concentration-asymmetric mixtures the target phi often cannot be
    # built directly this way (it would require overlapping same-species
    # neighbours), so when needed we instead build a loose, safely
    # non-overlapping configuration and grow it up to the target packing
    # fraction using DynamO's compression engine (dynarun --engine 3).
    direct_density = binary_density(N, n_large, n_small, delta, phi)
    if direct_density <= MAX_LATTICE_DENSITY:
        check_call(pack_args + ["-d", repr(direct_density), "-o", config], stdout=logfile, stderr=logfile)
    else:
        loose_config = os.path.join(workdir, "loose.config.xml.bz2")
        check_call(
            pack_args + ["-d", repr(MAX_LATTICE_DENSITY), "-o", loose_config], stdout=logfile, stderr=logfile
        )
        try:
            # cwd=workdir below, so arguments must be relative to it (not
            # workdir-prefixed) or dynarun looks for a doubled-up path.
            check_call(
                [
                    "dynarun",
                    os.path.basename(loose_config),
                    "--engine",
                    "3",
                    "--target-pack-frac",
                    repr(phi),
                    "-c",
                    "20000000",
                    "-o",
                    os.path.basename(config),
                    "--out-data-file",
                    "compression.data.xml.bz2",
                ],
                stdout=logfile,
                stderr=logfile,
                cwd=workdir,
            )
        except CalledProcessError:
            print(
                f"Skipping N={N} delta={delta} xhat={xhat} phi={phi}: compression "
                "to the target packing fraction failed (the system is likely too "
                "small for this size/concentration ratio -- try a larger N)",
                file=logfile,
                flush=True,
            )
            raise pydynamo.SkipThisPoint()

    # dynamod has no CLI switch for the restitution coefficient, nor for
    # adding a Gaussian thermostat, so both are patched into the XML here.
    xml = pydynamo.ConfigFile(config)
    interactions = xml.tree.findall(".//Interaction[@Type='HardSphere']")
    if not interactions:
        raise RuntimeError("No HardSphere interactions found after generation")
    for interaction in interactions:
        interaction.set("Elasticity", repr(e))

    system_events = xml.tree.find(".//SystemEvents")
    thermostat = pydynamo.ET.SubElement(system_events, "System")
    thermostat.set("Type", "Gaussian")
    thermostat.set("Name", "Thermostat")
    thermostat.set("MFT", repr(mft))
    thermostat.set("Temperature", "1")
    xml.save(config)


# ---------------------------------------------------------------------------
# Config-file introspection (lets pydynamo re-derive state vars if needed)
# ---------------------------------------------------------------------------
def _species_info(xmlconfig, name):
    tag = xmlconfig.tree.find(".//Species[@Name='%s']" % name)
    if tag is None:
        raise RuntimeError(f"Could not find Species named {name!r}")
    idrange = tag.find("IDRange")
    return float(tag.attrib["Mass"]), int(idrange.attrib["Start"]), int(idrange.attrib["End"])


def _interaction_diameter(xmlconfig, name):
    tag = xmlconfig.tree.find(".//Interaction[@Name='%s']" % name)
    if tag is None:
        raise RuntimeError(f"Could not find Interaction named {name!r}")
    return float(tag.attrib["Diameter"])


def delta_config(xmlconfig):
    return pydynamo.conv_to_14sf(
        _interaction_diameter(xmlconfig, "BBInt") / _interaction_diameter(xmlconfig, "AAInt")
    )


def xhat_config(xmlconfig):
    delta = delta_config(xmlconfig)
    _, startA, endA = _species_info(xmlconfig, "A")
    _, startB, endB = _species_info(xmlconfig, "B")
    n_large = endA - startA + 1
    n_small = endB - startB + 1
    small_vol = n_small * delta**3
    return pydynamo.conv_to_14sf(small_vol / (n_large + small_vol))


def phi_config(xmlconfig):
    delta = delta_config(xmlconfig)
    _, startA, endA = _species_info(xmlconfig, "A")
    _, startB, endB = _species_info(xmlconfig, "B")
    n_large = endA - startA + 1
    n_small = endB - startB + 1
    d_large = _interaction_diameter(xmlconfig, "AAInt")
    phi = (math.pi / 6.0) * (n_large * d_large**3 + n_small * (delta * d_large) ** 3) / xmlconfig.V()
    return pydynamo.conv_to_14sf(phi)


def e_config(xmlconfig):
    tag = xmlconfig.tree.find(".//Interaction[@Name='AAInt']")
    if tag is not None and "Elasticity" in tag.attrib:
        return pydynamo.conv_to_14sf(float(tag.attrib["Elasticity"]))
    return 1.0


def thermostat_mft_config(xmlconfig):
    tag = xmlconfig.tree.find(".//System[@Type='Gaussian']")
    if tag is None:
        return float("inf")
    return pydynamo.conv_to_14sf(float(tag.attrib["MFT"]))


pydynamo.ConfigFile.config_props["delta"] = {"recalculable": True, "recalc": delta_config}
pydynamo.ConfigFile.config_props["xhat"] = {"recalculable": True, "recalc": xhat_config}
pydynamo.ConfigFile.config_props["phi"] = {"recalculable": True, "recalc": phi_config}
pydynamo.ConfigFile.config_props["e"] = {"recalculable": True, "recalc": e_config}
pydynamo.ConfigFile.config_props["thermostat_mft"] = {
    "recalculable": True,
    "recalc": thermostat_mft_config,
}


# ---------------------------------------------------------------------------
# Output properties
# ---------------------------------------------------------------------------
pydynamo.OutputFile.output_props["D_A"] = SingleAttrib(
    "MSD/Species[@Name='A']", "diffusionCoeff", [], [], ["-LMSD"], missing_val=None, skip_missing=True
)
pydynamo.OutputFile.output_props["D_B"] = SingleAttrib(
    "MSD/Species[@Name='B']", "diffusionCoeff", [], [], ["-LMSD"], missing_val=None, skip_missing=True
)
pydynamo.OutputFile.output_props["T_system"] = SingleAttrib(
    "Temperature", "Mean", [], [], [], missing_val=None
)
pydynamo.OutputFile.output_props["phi_actual"] = SingleAttrib(
    "PackingFraction", "val", [], [], [], missing_val=None
)


class _SpeciesTemperature(OutputProperty):
    """Per-species kinetic (granular) temperature (k_B=1), evaluated from the
    velocities in the final configuration snapshot of each production block.

    Used to check for a breakdown of energy equipartition between species,
    which is expected to become more pronounced as e decreases and as the
    mass/size ratio becomes more disparate.
    """

    def __init__(self, species_name):
        OutputProperty.__init__(self, [], [], [])
        self.species_name = species_name

    def init(self):
        return WeightedType()

    def result(self, state, outputfile, configfilename, counter, manager, output_dir):
        config = ConfigFile(configfilename)
        mass, start, end = _species_info(config, self.species_name)
        ke_sum, count = 0.0, 0
        for pt in config.tree.findall(".//Pt"):
            if start <= int(pt.attrib["ID"]) <= end:
                v = pt.find("V")
                ke_sum += (
                    float(v.attrib["x"]) ** 2 + float(v.attrib["y"]) ** 2 + float(v.attrib["z"]) ** 2
                )
                count += 1
        if count == 0:
            return None
        temperature = mass * (ke_sum / count) / 3.0
        weight = float(outputfile.tree.find(".//Duration").attrib["Time"])
        return WeightedType(temperature, weight)


pydynamo.OutputFile.output_props["T_A"] = _SpeciesTemperature("A")
pydynamo.OutputFile.output_props["T_B"] = _SpeciesTemperature("B")
