#!/usr/bin/env python3
"""
Analyze/plot the results produced by run_restitution_sweep.py.

Two kinds of output are produced:

1. fig1_reproduction.png -- an attempt at reproducing Voigtmann's Fig. 1(a):
   phi_c(xhat) curves for each size ratio delta, one colour per restitution
   coefficient e. phi_c is estimated by fitting the large-species diffusion
   coefficient D_A(phi) to a vanishing power law near the transition (the
   large species always freezes at the primary glass line, whether or not
   the small species remains mobile in a "single glass"). This fit is
   noisy for finite MD runs and may fail for many (delta, xhat, e)
   combinations -- treat it as qualitative at best.

2. state_diagram_delta_<delta>.png -- a much more robust fallback that does
   not require any extrapolation: each simulated (xhat, phi) point is
   classified directly as "liquid" (both species diffusing), "single
   glass" (only the small species still diffusing) or "double glass" (both
   arrested), based on a diffusion-coefficient threshold. One panel per
   restitution coefficient e, so the shift of the arrested region with e
   is directly visible.

Run this after (or repeatedly during) run_restitution_sweep.py:

    python3 analyze_restitution_sweep.py
"""
import numpy
import pydynamo
from scipy.optimize import curve_fit

import restitution_common as common

# Below this, a species' diffusion coefficient is considered zero within
# the noise of a finite MD run (units: particle diameters^2 / unitTime).
ARREST_THRESHOLD = 2e-3


def load_results():
    mgr = pydynamo.SimManager(
        common.WORKDIR,
        common.STATEVARS,
        common.OUTPUTS,
        restarts=common.RESTARTS,
        processes=None,
    )
    df = mgr.fetch_data(common.PARTICLE_EQUIL_EVENTS)
    if len(df) == 0:
        raise SystemExit(
            f'No results found in "{common.WORKDIR}". Run run_restitution_sweep.py first.'
        )

    for col in ("D_A", "D_B", "T_system", "phi_actual", "T_A", "T_B"):
        if col not in df.columns:
            continue
        # State points without production data yet (e.g. an interrupted
        # sweep) show up as plain NaN floats rather than ufloat objects.
        df[col + "_val"] = df[col].apply(lambda x: x.nominal_value if hasattr(x, "nominal_value") else numpy.nan)
        df[col + "_err"] = df[col].apply(lambda x: x.std_dev if hasattr(x, "nominal_value") else numpy.nan)

    df.to_csv(common.WORKDIR + "_results.csv", index=False)
    print(f"Loaded {len(df)} state points, wrote {common.WORKDIR}_results.csv")
    return df


def classify_state(row):
    d_a, d_b = row["D_A_val"], row["D_B_val"]
    if numpy.isnan(d_a) or numpy.isnan(d_b):
        return "no data"
    mobile_a = d_a > ARREST_THRESHOLD
    mobile_b = d_b > ARREST_THRESHOLD
    if mobile_a and mobile_b:
        return "liquid"
    if (not mobile_a) and mobile_b:
        return "single glass"
    if (not mobile_a) and (not mobile_b):
        return "double glass"
    return "anomalous"  # B arrested but A mobile: not physically expected


def _power_law(phi, amplitude, phi_c, gamma):
    return amplitude * numpy.clip(phi_c - phi, 0, None) ** gamma


def fit_phi_c(phi, diffusion):
    """Fit D(phi) = amplitude * max(phi_c - phi, 0)**gamma and return
    (phi_c, sigma_phi_c), or None if there isn't enough usable data."""
    order = numpy.argsort(phi)
    phi = numpy.asarray(phi)[order]
    diffusion = numpy.asarray(diffusion)[order]
    mobile = diffusion > ARREST_THRESHOLD
    if mobile.sum() < 4:
        return None

    p0 = [max(diffusion[mobile][0], 1e-3), phi.max() + 0.01, 2.0]
    try:
        popt, pcov = curve_fit(
            _power_law,
            phi[mobile],
            diffusion[mobile],
            p0=p0,
            maxfev=20000,
            bounds=([0, phi.min(), 0.3], [numpy.inf, phi.max() + 0.25, 6.0]),
        )
    except RuntimeError:
        return None
    sigma = numpy.sqrt(pcov[1, 1]) if numpy.isfinite(pcov[1, 1]) else float("nan")
    return popt[1], sigma


def make_state_diagrams(df):
    import matplotlib.pyplot as plt

    df = df.copy()
    df["state"] = df.apply(classify_state, axis=1)
    n_missing = (df["state"] == "no data").sum()
    if n_missing:
        print(f"Note: {n_missing} state points have no MSD data yet (still running?)")
    df = df[df["state"] != "no data"]

    colours = {
        "liquid": "tab:blue",
        "single glass": "tab:orange",
        "double glass": "tab:red",
        "anomalous": "tab:gray",
    }

    for delta, group_delta in df.groupby("delta"):
        e_values = sorted(group_delta["e"].unique())
        fig, axes = plt.subplots(
            1, len(e_values), figsize=(4 * len(e_values), 4), sharey=True, squeeze=False
        )
        for ax, e in zip(axes[0], e_values):
            sub = group_delta[group_delta["e"] == e]
            for state, subsub in sub.groupby("state"):
                ax.scatter(subsub["xhat"], subsub["phi"], c=colours[state], label=state, s=40)
            ax.set_title(f"delta={delta}, e={e}")
            ax.set_xlabel(r"$\hat{x}$")
        axes[0][0].set_ylabel(r"$\varphi$")
        axes[0][0].legend(loc="best", fontsize=8)
        fig.suptitle(f"State diagram, size ratio delta={delta}")
        fig.tight_layout()
        outname = f"state_diagram_delta_{delta}.png"
        fig.savefig(outname, dpi=150)
        plt.close(fig)
        print(f"Wrote {outname}")


def make_fig1_reproduction(df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    e_values = sorted(df["e"].unique())
    cmap = plt.get_cmap("viridis")
    linestyles = ["-", "--", "-.", ":"]

    any_fit = False
    for d_idx, (delta, group_delta) in enumerate(sorted(df.groupby("delta"))):
        for e in e_values:
            group = group_delta[group_delta["e"] == e]
            xs, ys, yerrs = [], [], []
            for xhat, group_x in group.groupby("xhat"):
                fit = fit_phi_c(group_x["phi"].values, group_x["D_A_val"].values)
                if fit is None:
                    continue
                xs.append(xhat)
                ys.append(fit[0])
                yerrs.append(fit[1])
            if len(xs) < 2:
                continue
            any_fit = True
            order = numpy.argsort(xs)
            xs = numpy.array(xs)[order]
            ys = numpy.array(ys)[order]
            yerrs = numpy.array(yerrs)[order]
            colour = cmap(e_values.index(e) / max(1, len(e_values) - 1))
            ax.errorbar(
                xs,
                ys,
                yerr=yerrs,
                marker="o",
                linestyle=linestyles[d_idx % len(linestyles)],
                color=colour,
                label=f"delta={delta}, e={e}",
            )

    ax.set_xlabel(r"small-particle volume concentration $\hat{x}$")
    ax.set_ylabel(r"glass-transition packing fraction $\varphi_c$ (large-species arrest)")
    ax.set_title("MD attempt at reproducing Voigtmann EPL 96 36006 Fig. 1(a)")
    if any_fit:
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig("fig1_reproduction.png", dpi=150)
        print("Wrote fig1_reproduction.png")
    else:
        print(
            "Not enough data to fit any phi_c(xhat) curve yet (need at least "
            "4 packing fractions per point with a resolvable D_A(phi) decay). "
            "See the state_diagram_delta_*.png files instead."
        )
    plt.close(fig)


def main():
    df = load_results()
    make_state_diagrams(df)
    make_fig1_reproduction(df)


if __name__ == "__main__":
    main()
