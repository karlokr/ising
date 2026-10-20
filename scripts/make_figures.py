"""Regenerate every figure in the README.

Simulation results are cached in docs/data, so re-running only redraws the
figures; pass --force to recompute. A full run takes about 25 minutes on
12 cores.

    python scripts/make_figures.py [--force] [--only NAME ...]
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

import ising
from ising import critical, exact, plotting
from ising.analysis import autocorrelation_time
from ising.scan import read_csv, temperature_scan, write_csv

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "docs" / "figures"
DATA = ROOT / "docs" / "data"

SIZES = (16, 32, 64, 128)
CRITICAL_SIZES = (16, 32, 64, 128, 256)
BROAD_T = np.round(
    np.r_[np.arange(1.5, 2.1, 0.1), np.arange(2.1, 2.5, 0.02), np.arange(2.5, 3.55, 0.1)], 4
)
# Twice as dense where the large-L peaks and Binder crossings sit.
CRITICAL_T = np.unique(
    np.round(np.r_[np.arange(2.22, 2.4001, 0.005), np.arange(2.255, 2.3101, 0.0025)], 4)
)


def save(fig, name):
    path = FIGURES / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def cached(name, force, compute):
    """Load JSON data from docs/data, or compute and store it."""
    path = DATA / name
    if path.exists() and not force:
        return json.loads(path.read_text())
    data = compute()
    path.write_text(json.dumps(data, indent=1))
    return data


def cached_scan(name, force, temps, sizes, **kwargs):
    path = DATA / name
    if path.exists() and not force:
        return read_csv(path)
    scans = []
    for i, L in enumerate(sizes):
        start = time.perf_counter()
        scans.append(temperature_scan(temps, L, seed=1000 + i, **kwargs))
        print(f"    L = {L}: {time.perf_counter() - start:.0f} s")
    write_csv(scans, path)
    return {s.L: s for s in scans}


# --- figures -----------------------------------------------------------------


def fig_snapshots(force):
    temps = [1.8, 2.1, round(exact.TC, 3), 2.5, 3.2]
    configs = [
        ising.simulate(T, 256, n_sweeps=1, burn_in=3000, algorithm="wolff", seed=7).spins
        for T in temps
    ]
    titles = [f"T = {T}" + ("  (≈ $T_c$)" if abs(T - exact.TC) < 0.01 else "") for T in temps]
    save(plotting.plot_snapshots(configs, titles), "snapshots.png")


def fig_animation(force):
    anim = plotting.animate(exact.TC, 160, frames=120, sweeps_per_frame=2, seed=3)
    path = FIGURES / "quench.gif"
    anim.save(path, writer="pillow", fps=15, dpi=80)
    plt.close("all")
    print(f"  wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")


def fig_equilibration(force):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, T in zip(axes, (1.8, 3.0), strict=True):
        eq = ising.equilibration(T, 40, n_sweeps=3000, seed=4)
        plotting.plot_equilibration(eq, ax=ax, legend=False)
        ax.text(eq.sweep * 1.1, 0.6, f"equilibrated\n{eq.sweep} sweeps", fontsize=8, color="0.3")
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "equilibration.png")


def _equilibration_time(args):
    T, L, seed = args
    return ising.equilibration(T, L, n_sweeps=40_000, seed=seed).sweep


def _tau(args):
    T, L, algorithm, n_sweeps, seed = args
    r = ising.simulate(
        T, L, n_sweeps=n_sweeps, burn_in=n_sweeps // 10, algorithm=algorithm, seed=seed
    )
    return autocorrelation_time(np.abs(r.magnetization))


def fig_dynamics(force):
    def compute():
        workers = os.cpu_count()
        out = {"equilibration": {}, "tau": {}}
        seeds = range(12)
        with ProcessPoolExecutor(workers) as pool:
            for label, T, sizes in (
                ("T=3.0", 3.0, (8, 16, 32, 64, 128)),
                ("T=Tc", exact.TC, (8, 16, 32, 64)),
            ):
                times = {}
                for L in sizes:
                    sweeps = list(pool.map(_equilibration_time, [(T, L, s) for s in seeds]))
                    times[L] = [s for s in sweeps if s is not None]
                    print(f"    equilibration {label} L={L}: median {np.median(times[L]):.0f}")
                out["equilibration"][label] = times
            jobs = {
                "metropolis": [(L, 400 * L**2 // 8 + 20_000) for L in (8, 16, 32, 64)],
                "wolff": [(L, 20_000) for L in (8, 16, 32, 64, 128, 256)],
            }
            for algorithm, sizes in jobs.items():
                args = [(exact.TC, L, algorithm, n, 50 + L) for L, n in sizes]
                taus = list(pool.map(_tau, args))
                out["tau"][algorithm] = {L: t for (L, _), t in zip(sizes, taus, strict=True)}
                print(f"    tau {algorithm}: {out['tau'][algorithm]}")
        return out

    data = cached("dynamics.json", force, compute)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for (label, times), color in zip(data["equilibration"].items(), ("C0", "C3"), strict=True):
        L = np.array([int(k) for k in times])
        med = np.array([np.median(v) for v in times.values()])
        lo = med - np.array([np.percentile(v, 25) for v in times.values()])
        hi = np.array([np.percentile(v, 75) for v in times.values()]) - med
        ax1.errorbar(
            L, med, [lo, hi], fmt="o-", color=color, capsize=3, label=label.replace("Tc", "$T_c$")
        )
    ax1.set(xscale="log", yscale="log", xlabel="L", ylabel="sweeps to equilibrate (median, IQR)")
    ax1.set_title("Metropolis equilibration time")
    ax1.legend()
    for (algorithm, taus), color in zip(data["tau"].items(), ("C3", "C2"), strict=True):
        L = np.array([int(k) for k in taus])
        tau = np.array(list(taus.values()))
        z, logc = np.polyfit(np.log(L), np.log(tau), 1)
        ax2.plot(L, tau, "o", color=color)
        Lfit = np.geomspace(L.min(), L.max())
        ax2.plot(
            Lfit,
            np.exp(logc) * Lfit**z,
            "-",
            color=color,
            label=f"{algorithm}: $\\tau \\propto L^{{{z:.2f}}}$",
        )
    ax2.set(
        xscale="log", yscale="log", xlabel="L", ylabel=r"autocorrelation time of $|m|$ (sweeps)"
    )
    ax2.set_title(r"Critical slowing down at $T_c$")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    save(fig, "dynamics.png")


def fig_observables(force):
    scans = cached_scan(
        "observables.csv", force, BROAD_T, SIZES, algorithm="wolff", n_sweeps=20_000, burn_in=2_000
    )
    save(plotting.plot_observables(scans), "observables.png")


def fig_critical(force):
    scans = cached_scan(
        "critical.csv",
        force,
        CRITICAL_T,
        CRITICAL_SIZES,
        algorithm="wolff",
        n_sweeps=40_000,
        burn_in=4_000,
    )
    sizes = sorted(scans)
    results = {}

    # Binder cumulant crossings of successive sizes.
    crossings = {}
    for small, large in zip(sizes[:-1], sizes[1:], strict=True):
        a, b = scans[small], scans[large]
        crossings[(small, large)] = critical.binder_crossing(
            a.T, a.binder, b.binder, a.binder_err, b.binder_err
        )
    best = crossings[(sizes[-2], sizes[-1])]
    fig = plotting.plot_binder(scans, crossing=best)
    fig.axes[0].set_xlim(2.22, 2.34)
    fig.axes[0].set_ylim(0.45, 0.68)
    save(fig, "binder.png")
    results["binder"] = {f"{a}/{b}": list(c) for (a, b), c in crossings.items()}

    # Peak positions and heights.
    peaks = {}
    for name in ("heat_capacity", "susceptibility"):
        pts = [
            critical.peak(scans[L].T, getattr(scans[L], name), getattr(scans[L], f"{name}_err"))
            for L in sizes
        ]
        peaks[name] = pts
    L = np.array(sizes, float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    x_fit = np.linspace(0, 1 / L.min() * 1.05, 50)
    for (name, pts), color, label in zip(
        peaks.items(), ("C0", "C3"), ("heat capacity", "susceptibility"), strict=True
    ):
        tp = np.array([p[0].value for p in pts])
        te = np.array([p[0].error for p in pts])
        tc, slope = critical.extrapolate_peak(L, tp, te)
        results[f"peak_{name}"] = list(tc)
        ax.errorbar(1 / L, tp, te, fmt="o", color=color, capsize=3, label=f"{label} peak")
        ax.plot(x_fit, tc.value + slope.value * x_fit, "-", color=color, lw=1,
                label=f"fit: $T_c$ = {tc.value:.4f} ± {tc.error:.4f}")  # fmt: skip
    ax.axhline(exact.TC, label=f"exact {exact.TC:.4f}", **plotting.EXACT_STYLE)
    ax.set(xlabel="1 / L", ylabel="peak temperature", xlim=(0, None))
    ax.set_title("Peak positions extrapolated to $L \\to \\infty$")
    ax.legend(fontsize=8)

    ax = axes[1]
    heights = np.array([p[1].value for p in peaks["heat_capacity"]])
    herr = np.array([p[1].error for p in peaks["heat_capacity"]])
    A, B = critical.fit_log_growth(L, heights, herr)
    results["log_growth_A"] = list(A)
    ax.errorbar(L, heights, herr, fmt="o", color="C0", capsize=3, label="heat capacity peak")
    Lf = np.geomspace(L.min() / 1.3, L.max() * 1.3)
    ax.plot(Lf, A.value * np.log(Lf) + B.value, "-", color="C0", lw=1,
            label=f"fit: $c_{{max}}$ = {A.value:.3f} ln L + {B.value:.2f}")  # fmt: skip
    A_exact = 2 / np.pi * (2 / exact.TC) ** 2
    ax.plot(
        Lf,
        A_exact * np.log(Lf) + 0.1385,
        label=f"exact at $T_c$: {A_exact:.4f} ln L + 0.1385",
        **plotting.EXACT_STYLE,
    )
    ax.set(xscale="log", xlabel="L", ylabel="peak heat capacity per spin")
    ax.set_xticks(L, [str(int(x)) for x in L])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title("Peak height grows as ln L  ($\\alpha = 0$)")
    ax.legend(fontsize=8)

    ax = axes[2]
    big = scans[sizes[-1]]
    broad = read_csv(DATA / "observables.csv")[max(SIZES)]
    T = np.r_[broad.T[broad.T < CRITICAL_T[0]], big.T, broad.T[broad.T > CRITICAL_T[-1]]]
    c = np.r_[
        broad.heat_capacity[broad.T < CRITICAL_T[0]],
        big.heat_capacity,
        broad.heat_capacity[broad.T > CRITICAL_T[-1]],
    ]
    ce = np.r_[
        broad.heat_capacity_err[broad.T < CRITICAL_T[0]],
        big.heat_capacity_err,
        broad.heat_capacity_err[broad.T > CRITICAL_T[-1]],
    ]
    window = (T > 1.95) & (T < 2.65)
    T, c, ce = T[window], c[window], ce[window]
    fit = critical.fit_log_divergence(T, c, ce, exclude=0.03)
    results["log_divergence"] = {k: list(v) for k, v in fit.items() if k in ("Tc", "A")}
    used = fit["used"]
    ax.errorbar(T[used], c[used], ce[used], fmt="o", ms=3, color="C0", label=f"L = {big.L}, fitted")
    ax.errorbar(
        T[~used],
        c[~used],
        ce[~used],
        fmt="o",
        ms=3,
        mfc="none",
        color="C0",
        alpha=0.6,
        label="excluded (finite-size rounding)",
    )
    Tf = np.linspace(T.min(), T.max(), 600)
    tc, a = fit["Tc"].value, fit["A"].value
    below = Tf < tc
    curve = -a * np.log(np.abs(1 - Tf / tc))
    # Rebuild the full model for drawing: backgrounds and slope from a refit at tc.
    X = np.column_stack(
        [-np.log(np.abs(1 - T[used] / tc)), T[used] < tc, T[used] >= tc, T[used] - tc]
    )
    coef, *_ = np.linalg.lstsq(X / ce[used, None], c[used] / ce[used], rcond=None)
    curve = coef[0] * curve / a + np.where(below, coef[1], coef[2]) + coef[3] * (Tf - tc)
    ax.plot(
        Tf,
        curve,
        "-",
        color="C3",
        lw=1.2,
        label=f"$-A\\,\\ln|1-T/T_c(L)|$ fit, A = {a:.3f}",
    )
    ax.plot(Tf, exact.onsager_heat_capacity(Tf), label="exact (L = ∞)", **plotting.EXACT_STYLE)
    ax.set(xlabel="temperature  T", ylabel="heat capacity per spin", ylim=(-0.1, 3.6))
    ax.set_title("Logarithmic divergence of the heat capacity")
    ax.legend(fontsize=8, loc="lower center")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "critical.png")
    (DATA / "tc_estimates.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


FIGURES_BY_NAME = {
    "snapshots": fig_snapshots,
    "animation": fig_animation,
    "equilibration": fig_equilibration,
    "dynamics": fig_dynamics,
    "observables": fig_observables,
    "critical": fig_critical,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--force", action="store_true", help="recompute cached simulation data")
    parser.add_argument("--only", nargs="+", choices=FIGURES_BY_NAME, help="figures to make")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    for name in args.only or FIGURES_BY_NAME:
        start = time.perf_counter()
        print(f"{name}:")
        FIGURES_BY_NAME[name](args.force)
        print(f"  ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
