"""Matplotlib figures for simulations, scans and critical-point analysis.

Every function returns the figure and leaves saving and closing to the
caller, so nothing accumulates when plotting in a loop.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import NullFormatter, ScalarFormatter

from . import exact

SPIN_CMAP = ListedColormap(["#1f3b73", "#f2c14e"])
EXACT_STYLE = {"color": "0.2", "linestyle": "--", "linewidth": 1.2}


def _size_colors(sizes):
    cmap = plt.get_cmap("viridis")
    sizes = sorted(sizes)
    n = max(len(sizes) - 1, 1)
    return {L: cmap(0.1 + 0.75 * i / n) for i, L in enumerate(sizes)}


def plot_equilibration(eq, ax=None, title=None, legend=True):
    """Hot- and cold-start energy and |magnetization| per spin against sweep."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4))
    n = eq.hot.n_sites
    sweeps = np.arange(1, len(eq.hot.energy) + 1)
    for run, style in ((eq.hot, "-"), (eq.cold, ":")):
        start = "hot" if run is eq.hot else "cold"
        ax.plot(
            sweeps, np.abs(run.magnetization) / n, "C0", ls=style, lw=1, label=f"|m|, {start} start"
        )
    for run, style in ((eq.hot, "-"), (eq.cold, ":")):
        start = "hot" if run is eq.hot else "cold"
        ax.plot(sweeps, run.energy / n, "C3", ls=style, lw=1, label=f"e, {start} start")
    if eq.sweep is not None:
        ax.axvline(eq.sweep, color="0.4", lw=1, label=f"equilibrated ({eq.sweep} sweeps)")
    ax.set_xscale("log")
    ax.set_xlabel("sweep")
    ax.set_ylabel("per spin")
    ax.set_ylim(-2.1, 1.1)
    ax.set_title(title or f"T = {eq.hot.T:g}, L = {round(n ** (1 / eq.hot.spins.ndim))}")
    if legend:
        ax.legend(fontsize=8, loc="center right")
    return fig


def plot_observables(scans, exact_curves=True):
    """2x2 grid of energy, |magnetization|, heat capacity and susceptibility.

    ``scans`` is a mapping ``{L: Scan}`` or a list of scans.
    """
    scans = {s.L: s for s in scans} if not isinstance(scans, dict) else scans
    colors = _size_colors(scans)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    panels = [
        ("energy", "energy per spin  $e$", exact.onsager_energy),
        ("abs_magnetization", r"magnetization  $\langle|m|\rangle$", exact.onsager_magnetization),
        ("heat_capacity", "heat capacity per spin  $c$", exact.onsager_heat_capacity),
        ("susceptibility", r"susceptibility per spin  $\chi$", None),
    ]
    T_all = np.concatenate([s.T for s in scans.values()])
    T_fine = np.linspace(T_all.min(), T_all.max(), 800)
    for ax, (name, label, curve) in zip(axes.flat, panels, strict=True):
        for L in sorted(scans):
            s = scans[L]
            ax.errorbar(
                s.T,
                getattr(s, name),
                getattr(s, f"{name}_err"),
                fmt="o-",
                ms=3,
                lw=1,
                capsize=1.5,
                color=colors[L],
                label=f"L = {L}",
            )
        if exact_curves and curve is not None:
            ax.plot(T_fine, curve(T_fine), label="exact (L = ∞)", **EXACT_STYLE)
        ax.axvline(exact.TC, color="0.7", lw=0.8, zorder=0)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[1, 1].set_yscale("log")
    axes[1, 1].yaxis.set_major_formatter(ScalarFormatter())
    axes[1, 1].yaxis.set_minor_formatter(NullFormatter())
    for ax in axes[1]:
        ax.set_xlabel("temperature  $T$")
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_binder(scans, ax=None, crossing=None):
    """Binder cumulant against temperature for several lattice sizes."""
    scans = {s.L: s for s in scans} if not isinstance(scans, dict) else scans
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6, 4.5))
    colors = _size_colors(scans)
    for L in sorted(scans):
        s = scans[L]
        ax.errorbar(
            s.T,
            s.binder,
            s.binder_err,
            fmt="o-",
            ms=3,
            lw=1,
            capsize=1.5,
            color=colors[L],
            label=f"L = {L}",
        )
    ax.axvline(exact.TC, label=f"exact $T_c$ = {exact.TC:.4f}", **EXACT_STYLE)
    if crossing is not None:
        ax.axvspan(
            crossing.value - crossing.error,
            crossing.value + crossing.error,
            color="C3",
            alpha=0.25,
            label=f"crossing {crossing.value:.4f}",
        )
    ax.set_xlabel("temperature  $T$")
    ax.set_ylabel(r"Binder cumulant  $U = 1 - \langle m^4\rangle / 3\langle m^2\rangle^2$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return fig


def plot_snapshots(configs, titles=None, ncols=None):
    """Side-by-side images of spin configurations (+1 gold, -1 blue)."""
    configs = list(configs)
    ncols = ncols or len(configs)
    nrows = int(np.ceil(len(configs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (ax, spins) in enumerate(zip(axes.flat, configs, strict=False)):
        ax.imshow(np.atleast_2d(spins), cmap=SPIN_CMAP, vmin=-1, vmax=1, interpolation="nearest")
        if titles:
            ax.set_title(titles[i], fontsize=11)
    fig.tight_layout()
    return fig


def animate(T, L, frames=200, sweeps_per_frame=1, algorithm="metropolis", seed=None, init="hot"):
    """Animation of a lattice evolving at temperature ``T``.

    Save it with ``anim.save("ising.gif", writer="pillow", fps=20)``.
    """
    from matplotlib.animation import FuncAnimation

    from .simulate import simulate

    rng = np.random.default_rng(seed)
    state = simulate(T, L, n_sweeps=1, burn_in=0, algorithm=algorithm, init=init, seed=rng).spins
    fig, ax = plt.subplots(figsize=(4, 4.3))
    ax.axis("off")
    image = ax.imshow(state, cmap=SPIN_CMAP, vmin=-1, vmax=1, interpolation="nearest")
    title = ax.set_title("")
    fig.tight_layout()
    sweep = 1

    def update(_):
        nonlocal state, sweep
        result = simulate(
            T, n_sweeps=sweeps_per_frame, burn_in=0, algorithm=algorithm, init=state, seed=rng
        )
        state = result.spins
        sweep += sweeps_per_frame
        image.set_data(state)
        title.set_text(f"T = {T:g}   sweep {sweep}")
        return image, title

    return FuncAnimation(fig, update, frames=frames, blit=False)
