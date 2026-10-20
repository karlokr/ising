"""High-level entry points for running Ising model simulations."""

from dataclasses import dataclass

import numpy as np

from . import _kernels
from .analysis import thermodynamics
from .lattice import neighbour_table, ordered_spins, random_spins, validate_spins

ALGORITHMS = ("metropolis", "wolff")


@dataclass(frozen=True)
class Result:
    """Output of :func:`simulate`.

    ``energy`` and ``magnetization`` hold the total (not per-spin) values after
    each measured sweep; ``spins`` is the final configuration.
    """

    T: float
    J: float
    h: float
    algorithm: str
    burn_in: int
    spins: np.ndarray
    energy: np.ndarray
    magnetization: np.ndarray

    @property
    def n_sites(self):
        return self.spins.size

    def observables(self, n_blocks=20):
        """Per-spin averages with jackknife errors; see :class:`ising.analysis.Observables`."""
        return thermodynamics(self.energy, self.magnetization, self.n_sites, self.T, n_blocks)


def initial_spins(init, L=None, dim=2, rng=None):
    """Build a starting configuration from ``"hot"``, ``"cold"`` or an array.

    Arrays are copied, so the caller's configuration is never modified.
    """
    if isinstance(init, str):
        if L is None:
            raise ValueError("L is required unless init is an array")
        if dim not in (1, 2):
            raise ValueError(f"dim must be 1 or 2, got {dim}")
        shape = (int(L),) * dim
        if init == "hot":
            return validate_spins(random_spins(shape, rng))
        if init == "cold":
            return validate_spins(ordered_spins(shape))
        raise ValueError(f"init must be 'hot', 'cold' or an array, got {init!r}")
    spins = validate_spins(np.array(init, copy=True))
    if L is not None and spins.shape != (int(L),) * spins.ndim:
        raise ValueError(f"init has shape {spins.shape}, which does not match L={L}")
    return spins


def _run_metropolis(spins, neighbours, T, J, h, n_sweeps, burn_in, rng):
    flat = spins.reshape(-1)
    accept = _kernels.acceptance_table(T, J, h, neighbours.shape[1])
    if burn_in:
        _kernels.metropolis(flat, neighbours, accept, J, h, burn_in, rng)
    return _kernels.metropolis(flat, neighbours, accept, J, h, n_sweeps, rng)


def _run_wolff(spins, neighbours, T, J, n_sweeps, burn_in, rng):
    flat = spins.reshape(-1)
    p_add = 1.0 - np.exp(-2.0 * J / T)
    # Measuring after a fixed number of clusters keeps the samples unbiased; the
    # number is chosen so a sweep flips about one lattice's worth of spins.
    if burn_in:
        *_, sizes = _kernels.wolff(flat, neighbours, p_add, J, burn_in, 0, rng)
        mean_size = sizes[burn_in // 2 :].mean()
    else:
        *_, sizes = _kernels.wolff(flat.copy(), neighbours, p_add, J, 20, 0, rng)
        mean_size = sizes[10:].mean()
    clusters_per_sweep = max(1, round(flat.size / mean_size))
    energy, magnetization, _ = _kernels.wolff(
        flat, neighbours, p_add, J, n_sweeps, clusters_per_sweep, rng
    )
    return energy, magnetization


def _check_parameters(T, J, h, algorithm, n_sweeps, burn_in):
    if not T > 0:
        raise ValueError(f"T must be positive, got {T}")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}")
    if algorithm == "wolff" and (h != 0 or J <= 0):
        raise ValueError("wolff requires h = 0 and a ferromagnetic J > 0")
    if n_sweeps < 1 or burn_in < 0:
        raise ValueError("need n_sweeps >= 1 and burn_in >= 0")


def simulate(
    T,
    L=None,
    *,
    dim=2,
    J=1.0,
    h=0.0,
    n_sweeps=10_000,
    burn_in=1_000,
    algorithm="metropolis",
    init="hot",
    seed=None,
):
    """Simulate the Ising model at temperature ``T``.

    Parameters
    ----------
    T : float
        Temperature in units of J / k_B.
    L : int
        Linear lattice size; the lattice has ``L**dim`` sites. May be omitted
        when ``init`` is an array.
    dim : {1, 2}
        Lattice dimension.
    J, h : float
        Coupling and external field.
    n_sweeps : int
        Number of measured sweeps (one measurement per sweep).
    burn_in : int
        Sweeps discarded before measuring, to reach equilibrium.
    algorithm : {"metropolis", "wolff"}
        Single-spin Metropolis, or Wolff cluster updates (h = 0, J > 0 only).
        Wolff is much faster near the critical temperature.
    init : {"hot", "cold"} or array of +/-1
        Starting configuration. Arrays are copied, not modified.
    seed : int or numpy.random.Generator, optional
        Seed for reproducible runs.
    """
    T, J, h = float(T), float(J), float(h)
    n_sweeps, burn_in = int(n_sweeps), int(burn_in)
    _check_parameters(T, J, h, algorithm, n_sweeps, burn_in)
    rng = np.random.default_rng(seed)
    spins = initial_spins(init, L, dim, rng)
    neighbours = neighbour_table(spins.shape)
    if algorithm == "metropolis":
        energy, magnetization = _run_metropolis(spins, neighbours, T, J, h, n_sweeps, burn_in, rng)
    else:
        energy, magnetization = _run_wolff(spins, neighbours, T, J, n_sweeps, burn_in, rng)
    return Result(T, J, h, algorithm, burn_in, spins, energy, magnetization)


@dataclass(frozen=True)
class Equilibration:
    """Hot- and cold-start runs from :func:`equilibration`.

    ``sweep`` is the estimated equilibration time in sweeps, or ``None`` if the
    two runs never met.
    """

    hot: Result
    cold: Result
    sweep: int | None


def equilibration(
    T,
    L,
    *,
    dim=2,
    J=1.0,
    h=0.0,
    n_sweeps=1_000,
    window=10,
    algorithm="metropolis",
    seed=None,
):
    """Estimate the equilibration time by comparing a hot and a cold start.

    Both runs start from scratch (no burn-in). A hot start begins with higher
    energy and lower |magnetization| than a cold start; the system is taken to
    be in equilibrium once the ``window``-sweep averages of both quantities
    have crossed. Noise can only make the crossing happen early, once the runs
    are already within their fluctuations of each other, so the estimate does
    not stall on small, noisy lattices. A hot start trapped in a metastable
    state never crosses and gives ``None``. For production runs pick a
    ``burn_in`` comfortably larger than the result.
    """
    rng = np.random.default_rng(seed)
    hot, cold = (
        simulate(
            T,
            L,
            dim=dim,
            J=J,
            h=h,
            n_sweeps=n_sweeps,
            burn_in=0,
            algorithm=algorithm,
            init=start,
            seed=rng,
        )
        for start in ("hot", "cold")
    )
    n_windows = n_sweeps // window

    def window_means(x):
        return x[: n_windows * window].reshape(n_windows, window).mean(axis=1)

    energy_met = window_means(hot.energy) <= window_means(cold.energy)
    magnet_met = window_means(np.abs(hot.magnetization)) >= window_means(np.abs(cold.magnetization))
    if not (energy_met.any() and magnet_met.any()):
        return Equilibration(hot, cold, None)
    first = max(np.argmax(energy_met), np.argmax(magnet_met))
    return Equilibration(hot, cold, int((first + 1) * window))
