"""Numba-compiled Monte Carlo update kernels.

Every kernel works on a flattened spin array together with a neighbour table
from :func:`ising.lattice.neighbour_table`, so the same code handles 1D chains
and 2D lattices. Spins are modified in place and the total energy and
magnetization are recorded once per sweep.
"""

import numpy as np
from numba import njit


def acceptance_table(T, J, h, z):
    """Metropolis acceptance probabilities min(1, exp(-dE/T)).

    Indexed as ``table[(s + 1) // 2, (nn + z) // 2]`` where ``s`` is the spin
    being flipped and ``nn`` the sum of its ``z`` neighbours.
    """
    s = np.array([-1, 1])[:, None]
    nn = np.arange(-z, z + 1, 2)[None, :]
    dE = 2.0 * s * (J * nn + h)
    return np.exp(-np.maximum(dE, 0.0) / T)


@njit(cache=True)
def total_energy(spins, neighbours, J, h):
    bonds = 0
    total = 0
    for i in range(spins.size):
        s = np.int64(spins[i])
        nn = 0
        for k in range(neighbours.shape[1]):
            nn += spins[neighbours[i, k]]
        bonds += s * nn
        total += s
    return -J * 0.5 * bonds - h * total


@njit(cache=True)
def metropolis(spins, neighbours, accept, J, h, n_sweeps, rng):
    """Single-spin-flip Metropolis with random site selection.

    One sweep is ``spins.size`` attempted flips. Returns the per-sweep total
    energy and magnetization.
    """
    n = spins.size
    z = neighbours.shape[1]
    energy = total_energy(spins, neighbours, J, h)
    magnet = np.int64(0)
    for i in range(n):
        magnet += spins[i]
    energies = np.empty(n_sweeps)
    mags = np.empty(n_sweeps)
    for sweep in range(n_sweeps):
        for _ in range(n):
            i = rng.integers(0, n)
            s = np.int64(spins[i])
            nn = np.int64(0)
            for k in range(z):
                nn += spins[neighbours[i, k]]
            p = accept[(s + 1) // 2, (nn + z) // 2]
            if p >= 1.0 or rng.random() < p:
                spins[i] = -s
                energy += 2.0 * s * (J * nn + h)
                magnet -= 2 * s
        energies[sweep] = energy
        mags[sweep] = magnet
    return energies, mags


@njit(cache=True)
def wolff(spins, neighbours, p_add, J, n_sweeps, clusters_per_sweep, rng):
    """Wolff single-cluster updates (zero field only).

    One sweep is ``clusters_per_sweep`` cluster flips. Pass 0 to instead flip
    clusters until at least ``spins.size`` spins have flipped; that stopping
    rule depends on the cluster sizes and so biases measurements, but it is
    fine for burn-in and for estimating the mean cluster size. Returns the
    per-sweep total energy, magnetization and mean cluster size.
    """
    n = spins.size
    z = neighbours.shape[1]
    stack = np.empty(n, dtype=np.int64)
    magnet = np.int64(0)
    for i in range(n):
        magnet += spins[i]
    energies = np.empty(n_sweeps)
    mags = np.empty(n_sweeps)
    mean_sizes = np.empty(n_sweeps)
    for sweep in range(n_sweeps):
        flipped = 0
        clusters = 0
        while (clusters < clusters_per_sweep) if clusters_per_sweep > 0 else (flipped < n):
            seed = rng.integers(0, n)
            s0 = spins[seed]
            # Flip sites as they join the cluster so each is pushed at most once.
            spins[seed] = -s0
            stack[0] = seed
            top = 1
            size = 1
            while top > 0:
                top -= 1
                site = stack[top]
                for k in range(z):
                    nb = neighbours[site, k]
                    if spins[nb] == s0 and rng.random() < p_add:
                        spins[nb] = -s0
                        stack[top] = nb
                        top += 1
                        size += 1
            flipped += size
            clusters += 1
            magnet -= 2 * np.int64(s0) * size
        energies[sweep] = total_energy(spins, neighbours, J, 0.0)
        mags[sweep] = magnet
        mean_sizes[sweep] = flipped / clusters
    return energies, mags, mean_sizes
