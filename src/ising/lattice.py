"""Spin configurations and observables for the nearest-neighbour Ising model.

Spins are ``int8`` arrays of +/-1 on a periodic 1D chain or 2D square lattice,
with Hamiltonian

    H = -J * sum_<ij> s_i s_j - h * sum_i s_i

where the first sum runs over nearest-neighbour pairs (each pair once).
"""

import numpy as np

SPIN_DTYPE = np.int8


def random_spins(shape, rng=None):
    """Infinite-temperature ("hot") configuration of independent random spins."""
    rng = np.random.default_rng(rng)
    return rng.choice(np.array([-1, 1], dtype=SPIN_DTYPE), size=shape)


def ordered_spins(shape, value=1):
    """Zero-temperature ("cold") configuration with every spin equal to ``value``."""
    if value not in (-1, 1):
        raise ValueError("value must be +1 or -1")
    return np.full(shape, value, dtype=SPIN_DTYPE)


def validate_spins(spins):
    """Return ``spins`` as a contiguous int8 array, checking shape and values."""
    spins = np.ascontiguousarray(spins)
    if spins.ndim not in (1, 2):
        raise ValueError(f"spins must be 1D or 2D, got {spins.ndim}D")
    if min(spins.shape) < 2:
        raise ValueError(f"every lattice dimension must be >= 2, got {spins.shape}")
    if not np.all((spins == 1) | (spins == -1)):
        raise ValueError("spins must contain only +1 and -1")
    return spins.astype(SPIN_DTYPE, copy=False)


def neighbour_table(shape):
    """Flat indices of the 2*ndim nearest neighbours of every site, shape (n_sites, z)."""
    index = np.arange(int(np.prod(shape))).reshape(shape)
    columns = []
    for axis in range(len(shape)):
        columns.append(np.roll(index, 1, axis).ravel())
        columns.append(np.roll(index, -1, axis).ravel())
    return np.stack(columns, axis=1).astype(np.int64)


def neighbour_sum(spins):
    """Sum of the nearest neighbours of every site (periodic boundaries)."""
    total = np.zeros(spins.shape, dtype=np.int64)
    for axis in range(spins.ndim):
        total += np.roll(spins, 1, axis) + np.roll(spins, -1, axis)
    return total


def energy(spins, J=1.0, h=0.0):
    """Total energy of a configuration."""
    s = np.asarray(spins, dtype=np.int64)
    bonds = sum(int(np.sum(s * np.roll(s, 1, axis))) for axis in range(s.ndim))
    return -J * bonds - h * int(s.sum())


def magnetization(spins):
    """Total (signed) magnetization of a configuration."""
    return int(np.sum(spins, dtype=np.int64))


def delta_energy(spins, site, J=1.0, h=0.0):
    """Energy change from flipping the spin at ``site`` (a tuple index)."""
    s = int(spins[site])
    nn = 0
    for axis, i in enumerate(site):
        n = spins.shape[axis]
        for step in (-1, 1):
            neighbour = list(site)
            neighbour[axis] = (i + step) % n
            nn += int(spins[tuple(neighbour)])
    return 2 * s * (J * nn + h)
