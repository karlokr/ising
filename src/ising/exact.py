"""Exact thermodynamic-limit results used for validation and plotting.

All quantities are per spin with k_B = 1. The 2D results are Onsager's solution
for the square lattice in zero field; the 1D results come from the transfer
matrix of the infinite chain in a field h.
"""

import numpy as np

#: Critical temperature of the 2D square-lattice model for J = 1.
TC = 2.0 / np.log(1.0 + np.sqrt(2.0))


def critical_temperature(J=1.0):
    """Onsager critical temperature 2J / ln(1 + sqrt 2)."""
    return J * TC


def _elliptic_ke(k):
    """Complete elliptic integrals K(k) and E(k) (modulus k) via the AGM."""
    k = np.asarray(k, dtype=float)
    a = np.ones_like(k)
    b = np.sqrt(1.0 - k * k)
    c = k.copy()
    weighted = 0.5 * c * c  # sum of 2^(n-1) c_n^2, starting at n = 0
    power = 0.5
    for _ in range(40):
        a, b, c = 0.5 * (a + b), np.sqrt(a * b), 0.5 * (a - b)
        power *= 2.0
        weighted += power * c * c
    singular = k >= 1.0  # K diverges logarithmically and E -> 1 as k -> 1
    K = np.where(singular, np.inf, np.pi / (2.0 * a))
    E = np.where(singular, 1.0, np.pi / (2.0 * a) * (1.0 - weighted))
    return K, E


def onsager_magnetization(T, J=1.0):
    """Spontaneous magnetization per spin, (1 - sinh(2J/T)^-4)^(1/8) below Tc."""
    T = np.asarray(T, dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        x = 1.0 - np.sinh(2.0 * J / T) ** -4
        m = np.where(T < critical_temperature(J), np.abs(x) ** 0.125, 0.0)
    return m[()] if m.ndim == 0 else m


def onsager_energy(T, J=1.0):
    """Internal energy per spin of the infinite 2D lattice."""
    T = np.asarray(T, dtype=float)
    beta_j = J / T
    t2 = np.tanh(2.0 * beta_j) ** 2
    kappa = 2.0 * np.sinh(2.0 * beta_j) / np.cosh(2.0 * beta_j) ** 2
    K1, _ = _elliptic_ke(np.minimum(kappa, 1.0))
    with np.errstate(invalid="ignore"):
        term = np.where(np.isinf(K1), 0.0, (2.0 * t2 - 1.0) * K1)
    u = -J / np.tanh(2.0 * beta_j) * (1.0 + 2.0 / np.pi * term)
    return u[()] if u.ndim == 0 else u


def onsager_heat_capacity(T, J=1.0):
    """Heat capacity per spin of the infinite 2D lattice (diverges at Tc)."""
    T = np.asarray(T, dtype=float)
    beta_j = J / T
    t2 = np.tanh(2.0 * beta_j) ** 2
    kappa = 2.0 * np.sinh(2.0 * beta_j) / np.cosh(2.0 * beta_j) ** 2
    K1, E1 = _elliptic_ke(np.minimum(kappa, 1.0))
    prefactor = 4.0 / np.pi * (beta_j / np.tanh(2.0 * beta_j)) ** 2
    with np.errstate(invalid="ignore"):
        bracket = K1 - E1 - (1.0 - t2) * (np.pi / 2.0 + (2.0 * t2 - 1.0) * K1)
    c = prefactor * bracket
    return c[()] if c.ndim == 0 else c


def _chain_eigenvalue(beta, J, h):
    """Largest transfer-matrix eigenvalue of the 1D chain and its beta-derivative."""
    eK, ch, sh = np.exp(beta * J), np.cosh(beta * h), np.sinh(beta * h)
    root = np.sqrt(eK**2 * sh**2 + eK**-2)
    lam = eK * ch + root
    droot = (J * eK**2 * sh**2 + h * eK**2 * sh * ch - J * eK**-2) / root
    dlam = J * eK * ch + h * eK * sh + droot
    return lam, dlam


def chain_energy(T, J=1.0, h=0.0):
    """Energy per spin of the infinite 1D chain; -J tanh(J/T) when h = 0."""
    beta = 1.0 / np.asarray(T, dtype=float)
    lam, dlam = _chain_eigenvalue(beta, J, h)
    u = -dlam / lam
    return u[()] if np.ndim(u) == 0 else u


def chain_magnetization(T, J=1.0, h=0.0):
    """Magnetization per spin of the infinite 1D chain (zero when h = 0)."""
    beta = 1.0 / np.asarray(T, dtype=float)
    sh = np.sinh(beta * h)
    m = sh / np.sqrt(sh**2 + np.exp(-4.0 * beta * J))
    return m[()] if np.ndim(m) == 0 else m
