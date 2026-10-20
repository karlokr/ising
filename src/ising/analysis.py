"""Statistical analysis of Monte Carlo time series.

Successive sweeps are correlated, so naive standard errors are too small.
Errors here come from a jackknife over contiguous blocks, which is valid as
long as each block is much longer than the integrated autocorrelation time.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class Estimate(NamedTuple):
    value: float
    error: float

    def __str__(self):
        return f"{self.value:.6g} ± {self.error:.2g}"


def jackknife(estimator, *series, n_blocks=20):
    """Value and blocked-jackknife error of ``estimator(*series)``.

    ``estimator`` maps one or more equal-length 1D arrays to a scalar.
    """
    series = [np.asarray(s, dtype=float) for s in series]
    n = len(series[0])
    size = n // n_blocks
    if size < 1:
        raise ValueError(f"need at least n_blocks={n_blocks} samples, got {n}")
    value = estimator(*series)
    trimmed = [s[: size * n_blocks] for s in series]
    keep = np.ones(size * n_blocks, dtype=bool)
    estimates = np.empty(n_blocks)
    for b in range(n_blocks):
        keep[b * size : (b + 1) * size] = False
        estimates[b] = estimator(*(s[keep] for s in trimmed))
        keep[b * size : (b + 1) * size] = True
    error = np.sqrt((n_blocks - 1) / n_blocks * np.sum((estimates - estimates.mean()) ** 2))
    return Estimate(float(value), float(error))


def autocorrelation_time(x, c=5.0):
    """Integrated autocorrelation time tau = 1/2 + sum_t rho(t), in samples.

    Uses Sokal's self-consistent window (the smallest W with W >= c * tau(W)).
    Uncorrelated data gives tau = 0.5.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 2 or not np.any(x):
        return 0.5
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    taus = 0.5 + np.cumsum(acf[1:])
    windows = np.arange(1, n)
    ok = windows >= c * taus
    W = np.argmax(ok) if np.any(ok) else n - 2
    return float(taus[W])


@dataclass(frozen=True)
class Observables:
    """Thermodynamic averages per spin, each with a jackknife error.

    The susceptibility uses <|M|> rather than <M>, the usual choice for finite
    systems in zero field where <M> averages to zero by symmetry.
    """

    T: float
    n_sites: int
    n_samples: int
    energy: Estimate  # <E> / N
    magnetization: Estimate  # <M> / N
    abs_magnetization: Estimate  # <|M|> / N
    heat_capacity: Estimate  # (<E^2> - <E>^2) / (N T^2)
    susceptibility: Estimate  # (<M^2> - <|M|>^2) / (N T)
    binder: Estimate  # 1 - <M^4> / (3 <M^2>^2)
    tau_energy: float  # integrated autocorrelation times, in sweeps
    tau_magnetization: float
    n_blocks: int = 20

    @property
    def errors_reliable(self):
        """False if jackknife blocks are shorter than 10 autocorrelation times."""
        block = self.n_samples // self.n_blocks
        return block >= 10 * max(self.tau_energy, self.tau_magnetization)

    def __str__(self):
        rows = [
            ("energy", self.energy),
            ("magnetization", self.magnetization),
            ("|magnetization|", self.abs_magnetization),
            ("heat capacity", self.heat_capacity),
            ("susceptibility", self.susceptibility),
            ("binder cumulant", self.binder),
        ]
        lines = [f"T = {self.T:g}, {self.n_sites} sites, {self.n_samples} samples"]
        lines += [f"  {name:<16} {est}" for name, est in rows]
        lines.append(
            f"  tau_int (sweeps) energy {self.tau_energy:.3g}, |M| {self.tau_magnetization:.3g}"
        )
        if not self.errors_reliable:
            lines.append(
                f"  warning: blocks of {self.n_samples // self.n_blocks} sweeps are shorter than"
                " 10 autocorrelation times, so errors are underestimated;"
                " run longer or use wolff"
            )
        return "\n".join(lines)


def thermodynamics(energy, magnetization, n_sites, T, n_blocks=20):
    """Observables from per-sweep total energy and magnetization series."""
    e = np.asarray(energy, dtype=float) / n_sites
    m = np.asarray(magnetization, dtype=float) / n_sites
    a = np.abs(m)

    def heat_capacity(e):
        return n_sites * np.var(e) / T**2

    def susceptibility(a):
        return n_sites * (np.mean(a * a) - np.mean(a) ** 2) / T

    def binder(m):
        m2 = np.mean(m * m)
        return 1.0 - np.mean(m**4) / (3.0 * m2 * m2) if m2 > 0 else 0.0

    return Observables(
        T=float(T),
        n_sites=int(n_sites),
        n_samples=len(e),
        energy=jackknife(np.mean, e, n_blocks=n_blocks),
        magnetization=jackknife(np.mean, m, n_blocks=n_blocks),
        abs_magnetization=jackknife(np.mean, a, n_blocks=n_blocks),
        heat_capacity=jackknife(heat_capacity, e, n_blocks=n_blocks),
        susceptibility=jackknife(susceptibility, a, n_blocks=n_blocks),
        binder=jackknife(binder, m, n_blocks=n_blocks),
        tau_energy=autocorrelation_time(e),
        tau_magnetization=autocorrelation_time(a),
        n_blocks=n_blocks,
    )
