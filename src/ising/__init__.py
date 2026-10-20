"""Monte Carlo simulation of the nearest-neighbour Ising model in 1D and 2D."""

from . import exact
from .analysis import Estimate, Observables, autocorrelation_time, jackknife
from .lattice import energy, magnetization, ordered_spins, random_spins
from .simulate import ALGORITHMS, Equilibration, Result, equilibration, simulate

__all__ = [
    "ALGORITHMS",
    "Equilibration",
    "Estimate",
    "Observables",
    "Result",
    "autocorrelation_time",
    "energy",
    "equilibration",
    "exact",
    "jackknife",
    "magnetization",
    "ordered_spins",
    "random_spins",
    "simulate",
]
